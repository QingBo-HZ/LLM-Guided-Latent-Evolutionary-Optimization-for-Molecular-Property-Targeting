#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sweet gated latent-space GA, corrected 4-group version.

Task:
    Gated logSw maximization.

Four groups:
    Group A: group_a_random
        Random sweet-like seed from OOD background latent.

    Group B: group_b_dataset
        Dataset seed from SweetDB + FlavorDB sweet latent.
        Select seeds with high P_sweet, high pred_logSw, and low D_OOD.

    Group C: group_c_llm
        One-shot LLM seed from LLM-generated SMILES encoded to latent.

    Group D: group_d_llm_iterative
        LLM-guided iterative seed.
        Export feedback every N generations; inject newly encoded LLM latent if available.

GA internal score:
    score_ga = P_sweet(z) * min(pred_logSw(z), logsw_score_cap)
               - lambda_ood * D_OOD_norm(z)

Final reliable candidate score:
    decode z -> SMILES
    SMILES -> strict-BPE re-encode -> z_re
    re-score z_re
    descriptor filter
    final_score = P_sweet_re * min(pred_logSw_re, logsw_score_cap)
                  - lambda_ood * D_OOD_norm_re
                  - lambda_desc * (1 - descriptor_score)
                  + lambda_llm * llm_score_optional

Required prepared files:
    latent_dir/
        sweetdb_latent.npy
        sweetdb_labels.npy
        flavor_binary_latent.npy
        flavor_binary_labels.npy

    predictor_dir/
        latent_classifier.pt or latent_classifier_bundle.pt
        latent_classifier_scaler.pkl
        latent_regressor.pt or latent_regressor_bundle.pt
        latent_regressor_scaler_z.pkl
        latent_regressor_scaler_y.pkl

    ood_dir/
        background_latent.npy
        ood_knn.pkl
        ood_scaler.pkl
        ood_stats.json
"""

import os
import sys
import json
import time
import random
import argparse
import warnings
from types import SimpleNamespace
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

RDLogger.DisableLog("rdApp.*")


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def add_psvae_path(psvae_root):
    src_dir = os.path.join(psvae_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def canonicalize_smiles(smi):
    if smi is None or pd.isna(smi):
        return None

    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def compute_diversity(smiles_list):
    mols = []
    for smi in smiles_list:
        if smi is None or pd.isna(smi):
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            mols.append(mol)

    if len(mols) < 2:
        return 0.0

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        for m in mols
    ]

    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    if len(sims) == 0:
        return 0.0

    return 1.0 - float(np.mean(sims))


# ============================================================
# Descriptor scoring
# ============================================================

def descriptor_profile(smiles):
    can = canonicalize_smiles(smiles)

    if can is None:
        return {
            "descriptor_ok": False,
            "descriptor_score": 0.0,
            "filter_reason": "invalid_smiles",
        }

    mol = Chem.MolFromSmiles(can)
    if mol is None:
        return {
            "descriptor_ok": False,
            "descriptor_score": 0.0,
            "filter_reason": "invalid_mol",
        }

    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    heavy_atoms = mol.GetNumHeavyAtoms()
    hetero_atoms = sum(1 for a in atoms if a not in ["C", "H"])
    carbon_atoms = sum(1 for a in atoms if a == "C")

    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    logp = Crippen.MolLogP(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)

    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    aromatic_ratio = aromatic_atoms / max(heavy_atoms, 1)

    halogens = sum(1 for a in atoms if a in ["F", "Cl", "Br", "I"])
    hetero_ratio = hetero_atoms / max(heavy_atoms, 1)

    is_hydrocarbon_like = hetero_atoms <= 1 and aromatic_ratio > 0.45
    simple_halogen_aromatic = halogens >= 1 and hetero_atoms <= 2 and aromatic_ratio > 0.45
    too_hydrophobic = logp > 4.0 and tpsa < 40
    too_small_nonpolar = mw < 120 and (hba + hbd) < 3
    low_hbond_capacity = (hba + hbd) < 3
    low_polarity = tpsa < 35
    very_low_hetero = hetero_atoms < 3

    # Soft descriptor score: sweet-like molecules often need multiple polar/H-bond features.
    score = 0.0

    if 100 <= mw <= 800:
        score += 0.12
    if 35 <= tpsa <= 220:
        score += 0.18
    if hba + hbd >= 3:
        score += 0.20
    if hetero_atoms >= 3:
        score += 0.18
    if -3.0 <= logp <= 3.5:
        score += 0.12
    if heavy_atoms >= 8:
        score += 0.08
    if ring_count <= 6:
        score += 0.04
    if carbon_atoms > 0:
        score += 0.04
    if not is_hydrocarbon_like:
        score += 0.04

    risk_flags = []
    if is_hydrocarbon_like:
        risk_flags.append("hydrophobic_aromatic_or_hydrocarbon_like")
    if simple_halogen_aromatic:
        risk_flags.append("simple_halogenated_aromatic")
    if too_hydrophobic:
        risk_flags.append("too_hydrophobic")
    if too_small_nonpolar:
        risk_flags.append("too_small_nonpolar")
    if low_hbond_capacity:
        risk_flags.append("low_hbond_capacity")
    if low_polarity:
        risk_flags.append("low_tpsa")
    if very_low_hetero:
        risk_flags.append("few_hetero_atoms")

    descriptor_ok = True
    hard_reject_reasons = []

    if is_hydrocarbon_like:
        descriptor_ok = False
        hard_reject_reasons.append("hydrophobic_aromatic_or_hydrocarbon_like")

    if simple_halogen_aromatic:
        descriptor_ok = False
        hard_reject_reasons.append("simple_halogenated_aromatic")

    if hetero_atoms < 2:
        descriptor_ok = False
        hard_reject_reasons.append("hetero_atoms_less_than_2")

    if hba + hbd < 2:
        descriptor_ok = False
        hard_reject_reasons.append("hba_hbd_sum_less_than_2")

    if mw < 80:
        descriptor_ok = False
        hard_reject_reasons.append("molwt_too_low")

    if not hard_reject_reasons:
        filter_reason = "pass"
    else:
        filter_reason = ";".join(hard_reject_reasons)

    return {
        "descriptor_ok": bool(descriptor_ok),
        "descriptor_score": float(np.clip(score, 0.0, 1.0)),
        "filter_reason": filter_reason,
        "canonical_smiles": can,
        "MolWt": float(mw),
        "TPSA": float(tpsa),
        "HBD": int(hbd),
        "HBA": int(hba),
        "HBA_HBD_sum": int(hba + hbd),
        "LogP": float(logp),
        "RotB": int(rotb),
        "RingCount": int(ring_count),
        "HeavyAtoms": int(heavy_atoms),
        "HeteroAtoms": int(hetero_atoms),
        "HeteroRatio": float(hetero_ratio),
        "AromaticRatio": float(aromatic_ratio),
        "HalogenCount": int(halogens),
        "risk_flags": ";".join(risk_flags),
    }


# ============================================================
# Predictor models
# ============================================================

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=56, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class LatentRegressor(nn.Module):
    def __init__(self, latent_dim=56, hidden_dim=128, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ============================================================
# PS-VAE loading, decoding, strict re-encoding
# ============================================================

def make_psvae_config(tokenizer, args):
    from utils.nn_utils import common_config, predictor_config, encoder_config, ps_vae_config

    ns = SimpleNamespace(
        lr=args.psvae_lr,
        alpha=args.psvae_alpha,
        beta=args.psvae_beta,
        step_beta=args.psvae_step_beta,
        max_beta=args.psvae_max_beta,
        kl_warmup=args.psvae_kl_warmup,
        kl_anneal_iter=args.psvae_kl_anneal_iter,

        props=args.psvae_props,
        predictor_hidden_dim=args.psvae_predictor_hidden_dim,
        node_hidden_dim=args.psvae_node_hidden_dim,
        graph_embedding_dim=args.psvae_graph_embedding_dim,
        latent_dim=args.latent_dim,

        max_pos=args.psvae_max_pos,
        atom_embedding_dim=args.psvae_atom_embedding_dim,
        piece_embedding_dim=args.psvae_piece_embedding_dim,
        pos_embedding_dim=args.psvae_pos_embedding_dim,
        piece_hidden_dim=args.psvae_piece_hidden_dim,
    )

    vocab = tokenizer.chem_vocab

    config = {
        **common_config(ns),
        **encoder_config(ns, vocab),
        **predictor_config(ns),
    }
    config.update(ps_vae_config(ns, tokenizer))

    return config


def load_psvae(args, device):
    add_psvae_path(args.psvae_root)

    from pl_models import PSVAEModel
    from data.mol_bpe import Tokenizer

    tokenizer = Tokenizer(args.vocab)
    config = make_psvae_config(tokenizer, args)

    model = PSVAEModel(config, tokenizer)

    ckpt = safe_torch_load(args.psvae_ckpt)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("=" * 80)
    print("[PS-VAE] loaded")
    print("ckpt:", args.psvae_ckpt)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("First missing:", missing[:10])
    if len(unexpected) > 0:
        print("First unexpected:", unexpected[:10])
    print("=" * 80)

    model.to(device)
    model.eval()

    return model, tokenizer


def latent_to_smiles(model_psvae, z, device, args):
    add_psvae_path(args.psvae_root)

    try:
        from utils.chem_utils import molecule2smiles
    except Exception:
        molecule2smiles = None

    try:
        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32, device=device)

            graph = model_psvae.inference_single_z(
                z_t,
                max_atom_num=args.max_atom_num,
                add_edge_th=args.add_edge_th,
                temperature=args.temperature,
            )

            mol = model_psvae.return_data_to_mol(graph)

            if mol is None:
                return None

            if molecule2smiles is not None:
                smi = molecule2smiles(mol)
            else:
                smi = Chem.MolToSmiles(mol, canonical=True)

            return canonicalize_smiles(smi)

    except Exception:
        return None


def strict_reencode_smiles(model_psvae, tokenizer, smiles, device, args):
    """
    SMILES -> chem_utils.smiles2molecule(kekulize=True)
           -> BPEMolDataset.process_step1/2/3
           -> model.get_z(batch)

    This matches the strict-BPE latent extraction logic.
    """
    if smiles is None or pd.isna(smiles):
        return None, "empty_smiles"

    add_psvae_path(args.psvae_root)

    try:
        from data.bpe_dataset import BPEMolDataset
        from utils import chem_utils

        mol = chem_utils.smiles2molecule(str(smiles), kekulize=True)
        if mol is None:
            return None, "smiles2molecule_none"

        step1 = BPEMolDataset.process_step1(mol, tokenizer)
        step2 = BPEMolDataset.process_step2(step1, tokenizer)

        batch = BPEMolDataset.process_step3(
            [step2],
            tokenizer,
            device=device,
        )

        if isinstance(batch, dict):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

        with torch.no_grad():
            z = model_psvae.get_z_mean(batch)
            z = z.detach().cpu().numpy().astype(np.float32)

        if z.ndim == 2:
            z = z[0]

        return z, "ok"

    except Exception as e:
        reason = str(e).split("\n")[0][:200]
        return None, reason


# ============================================================
# Gated evaluator
# ============================================================

class SweetGatedEvaluator:
    def __init__(
        self,
        predictor_dir,
        ood_dir,
        latent_dim,
        device,
        lambda_ood=0.2,
        lambda_reg_uncertainty=0.0,
        ood_k=10,
        logsw_score_cap=3.5,
        objective="legacy",
        p_sweet_threshold=0.70,
        logsw_success_threshold=2.30,
        ood_threshold=None,
    ):
        self.predictor_dir = predictor_dir
        self.ood_dir = ood_dir
        self.latent_dim = int(latent_dim)
        self.device = torch.device(device)
        self.lambda_ood = float(lambda_ood)
        self.lambda_reg_uncertainty = float(lambda_reg_uncertainty)
        self.ood_k = int(ood_k)
        self.logsw_score_cap = float(logsw_score_cap)
        self.objective = str(objective)
        self.p_sweet_threshold = float(p_sweet_threshold)
        self.logsw_success_threshold = float(logsw_success_threshold)
        self.ood_threshold = None if ood_threshold is None else float(ood_threshold)

        self._load_classifier()
        self._load_regressor()
        self._load_ood()

    def _load_classifier(self):
        bundle_path = os.path.join(self.predictor_dir, "latent_classifier_bundle.pt")
        state_path = os.path.join(self.predictor_dir, "latent_classifier.pt")
        scaler_path = os.path.join(self.predictor_dir, "latent_classifier_scaler.pkl")

        hidden_dim = 128
        dropout = 0.15
        threshold = 0.5

        if os.path.exists(bundle_path):
            bundle = safe_torch_load(bundle_path)
            state_dict = bundle["state_dict"]
            hidden_dim = int(bundle.get("hidden_dim", hidden_dim))
            dropout = float(bundle.get("dropout", dropout))
            threshold = float(bundle.get("threshold", threshold))
        else:
            state_dict = safe_torch_load(state_path)

        self.classifier = LatentClassifier(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)

        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()

        self.classifier_scaler = joblib.load(scaler_path)
        self.classifier_threshold = threshold

    def _load_regressor(self):
        ensemble_path = os.path.join(self.predictor_dir, "latent_regressor_ensemble_summary.json")
        if os.path.exists(ensemble_path):
            with open(ensemble_path, "r", encoding="utf-8") as f:
                self.regressor_ensemble_summary = json.load(f)

            self.regressor_is_ensemble = True
            self.regressor_members = []
            for member in self.regressor_ensemble_summary.get("models", []):
                member_dir = os.path.join(self.predictor_dir, member["dir"])
                bundle = safe_torch_load(os.path.join(member_dir, "latent_regressor_bundle.pt"))

                hidden_dim = int(bundle.get("hidden_dim", 128))
                dropout = float(bundle.get("dropout", 0.10))
                model = LatentRegressor(
                    latent_dim=self.latent_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                ).to(self.device)
                model.load_state_dict(bundle["state_dict"])
                model.eval()

                self.regressor_members.append({
                    "fold": int(member.get("fold", len(self.regressor_members))),
                    "model": model,
                    "scaler_z": joblib.load(os.path.join(member_dir, "latent_regressor_scaler_z.pkl")),
                    "scaler_y": joblib.load(os.path.join(member_dir, "latent_regressor_scaler_y.pkl")),
                })

            if not self.regressor_members:
                raise ValueError(f"No ensemble regressor members found in {ensemble_path}")

            print(f"[Evaluator] Loaded regressor ensemble: {len(self.regressor_members)} members")
            return

        self.regressor_is_ensemble = False
        bundle_path = os.path.join(self.predictor_dir, "latent_regressor_bundle.pt")
        state_path = os.path.join(self.predictor_dir, "latent_regressor.pt")

        scaler_z_path = os.path.join(self.predictor_dir, "latent_regressor_scaler_z.pkl")
        scaler_y_path = os.path.join(self.predictor_dir, "latent_regressor_scaler_y.pkl")

        hidden_dim = 128
        dropout = 0.10

        if os.path.exists(bundle_path):
            bundle = safe_torch_load(bundle_path)
            state_dict = bundle["state_dict"]
            hidden_dim = int(bundle.get("hidden_dim", hidden_dim))
            dropout = float(bundle.get("dropout", dropout))
        else:
            state_dict = safe_torch_load(state_path)

        self.regressor = LatentRegressor(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)

        self.regressor.load_state_dict(state_dict)
        self.regressor.eval()

        self.reg_scaler_z = joblib.load(scaler_z_path)
        self.reg_scaler_y = joblib.load(scaler_y_path)

    def _load_ood(self):
        self.ood_knn = joblib.load(os.path.join(self.ood_dir, "ood_knn.pkl"))
        self.ood_scaler = joblib.load(os.path.join(self.ood_dir, "ood_scaler.pkl"))

        stats_path = os.path.join(self.ood_dir, "ood_stats.json")
        with open(stats_path, "r", encoding="utf-8") as f:
            self.ood_stats = json.load(f)

        self.ood_p50 = float(
            self.ood_stats.get("distance_p50", self.ood_stats.get("distance_mean", 1.0))
        )
        self.ood_p95 = float(
            self.ood_stats.get("distance_p95", self.ood_stats.get("distance_mean", 1.0))
        )
        self.ood_p99 = float(
            self.ood_stats.get("distance_p99", self.ood_p95)
        )

    def _mean_knn_distance(self, distances):
        out = []
        for row in distances:
            r = np.asarray(row, dtype=np.float32)

            if len(r) > 1 and r[0] < 1e-8:
                r = r[1:]

            if len(r) > self.ood_k:
                r = r[:self.ood_k]

            out.append(float(np.mean(r)))

        return np.asarray(out, dtype=np.float32)

    def evaluate(self, z):
        z = np.asarray(z, dtype=np.float32)

        if z.ndim == 1:
            z = z[None, :]

        if z.shape[1] != self.latent_dim:
            raise ValueError(f"latent dim mismatch: got {z.shape[1]}, expected {self.latent_dim}")

        z_cls = self.classifier_scaler.transform(z)
        x_cls = torch.tensor(z_cls, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.classifier(x_cls)
            p_sweet = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

        if getattr(self, "regressor_is_ensemble", False):
            member_preds = []
            with torch.no_grad():
                for member in self.regressor_members:
                    z_reg = member["scaler_z"].transform(z)
                    x_reg = torch.tensor(z_reg, dtype=torch.float32, device=self.device)
                    pred_scaled = member["model"](x_reg).detach().cpu().numpy().astype(np.float32)
                    pred_raw = member["scaler_y"].inverse_transform(
                        pred_scaled.reshape(-1, 1)
                    ).reshape(-1).astype(np.float32)
                    member_preds.append(pred_raw)
            member_preds = np.stack(member_preds, axis=0)
            pred_logsw_mean = member_preds.mean(axis=0).astype(np.float32)
            pred_logsw_std = member_preds.std(axis=0).astype(np.float32)
            pred_logsw = (pred_logsw_mean - self.lambda_reg_uncertainty * pred_logsw_std).astype(np.float32)
        else:
            z_reg = self.reg_scaler_z.transform(z)
            x_reg = torch.tensor(z_reg, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                pred_scaled = self.regressor(x_reg).detach().cpu().numpy().astype(np.float32)

            pred_logsw_mean = self.reg_scaler_y.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).reshape(-1).astype(np.float32)
            pred_logsw_std = np.zeros_like(pred_logsw_mean, dtype=np.float32)
            pred_logsw = pred_logsw_mean

        z_ood = self.ood_scaler.transform(z)
        distances, _ = self.ood_knn.kneighbors(z_ood)
        d_ood = self._mean_knn_distance(distances)

        denom = max(self.ood_p95 - self.ood_p50, 1e-8)
        d_ood_norm = (d_ood - self.ood_p50) / denom
        d_ood_norm = np.maximum(d_ood_norm, 0.0).astype(np.float32)

        pred_logsw_clipped = np.minimum(pred_logsw, self.logsw_score_cap).astype(np.float32)

        if self.objective == "constrained_sweetness":
            ood_threshold = self.ood_p95 if self.ood_threshold is None else self.ood_threshold
            ood_penalty = np.maximum(0.0, d_ood - ood_threshold) / max(ood_threshold, 1e-8)
            p_penalty = np.maximum(0.0, self.p_sweet_threshold - p_sweet)
            score_ga = (
                pred_logsw
                + 0.50 * p_sweet
                - 0.80 * p_penalty
                - 0.50 * ood_penalty
            )
        else:
            score_ga = p_sweet * pred_logsw_clipped - self.lambda_ood * d_ood_norm

        return {
            "p_sweet": p_sweet.astype(np.float32),
            "pred_logsw": pred_logsw.astype(np.float32),
            "pred_logsw_mean": pred_logsw_mean.astype(np.float32),
            "pred_logsw_std": pred_logsw_std.astype(np.float32),
            "pred_logsw_clipped": pred_logsw_clipped.astype(np.float32),
            "d_ood": d_ood.astype(np.float32),
            "d_ood_norm": d_ood_norm.astype(np.float32),
            "score_ga": score_ga.astype(np.float32),
        }


# ============================================================
# GA operations
# ============================================================

def sample_population_from_pool(pool, pop_size, replace=True, augment_sigma=0.0):
    pool = np.asarray(pool, dtype=np.float32)

    if len(pool) == 0:
        raise ValueError("Empty latent seed pool.")

    if len(pool) >= pop_size:
        idx = np.random.choice(len(pool), pop_size, replace=False)
        return pool[idx].copy()

    if augment_sigma and augment_sigma > 0:
        extra = pool[np.random.choice(len(pool), pop_size - len(pool), replace=True)].copy()
        extra += np.random.normal(0.0, float(augment_sigma), size=extra.shape).astype(np.float32)
        return np.concatenate([pool, extra], axis=0).astype(np.float32)

    if replace:
        extra = pool[np.random.choice(len(pool), pop_size - len(pool), replace=True)]
        return np.concatenate([pool, extra], axis=0).astype(np.float32)

    raise ValueError(f"Pool size {len(pool)} < pop_size {pop_size}.")


def tournament_selection_max(pop, fitness, tourn_size=3):
    idx = np.random.choice(len(pop), tourn_size, replace=False)
    best = idx[np.argmax(fitness[idx])]
    return pop[best].copy()


def arithmetic_crossover(p1, p2, cross_prob, lb, ub):
    if np.random.random() > cross_prob:
        return p1.copy(), p2.copy()

    alpha = np.random.random()
    c1 = alpha * p1 + (1.0 - alpha) * p2
    c2 = (1.0 - alpha) * p1 + alpha * p2

    c1 = np.clip(c1, lb, ub).astype(np.float32)
    c2 = np.clip(c2, lb, ub).astype(np.float32)

    return c1, c2


def gaussian_mutation(individual, mut_prob, sigma_vec, lb, ub):
    x = individual.copy().astype(np.float32)

    mask = np.random.random(size=x.shape) < mut_prob
    noise = np.random.normal(0.0, sigma_vec, size=x.shape).astype(np.float32)

    x[mask] += noise[mask]
    x = np.clip(x, lb, ub)

    return x.astype(np.float32)


def farthest_point_sample(pool, n_select, seed=2026):
    pool = np.asarray(pool, dtype=np.float32)
    n = len(pool)

    if n <= n_select:
        return pool.copy()

    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, n))]
    dist = np.linalg.norm(pool - pool[selected[0]], axis=1)

    for _ in range(1, n_select):
        idx = int(np.argmax(dist))
        selected.append(idx)
        new_dist = np.linalg.norm(pool - pool[idx], axis=1)
        dist = np.minimum(dist, new_dist)

    return pool[selected].copy()


def select_top_by_score(pool, evaluator, top_n):
    metrics = evaluator.evaluate(pool)
    score = metrics["score_ga"]

    idx = np.argsort(score)[::-1]
    idx = idx[:min(top_n, len(idx))]

    return pool[idx].copy(), metrics, idx


def filter_pool_by_gate(pool, evaluator, min_p_sweet, max_ood):
    metrics = evaluator.evaluate(pool)

    mask = (
        (metrics["p_sweet"] >= float(min_p_sweet))
        & (metrics["d_ood"] <= float(max_ood))
    )

    return pool[mask], metrics, mask


def build_initial_population(args, evaluator, background_latent):
    """
    Correct group definition:

    Group A: random sweet-like seed
        Random sample from OOD background, optionally gate-filtered.

    Group B: dataset seed
        SweetDB + FlavorDB sweet seed.
        High P_sweet, high pred_logSw, low OOD.

    Group C: LLM seed
        One-shot LLM latent.

    Group D: LLM-guided iterative seed
        Initial LLM latent + iterative injection.
    """

    pop_size = args.pop_size

    if args.init_mode == "group_a_random":
        pool = background_latent

        if args.background_filter_by_evaluator:
            filtered, _, _ = filter_pool_by_gate(
                pool=background_latent,
                evaluator=evaluator,
                min_p_sweet=args.seed_min_p_sweet,
                max_ood=evaluator.ood_p95,
            )

            if len(filtered) >= max(10, pop_size // 2):
                pool = filtered
                print(f"[Group A] filtered OOD background pool: {len(pool)}")
            else:
                print(
                    f"[Group A] filtered pool too small ({len(filtered)}), "
                    f"fallback to full background: {len(pool)}"
                )

        population = sample_population_from_pool(
            pool,
            pop_size,
            replace=False if len(pool) >= pop_size else True,
        )
        source_labels = np.asarray(["initial_random"] * len(population), dtype=object)
        seed_source = "Group_A_random_sweet_like_seed_from_OOD_background"

    elif args.init_mode == "group_b_dataset":
        candidate_pools = []

        sweet_latent_path = os.path.join(args.latent_dir, "sweetdb_latent.npy")
        if os.path.exists(sweet_latent_path):
            sweet_latent = np.load(sweet_latent_path).astype(np.float32)
            if args.dataset_sweetdb_high_potency:
                sweet_csv_path = os.path.join(
                    args.latent_dir, "sweetdb_regression_dataset_aligned.csv"
                )
                assignments_path = args.dataset_scaffold_assignments
                if assignments_path is None:
                    assignments_path = os.path.join(
                        args.predictor_dir, "regressor_scaffold_assignments.csv"
                    )
                sweet_frame = pd.read_csv(sweet_csv_path)
                assignments = pd.read_csv(assignments_path)
                if len(sweet_frame) != len(sweet_latent):
                    raise ValueError(
                        "SweetDB CSV and latent rows are not aligned: "
                        f"{len(sweet_frame)} != {len(sweet_latent)}"
                    )
                required = {"index", "scaffold", "fold", "logSw"}
                if not required.issubset(assignments.columns):
                    raise ValueError(
                        f"Missing columns in scaffold assignments: "
                        f"{sorted(required - set(assignments.columns))}"
                    )
                assignments = assignments.sort_values("index").reset_index(drop=True)
                if len(assignments) != len(sweet_latent):
                    raise ValueError(
                        "Scaffold assignments and latent rows are not aligned: "
                        f"{len(assignments)} != {len(sweet_latent)}"
                    )

                eligible = assignments[
                    assignments["fold"].astype(int) != args.dataset_holdout_fold
                ].copy()
                cutoff = float(
                    eligible["logSw"].quantile(args.dataset_logsw_quantile)
                )
                eligible = eligible[eligible["logSw"] >= cutoff].copy()
                rng = np.random.default_rng(args.seed)

                # Sample one member per scaffold, then sample scaffolds uniformly.
                # This keeps Group B high-potency and diverse without turning it
                # into a top-labelled-molecule oracle.
                representatives = []
                for _, scaffold_frame in eligible.groupby("scaffold", sort=True):
                    choice = int(rng.integers(0, len(scaffold_frame)))
                    representatives.append(scaffold_frame.iloc[choice])
                diverse = pd.DataFrame(representatives)
                if len(diverse) > args.seed_pool_size:
                    take = rng.choice(
                        len(diverse), args.seed_pool_size, replace=False
                    )
                    selected = diverse.iloc[take].copy()
                else:
                    selected = diverse.copy()
                if len(selected) < min(args.seed_pool_size, len(eligible)):
                    remaining = eligible[~eligible["index"].isin(selected["index"])]
                    take_n = min(
                        args.seed_pool_size - len(selected), len(remaining)
                    )
                    if take_n > 0:
                        take = rng.choice(len(remaining), take_n, replace=False)
                        selected = pd.concat(
                            [selected, remaining.iloc[take]], ignore_index=True
                        )
                selected_idx = selected["index"].to_numpy(dtype=int)
                candidate_pools.append(sweet_latent[selected_idx])
                print(
                    "[Group B] SweetDB high-potency training pool: "
                    f"holdout_fold={args.dataset_holdout_fold}, "
                    f"quantile={args.dataset_logsw_quantile:.2f}, "
                    f"cutoff={cutoff:.3f}, candidates={len(eligible)}, "
                    f"selected={len(selected)}, "
                    f"unique_scaffolds={selected['scaffold'].nunique()}"
                )
            else:
                candidate_pools.append(sweet_latent)

        if not args.dataset_sweetdb_only:
            flavor_latent_path = os.path.join(args.latent_dir, "flavor_binary_latent.npy")
            flavor_label_path = os.path.join(args.latent_dir, "flavor_binary_labels.npy")
            if os.path.exists(flavor_latent_path) and os.path.exists(flavor_label_path):
                flavor_latent = np.load(flavor_latent_path).astype(np.float32)
                flavor_labels = np.load(flavor_label_path).astype(int)
                sweet_flavor = flavor_latent[flavor_labels == 1]
                if len(sweet_flavor) > 0:
                    candidate_pools.append(sweet_flavor)

        if len(candidate_pools) == 0:
            raise FileNotFoundError("No SweetDB or FlavorDB sweet latent found for Group B.")

        pool = np.concatenate(candidate_pools, axis=0)

        if args.dataset_filter_by_evaluator:
            metrics = evaluator.evaluate(pool)
            mask = (
                (metrics["p_sweet"] >= args.seed_min_p_sweet)
                & (metrics["d_ood"] <= evaluator.ood_p95)
            )
            filtered = pool[mask]
            if len(filtered) >= max(10, pop_size // 2):
                pool = filtered
                print(f"[Group B] filtered dataset seed pool: {len(pool)}")

        if args.dataset_rank_by_fitness:
            top_pool, _, _ = select_top_by_score(
                pool=pool,
                evaluator=evaluator,
                top_n=min(args.seed_pool_size, len(pool)),
            )
        else:
            top_pool = pool

        if args.diverse_seed:
            top_pool = farthest_point_sample(
                top_pool,
                min(len(top_pool), args.seed_pool_size),
                seed=args.seed,
            )

        population = sample_population_from_pool(
            top_pool,
            pop_size,
            replace=len(top_pool) < pop_size,
            augment_sigma=args.seed_augment_sigma,
        )
        source_labels = np.asarray(["initial_dataset"] * len(population), dtype=object)
        if args.dataset_sweetdb_high_potency:
            seed_source = (
                "Group_B_SweetDB_training_high_potency_scaffold_diverse_seed"
            )
        else:
            seed_source = "Group_B_dataset_seed_SweetDB_and_FlavorDB_sweet"

    elif args.init_mode == "group_c_llm":
        if args.llm_latent_path is None or not os.path.exists(args.llm_latent_path):
            raise FileNotFoundError("--llm_latent_path is required for Group C.")

        llm_latent = np.load(args.llm_latent_path).astype(np.float32)

        # The LLM seed pool is ordered by the LLM-prior judge before encoding.
        # Do not pre-filter it with the downstream GA fitness; fitness is only a search compass.
        top_pool = llm_latent[:min(args.seed_pool_size, len(llm_latent))]

        population = sample_population_from_pool(
            top_pool,
            pop_size,
            replace=len(top_pool) < pop_size,
            augment_sigma=args.seed_augment_sigma,
        )
        source_labels = np.asarray(["initial_llm_seed"] * len(population), dtype=object)
        seed_source = "Group_C_iterative_LLM_prior_seed"

    elif args.init_mode == "group_d_llm_iterative":
        if args.llm_latent_path is None or not os.path.exists(args.llm_latent_path):
            raise FileNotFoundError("--llm_latent_path is required as initial LLM seed for Group D.")

        llm_latent = np.load(args.llm_latent_path).astype(np.float32)

        top_pool = llm_latent[:min(args.seed_pool_size, len(llm_latent))]

        population = sample_population_from_pool(
            top_pool,
            pop_size,
            replace=len(top_pool) < pop_size,
            augment_sigma=args.seed_augment_sigma,
        )
        source_labels = np.asarray(["initial_llm_seed"] * len(population), dtype=object)
        seed_source = "Group_D_LLM_prior_seed_with_online_reflection"

    else:
        raise ValueError(f"Unknown init_mode: {args.init_mode}")

    return population.astype(np.float32), seed_source, source_labels


# ============================================================
# Decode / re-encode / final scoring
# ============================================================

def decode_population(model_psvae, population, device, args):
    smiles = []
    valid = []

    for i, z in enumerate(population):
        smi = latent_to_smiles(
            model_psvae=model_psvae,
            z=z,
            device=device,
            args=args,
        )

        if smi is None:
            smiles.append(None)
            valid.append(False)
        else:
            smiles.append(smi)
            valid.append(True)

        if (i + 1) % 50 == 0:
            print(f"Decoded {i + 1}/{len(population)}")

    return smiles, np.asarray(valid, dtype=bool)


def load_optional_llm_score_csv(path):
    """
    Optional CSV columns:
        canonical_smiles,llm_score,llm_decision,llm_reason
    """
    if path is None or not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    out = {}

    for _, row in df.iterrows():
        smi = canonicalize_smiles(row.get("canonical_smiles", row.get("smiles", None)))
        if smi is None:
            continue

        out[smi] = {
            "llm_score": float(row.get("llm_score", row.get("sweetness_plausibility", 0.0))),
            "llm_decision": str(row.get("llm_decision", row.get("decision", ""))),
            "llm_reason": str(row.get("llm_reason", row.get("short_reason", ""))),
        }

    return out


def final_rescore_candidates(
    df,
    population,
    model_psvae,
    tokenizer,
    evaluator,
    device,
    args,
    llm_score_map=None,
):
    if llm_score_map is None:
        llm_score_map = {}

    records = []

    for _, row in df.iterrows():
        idx = int(row["idx"])
        smiles = row.get("smiles", None)
        valid = bool(row.get("valid", False))

        desc = descriptor_profile(smiles)

        reencode_ok = False
        reencode_reason = "not_attempted"

        p_sweet_re = np.nan
        pred_logsw_re = np.nan
        pred_logsw_mean_re = np.nan
        pred_logsw_std_re = np.nan
        pred_logsw_clip_re = np.nan
        d_ood_re = np.nan
        d_ood_norm_re = np.nan
        score_reencoded = np.nan

        if valid and desc.get("canonical_smiles", None) is not None:
            z_re, reencode_reason = strict_reencode_smiles(
                model_psvae=model_psvae,
                tokenizer=tokenizer,
                smiles=desc["canonical_smiles"],
                device=device,
                args=args,
            )

            if z_re is not None:
                reencode_ok = True
                m_re = evaluator.evaluate(z_re)

                p_sweet_re = float(m_re["p_sweet"][0])
                pred_logsw_re = float(m_re["pred_logsw"][0])
                pred_logsw_mean_re = float(m_re.get("pred_logsw_mean", m_re["pred_logsw"])[0])
                pred_logsw_std_re = float(m_re.get("pred_logsw_std", np.zeros_like(m_re["pred_logsw"]))[0])
                pred_logsw_clip_re = float(m_re["pred_logsw_clipped"][0])
                d_ood_re = float(m_re["d_ood"][0])
                d_ood_norm_re = float(m_re["d_ood_norm"][0])
                score_reencoded = float(m_re["score_ga"][0])

        can = desc.get("canonical_smiles", None)
        llm_info = llm_score_map.get(can, {})
        llm_score = float(llm_info.get("llm_score", 0.0))
        llm_decision = str(llm_info.get("llm_decision", "not_used"))
        llm_reason = str(llm_info.get("llm_reason", ""))

        if reencode_ok:
            base_score = score_reencoded
        else:
            base_score = -999.0

        descriptor_score = float(desc.get("descriptor_score", 0.0))

        final_score = (
            base_score
            - args.lambda_desc * (1.0 - descriptor_score)
            + args.lambda_llm * llm_score
        )

        reliable = True
        reliable_reasons = []

        if not valid:
            reliable = False
            reliable_reasons.append("decode_invalid")

        if not reencode_ok:
            reliable = False
            reliable_reasons.append("reencode_failed")

        if not desc.get("descriptor_ok", False):
            reliable = False
            reliable_reasons.append("descriptor_filter_failed")

        if reencode_ok and p_sweet_re < args.success_p_sweet:
            reliable = False
            reliable_reasons.append("p_sweet_reencoded_low")

        if reencode_ok and pred_logsw_re < args.final_min_logsw:
            reliable = False
            reliable_reasons.append("pred_logsw_reencoded_low")

        if reencode_ok and d_ood_re > evaluator.ood_p95:
            reliable = False
            reliable_reasons.append("d_ood_reencoded_high")

        if args.use_llm_filter and llm_decision.lower() == "reject":
            reliable = False
            reliable_reasons.append("llm_reject")

        rec = dict(row)

        rec.update({
            "canonical_smiles": can,
            "descriptor_ok": bool(desc.get("descriptor_ok", False)),
            "descriptor_score": descriptor_score,
            "descriptor_filter_reason": desc.get("filter_reason", ""),
            "risk_flags": desc.get("risk_flags", ""),

            "MolWt": desc.get("MolWt", np.nan),
            "TPSA": desc.get("TPSA", np.nan),
            "HBD": desc.get("HBD", np.nan),
            "HBA": desc.get("HBA", np.nan),
            "HBA_HBD_sum": desc.get("HBA_HBD_sum", np.nan),
            "LogP": desc.get("LogP", np.nan),
            "RotB": desc.get("RotB", np.nan),
            "RingCount": desc.get("RingCount", np.nan),
            "HeavyAtoms": desc.get("HeavyAtoms", np.nan),
            "HeteroAtoms": desc.get("HeteroAtoms", np.nan),
            "AromaticRatio": desc.get("AromaticRatio", np.nan),
            "HalogenCount": desc.get("HalogenCount", np.nan),

            "reencode_ok": bool(reencode_ok),
            "reencode_reason": reencode_reason,

            "p_sweet_reencoded": p_sweet_re,
            "pred_logsw_reencoded": pred_logsw_re,
            "pred_logsw_mean_reencoded": pred_logsw_mean_re,
            "pred_logsw_std_reencoded": pred_logsw_std_re,
            "pred_logsw_clipped_reencoded": pred_logsw_clip_re,
            "d_ood_reencoded": d_ood_re,
            "d_ood_norm_reencoded": d_ood_norm_re,
            "score_reencoded": score_reencoded,

            "llm_score": llm_score,
            "llm_decision": llm_decision,
            "llm_reason": llm_reason,

            "final_score": float(final_score),
            "reliable_candidate": bool(reliable),
            "reliable_filter_reason": "pass" if reliable else ";".join(reliable_reasons),
        })

        records.append(rec)

    out = pd.DataFrame(records)
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["final_rank"] = np.arange(1, len(out) + 1)

    return out


# ============================================================
# Group D iterative LLM functions
# ============================================================

def export_llm_feedback(out_dir, generation, archive_df, top_n=20):
    if archive_df is None or len(archive_df) == 0:
        return None

    gen_df = archive_df[archive_df["generation"] == generation].copy()
    if len(gen_df) == 0:
        return None

    gen_df = gen_df.sort_values("score_ga", ascending=False).head(top_n)

    feedback_path = os.path.join(out_dir, f"llm_feedback_gen{generation:03d}.csv")
    gen_df.to_csv(feedback_path, index=False)

    return feedback_path


def maybe_load_iterative_llm_latent(args, generation):
    """
    For Group D:
        after LLM receives feedback, encode its new SMILES and save as:
        {llm_iterative_latent_dir}/llm_gen_005.npy
        {llm_iterative_latent_dir}/llm_gen_010.npy
    """
    if args.llm_iterative_latent_dir is None:
        return None, None

    path = os.path.join(args.llm_iterative_latent_dir, f"llm_gen_{generation:03d}.npy")

    if not os.path.exists(path):
        return None, path

    arr = np.load(path).astype(np.float32)

    if arr.ndim != 2 or arr.shape[1] != args.latent_dim:
        raise ValueError(f"Invalid iterative LLM latent shape at {path}: {arr.shape}")

    return arr, path


def inject_new_seeds(population, new_latent, evaluator, inject_ratio=0.25):
    if new_latent is None or len(new_latent) == 0:
        return population

    pop_size = len(population)
    n_inject = max(1, int(pop_size * inject_ratio))

    top_new, _, _ = select_top_by_score(
        pool=new_latent,
        evaluator=evaluator,
        top_n=min(n_inject, len(new_latent)),
    )

    metrics = evaluator.evaluate(population)
    score = metrics["score_ga"]

    sorted_idx = np.argsort(score)  # low score first
    replace_idx = sorted_idx[:len(top_new)]

    population_new = population.copy()
    population_new[replace_idx] = top_new

    return population_new.astype(np.float32)


def select_unique_smiles_elites(
    population,
    population_source,
    sorted_idx,
    model_psvae,
    device,
    args,
):
    if not args.enforce_unique_smiles:
        elite_idx = sorted_idx[:args.elite_size]
        return population[elite_idx].copy(), list(population_source[elite_idx]), {
            "elite_unique_smiles": None,
            "elite_duplicates_skipped": 0,
        }

    candidate_idx = sorted_idx[:min(len(sorted_idx), max(args.elite_size, args.unique_elite_candidates))]
    candidate_smiles, candidate_valid = decode_population(
        model_psvae,
        population[candidate_idx],
        device,
        args,
    )

    selected_idx = []
    seen = set()
    duplicate_skips = 0

    for local_i, idx in enumerate(candidate_idx):
        cano = canonicalize_smiles(candidate_smiles[local_i]) if candidate_valid[local_i] else None
        if cano is None:
            continue
        if cano in seen:
            duplicate_skips += 1
            continue
        seen.add(cano)
        selected_idx.append(idx)
        if len(selected_idx) >= args.elite_size:
            break

    if len(selected_idx) < args.elite_size:
        for idx in sorted_idx:
            if idx in selected_idx:
                continue
            selected_idx.append(idx)
            if len(selected_idx) >= args.elite_size:
                break

    elite_idx = np.asarray(selected_idx[:args.elite_size], dtype=int)
    return population[elite_idx].copy(), list(population_source[elite_idx]), {
        "elite_unique_smiles": int(len(seen)),
        "elite_duplicates_skipped": int(duplicate_skips),
    }


def enforce_unique_smiles_population(
    population,
    population_source,
    model_psvae,
    evaluator,
    device,
    args,
    lb,
    ub,
    background_latent,
    llm_seed_latent=None,
):
    if not args.enforce_unique_smiles:
        return population, population_source, None

    population = np.asarray(population, dtype=np.float32).copy()
    population_source = np.asarray(population_source, dtype=object).copy()

    smiles, valid = decode_population(model_psvae, population, device, args)
    seen = set()
    keep_idx = []
    duplicate_idx = []

    for i, smi in enumerate(smiles):
        cano = canonicalize_smiles(smi) if valid[i] else None
        if cano is None or cano in seen:
            duplicate_idx.append(i)
            continue
        seen.add(cano)
        keep_idx.append(i)

    target_unique = int(np.ceil(len(population) * args.unique_target_ratio))
    target_unique = max(1, min(len(population), target_unique))

    if len(duplicate_idx) == 0 or len(seen) >= target_unique:
        return population, population_source, {
            "duplicates_before_refill": int(len(duplicate_idx)),
            "refilled": 0,
            "unique_after_refill": int(len(seen)),
            "target_unique": int(target_unique),
        }

    seed_pool = None
    if args.unique_refill_pool == "group_seed":
        if llm_seed_latent is not None and len(llm_seed_latent) > 0:
            seed_pool = np.asarray(llm_seed_latent, dtype=np.float32)
        elif background_latent is not None and len(background_latent) > 0:
            seed_pool = np.asarray(background_latent, dtype=np.float32)
    elif args.unique_refill_pool == "background":
        if background_latent is not None and len(background_latent) > 0:
            seed_pool = np.asarray(background_latent, dtype=np.float32)

    refilled = 0
    quality_rejected = 0
    for dup_i in duplicate_idx:
        if len(seen) >= target_unique:
            break

        trial_latents = []
        for _ in range(args.unique_refill_attempts):
            use_seed = (
                seed_pool is not None
                and np.random.random() < args.unique_refill_from_seed_prob
            )
            if use_seed:
                z = seed_pool[np.random.randint(0, len(seed_pool))].copy()
                z += np.random.normal(0.0, args.unique_refill_sigma, size=z.shape).astype(np.float32)
            elif keep_idx:
                z = population[np.random.choice(keep_idx)].copy()
                z += np.random.normal(0.0, args.unique_refill_sigma, size=z.shape).astype(np.float32)
            else:
                z = population[np.random.randint(0, len(population))].copy()
                z += np.random.normal(0.0, args.unique_refill_sigma, size=z.shape).astype(np.float32)

            z = np.clip(z, lb, ub).astype(np.float32)
            trial_latents.append(z)

        trial_latents = np.asarray(trial_latents, dtype=np.float32)
        trial_metrics = evaluator.evaluate(trial_latents)
        quality_mask = (
            (trial_metrics["p_sweet"] >= args.unique_refill_min_p_sweet)
            & (trial_metrics["pred_logsw"] >= args.unique_refill_min_logsw)
            & (
                trial_metrics["d_ood"]
                <= evaluator.ood_p95 * args.unique_refill_max_ood_ratio
            )
        )
        quality_rejected += int(np.sum(~quality_mask))

        quality_idx = np.flatnonzero(quality_mask)
        if len(quality_idx) == 0:
            population_source[dup_i] = f"{population_source[dup_i]}_duplicate_kept"
            continue

        ranked_idx = quality_idx[
            np.argsort(trial_metrics["score_ga"][quality_idx])[::-1]
        ]
        ranked_latents = trial_latents[ranked_idx]
        candidate_smiles, candidate_valid = decode_population(
            model_psvae,
            ranked_latents,
            device,
            args,
        )

        accepted = False
        for candidate_i, z in enumerate(ranked_latents):
            cano = (
                canonicalize_smiles(candidate_smiles[candidate_i])
                if candidate_valid[candidate_i]
                else None
            )
            if cano is None or cano in seen:
                continue

            population[dup_i] = z
            population_source[dup_i] = "unique_smiles_refill"
            seen.add(cano)
            keep_idx.append(dup_i)
            refilled += 1
            accepted = True
            break

        if not accepted:
            population_source[dup_i] = f"{population_source[dup_i]}_duplicate_kept"

    return population.astype(np.float32), population_source, {
        "duplicates_before_refill": int(len(duplicate_idx)),
        "refilled": int(refilled),
        "quality_rejected": int(quality_rejected),
        "unique_after_refill": int(len(seen)),
        "target_unique": int(target_unique),
    }


def llm_online_reflection_injection(
    generation,
    population,
    metrics,
    sorted_idx,
    model_psvae,
    tokenizer,
    device,
    args,
    out_dir,
    existing_smiles=None,
):
    if not args.llm_online_enable:
        return None, []

    gen_dir = os.path.join(out_dir, "llm_online_injection")
    ensure_dir(gen_dir)

    helper_dir = args.llm_helper_dir
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)

    try:
        from generate_llm_sweet_seed_gpt55_gemini31_v3 import (
            build_client,
            call_chat_json,
            extract_smiles_list,
            rdkit_basic_gate,
            judge_candidates_batched,
            GENERATOR_SYSTEM_PROMPT,
        )
    except Exception as e:
        print(f"[Group D] online LLM helper import failed: {e}")
        return None, []

    top_idx = sorted_idx[:min(args.llm_feedback_topn, len(sorted_idx))]
    bottom_idx = sorted_idx[-min(args.llm_feedback_topn, len(sorted_idx)):]

    print(f"[Group D] decoding reflection examples at gen {generation}...")
    top_smiles, top_valid = decode_population(model_psvae, population[top_idx], device, args)
    bottom_smiles, bottom_valid = decode_population(model_psvae, population[bottom_idx], device, args)

    top_lines = []
    for rank, idx in enumerate(top_idx):
        smi = canonicalize_smiles(top_smiles[rank]) if top_valid[rank] else None
        if smi is None:
            continue
        top_lines.append(
            f"- {smi} | score={float(metrics['score_ga'][idx]):.3f} | "
            f"logSw={float(metrics['pred_logsw'][idx]):.3f} | "
            f"P={float(metrics['p_sweet'][idx]):.3f} | OOD={float(metrics['d_ood'][idx]):.3f}"
        )

    fail_lines = []
    for rank, idx in enumerate(bottom_idx):
        smi = canonicalize_smiles(bottom_smiles[rank]) if bottom_valid[rank] else None
        reason = []
        if float(metrics["p_sweet"][idx]) < args.success_p_sweet:
            reason.append("low_P_sweet")
        if float(metrics["pred_logsw"][idx]) < args.final_min_logsw:
            reason.append("low_logSw")
        if float(metrics["d_ood"][idx]) > metrics.get("success_ood", np.inf):
            reason.append("high_OOD")
        reason_txt = ",".join(reason) if reason else "low_internal_score"
        if smi is not None:
            fail_lines.append(
                f"- {smi} | {reason_txt} | score={float(metrics['score_ga'][idx]):.3f} | "
                f"logSw={float(metrics['pred_logsw'][idx]):.3f} | "
                f"P={float(metrics['p_sweet'][idx]):.3f} | OOD={float(metrics['d_ood'][idx]):.3f}"
            )
        else:
            fail_lines.append(f"- decode_failed | {reason_txt}")

    prompt = f"""
Task:
Reflect on the current latent-GA population and generate fresh sweetener-like SMILES for injection.

Use these current successful molecules as direction, but do not copy them:
{chr(10).join(top_lines[:12]) if top_lines else "- no valid decoded top molecules"}

Avoid these weak/failing or repetitive patterns:
{chr(10).join(fail_lines[:12]) if fail_lines else "- no decoded failures"}

Design requirements:
- Generate exactly {args.llm_online_candidates} unique, chemically plausible SMILES.
- They are seed molecules for latent-space GA, not final claimed sweeteners.
- Prefer diverse scaffold families and avoid near-duplicates.
- Allowed atoms: C, H, N, O, S, P, F, Cl, Br, I.
- Avoid salts, disconnected fragments, radicals, isotopes, metals, and obvious reactive structures.

Return JSON only:
{{"smiles": ["SMILES_1", "SMILES_2"]}}
""".strip()

    generator_client = build_client(
        args.llm_generator_base_url or args.llm_base_url,
        args.llm_generator_api_key or args.llm_api_key,
        env_prefix="OPENAI",
    )
    judge_client = build_client(
        args.llm_judge_base_url or args.llm_base_url,
        args.llm_judge_api_key or args.llm_api_key,
        env_prefix="GEMINI",
    )
    obj, raw_text, err = call_chat_json(
        client=generator_client,
        model=args.llm_generator_model,
        system_prompt=GENERATOR_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=args.llm_generator_temperature,
        max_tokens=args.llm_generator_max_tokens,
    )

    raw_log_path = os.path.join(gen_dir, f"gen_{generation:03d}_generation.json")
    save_json({"prompt": prompt, "raw_text": raw_text, "error": err}, raw_log_path)

    if obj is None:
        print(f"[Group D] LLM generation JSON failed at gen {generation}: {err}")
        return None, []

    existing = set()
    if existing_smiles:
        existing = {canonicalize_smiles(x) for x in existing_smiles if canonicalize_smiles(x) is not None}
    existing.update(
        canonicalize_smiles(x)
        for x, is_valid in zip(top_smiles, top_valid)
        if is_valid and canonicalize_smiles(x) is not None
    )
    existing.update(
        canonicalize_smiles(x)
        for x, is_valid in zip(bottom_smiles, bottom_valid)
        if is_valid and canonicalize_smiles(x) is not None
    )

    accepted = []
    seen = set()
    rejected = []
    for raw_smi in extract_smiles_list(obj):
        ok, cano, reason, meta = rdkit_basic_gate(raw_smi)
        if not ok:
            rejected.append({"raw_smiles": raw_smi, "reason": reason})
            continue
        if cano in seen or cano in existing:
            rejected.append({"raw_smiles": raw_smi, "canonical_smiles": cano, "reason": "duplicate"})
            continue
        seen.add(cano)
        accepted.append({
            "candidate_id": len(accepted),
            "raw_smiles": raw_smi,
            "smiles": cano,
            "source_generation": generation,
            "basic_gate_reason": "ok",
            **meta,
        })

    pd.DataFrame(accepted).to_csv(os.path.join(gen_dir, f"gen_{generation:03d}_accepted.csv"), index=False)
    pd.DataFrame(rejected).to_csv(os.path.join(gen_dir, f"gen_{generation:03d}_rejected.csv"), index=False)

    if len(accepted) == 0:
        print(f"[Group D] no basic-gate accepted LLM candidates at gen {generation}.")
        return None, []

    judged, judge_logs = judge_candidates_batched(
        client=judge_client,
        judge_model=args.llm_judge_model,
        candidates=accepted,
        temperature=args.llm_judge_temperature,
        max_tokens=args.llm_judge_max_tokens,
        batch_size=args.llm_judge_batch_size,
    )

    judged = [
        row for row in judged
        if row.get("llm_prior_score", 0.0) >= args.llm_min_prior
        and row.get("llm_risk_penalty", 1.0) <= args.llm_max_risk
    ]
    judged = judged[:args.llm_judge_keep]

    pd.DataFrame(judged).to_csv(os.path.join(gen_dir, f"gen_{generation:03d}_judged_selected.csv"), index=False)
    with open(os.path.join(gen_dir, f"gen_{generation:03d}_judge_logs.jsonl"), "w", encoding="utf-8") as f:
        for row in judge_logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(judged) == 0:
        print(f"[Group D] no LLM candidates passed judge thresholds at gen {generation}.")
        return None, []

    latents = []
    encoded_rows = []
    for row in judged:
        z, reason = strict_reencode_smiles(
            model_psvae=model_psvae,
            tokenizer=tokenizer,
            smiles=row["smiles"],
            device=device,
            args=args,
        )
        encoded_row = dict(row)
        encoded_row["strict_bpe_reason"] = reason
        encoded_row["strict_bpe_ok"] = z is not None
        if z is not None:
            latents.append(z.reshape(-1).astype(np.float32))
        encoded_rows.append(encoded_row)

    pd.DataFrame(encoded_rows).to_csv(os.path.join(gen_dir, f"gen_{generation:03d}_encoded.csv"), index=False)

    if len(latents) == 0:
        print(f"[Group D] no LLM candidates strict-BPE encoded at gen {generation}.")
        return None, encoded_rows

    arr = np.asarray(latents, dtype=np.float32)
    np.save(os.path.join(gen_dir, f"gen_{generation:03d}_latent.npy"), arr)
    print(f"[Group D] online LLM injection ready: {arr.shape[0]} latents at gen {generation}")
    return arr, encoded_rows


def locally_refine_llm_latents(
    seed_latents,
    current_population,
    evaluator,
    ref_std,
    lb,
    ub,
    args,
    out_dir,
    generation,
):
    """Short latent-space hill climb around each LLM-proposed chemical direction."""
    seeds = np.asarray(seed_latents, dtype=np.float32)
    current = np.asarray(current_population, dtype=np.float32)
    scale = np.maximum(ref_std, 1e-6)
    refined = []
    records = []

    for seed_idx, seed in enumerate(seeds):
        local_rng = np.random.default_rng(
            int(args.seed + generation * 10000 + seed_idx)
        )
        parent = seed.copy()
        initial_metrics = evaluator.evaluate(parent[None, :])
        initial_score = float(initial_metrics["score_ga"][0])

        for step in range(args.llm_local_refine_steps):
            step_sigma = (
                args.llm_local_refine_sigma
                * (args.llm_local_refine_sigma_decay ** step)
            )
            noise = local_rng.normal(
                0.0,
                step_sigma,
                size=(args.llm_local_refine_samples, parent.shape[0]),
            ).astype(np.float32)
            trials = parent[None, :] + noise * scale[None, :]
            trials = np.clip(trials, lb, ub).astype(np.float32)
            trials = np.concatenate([parent[None, :], trials], axis=0)
            metrics = evaluator.evaluate(trials)

            normalized_delta = (
                trials[:, None, :] - current[None, :, :]
            ) / scale[None, None, :]
            novelty = np.min(
                np.sqrt(np.mean(normalized_delta ** 2, axis=2)),
                axis=1,
            )
            eligible = np.flatnonzero(
                novelty >= args.llm_local_refine_min_novelty
            )
            if len(eligible) == 0:
                eligible = np.arange(len(trials))
            objective = (
                metrics["score_ga"][eligible]
                + args.llm_local_refine_novelty_weight * novelty[eligible]
            )
            parent = trials[eligible[int(np.argmax(objective))]].copy()

        final_metrics = evaluator.evaluate(parent[None, :])
        normalized_delta = (
            parent[None, :] - current
        ) / scale[None, :]
        final_novelty = float(
            np.min(np.sqrt(np.mean(normalized_delta ** 2, axis=1)))
        )
        final_score = float(final_metrics["score_ga"][0])
        refined.append(parent)
        records.append({
            "seed_idx": seed_idx,
            "initial_score": initial_score,
            "refined_score": final_score,
            "score_gain": final_score - initial_score,
            "novelty_to_population": final_novelty,
            "p_sweet": float(final_metrics["p_sweet"][0]),
            "pred_logsw": float(final_metrics["pred_logsw"][0]),
            "d_ood": float(final_metrics["d_ood"][0]),
        })

    refined = np.asarray(refined, dtype=np.float32)
    report = pd.DataFrame(records).sort_values(
        ["refined_score", "novelty_to_population"],
        ascending=False,
    )
    injection_dir = os.path.join(out_dir, "llm_online_injection")
    os.makedirs(injection_dir, exist_ok=True)
    report.to_csv(
        os.path.join(
            injection_dir,
            f"gen_{generation:03d}_local_refinement.csv",
        ),
        index=False,
    )
    order = report["seed_idx"].to_numpy(dtype=int)
    return refined[order], report.reset_index(drop=True)


# ============================================================
# Plotting
# ============================================================

def plot_progress(progress_df, out_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(progress_df["generation"], progress_df["best_score_so_far"], marker="o", label="Best-so-far score")
    plt.plot(progress_df["generation"], progress_df["top10_mean_score"], marker="s", label="Top-10 mean score")
    plt.xlabel("Generation")
    plt.ylabel("GA score")
    plt.title("Gated sweetness GA convergence")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_convergence_score.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(progress_df["generation"], progress_df["best_pred_logsw_so_far"], marker="o", label="Best-so-far pred_logSw")
    plt.plot(progress_df["generation"], progress_df["top10_mean_pred_logsw"], marker="s", label="Top-10 mean pred_logSw")
    plt.xlabel("Generation")
    plt.ylabel("Predicted logSw")
    plt.title("Predicted sweetness intensity during GA")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_convergence_logsw.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(progress_df["generation"], progress_df["success_count"], marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Success count")
    plt.title("Success count during GA")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_success_count.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(progress_df["generation"], progress_df["mean_p_sweet"], marker="o", label="Mean P_sweet")
    plt.plot(progress_df["generation"], progress_df["mean_d_ood"], marker="s", label="Mean D_OOD")
    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("Gate probability and OOD distance")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_gate_ood_curve.png"), dpi=300)
    plt.close()


def plot_latent_space(background_latent, final_population, out_dir, seed=2026, n_bg=3000):
    if len(background_latent) > n_bg:
        idx = np.random.default_rng(seed).choice(len(background_latent), n_bg, replace=False)
        bg = background_latent[idx]
    else:
        bg = background_latent

    X = np.vstack([bg, final_population])
    labels = np.array(["background"] * len(bg) + ["generated"] * len(final_population))

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels,
    })
    df.to_csv(os.path.join(out_dir, "latent_space_pca.csv"), index=False)

    plt.figure(figsize=(7, 6))
    mask_bg = labels == "background"
    mask_gen = labels == "generated"

    plt.scatter(coords[mask_bg, 0], coords[mask_bg, 1], s=8, alpha=0.35, label="OOD background")
    plt.scatter(coords[mask_gen, 0], coords[mask_gen, 1], s=20, alpha=0.85, label="GA final")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent chemical space")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_latent_space_pca.png"), dpi=300)
    plt.close()

    if HAS_UMAP:
        try:
            reducer = umap.UMAP(n_components=2, random_state=seed)
            umap_coords = reducer.fit_transform(X)

            umap_df = pd.DataFrame({
                "x": umap_coords[:, 0],
                "y": umap_coords[:, 1],
                "label": labels,
            })
            umap_df.to_csv(os.path.join(out_dir, "latent_space_umap.csv"), index=False)

            plt.figure(figsize=(7, 6))
            plt.scatter(umap_coords[mask_bg, 0], umap_coords[mask_bg, 1], s=8, alpha=0.35, label="OOD background")
            plt.scatter(umap_coords[mask_gen, 0], umap_coords[mask_gen, 1], s=20, alpha=0.85, label="GA final")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.title("Latent chemical space")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_latent_space_umap.png"), dpi=300)
            plt.close()
        except Exception as e:
            print("[WARN] UMAP failed:", str(e))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Sweet gated latent-space GA, corrected 4 groups")

    parser.add_argument("--psvae_root", default="/home/jqb/PS-VAE-main")
    parser.add_argument(
        "--psvae_ckpt",
        default="/home/jqb/PS-VAE-main/ckpts/sweet_pretrain_encoder_decoder_manifold_v2/lightning_logs/version_0/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--vocab",
        default="/home/jqb/PS-VAE-main/data/Sweet/Sweet_bpe_1000.txt",
    )
    parser.add_argument(
        "--latent_dir",
        default="/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_manifold_v2",
    )
    parser.add_argument(
        "--predictor_dir",
        default="/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/latent_evaluator_data_manifold_v2/gated_predictor_scaffold_ensemble_v1",
    )
    parser.add_argument(
        "--ood_dir",
        default="/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_like_dataset_out/ood_background_manifold_v2",
    )
    parser.add_argument(
        "--output_root",
        default="/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected",
    )

    parser.add_argument("--llm_latent_path", default=None)
    parser.add_argument("--llm_iterative_latent_dir", default=None)
    parser.add_argument("--llm_score_csv", default=None)

    parser.add_argument("--latent_dim", type=int, default=56)
    parser.add_argument("--psvae_node_hidden_dim", type=int, default=300)
    parser.add_argument("--psvae_graph_embedding_dim", type=int, default=400)
    parser.add_argument("--psvae_predictor_hidden_dim", type=int, default=200)

    parser.add_argument("--psvae_atom_embedding_dim", type=int, default=50)
    parser.add_argument("--psvae_piece_embedding_dim", type=int, default=100)
    parser.add_argument("--psvae_pos_embedding_dim", type=int, default=50)
    parser.add_argument("--psvae_piece_hidden_dim", type=int, default=200)
    parser.add_argument("--psvae_max_pos", type=int, default=50)

    parser.add_argument("--psvae_alpha", type=float, default=1.0)
    parser.add_argument("--psvae_beta", type=float, default=0.0)
    parser.add_argument("--psvae_max_beta", type=float, default=0.005)
    parser.add_argument("--psvae_step_beta", type=float, default=0.00025)
    parser.add_argument("--psvae_kl_anneal_iter", type=int, default=2500)
    parser.add_argument("--psvae_kl_warmup", type=int, default=5000)
    parser.add_argument("--psvae_lr", type=float, default=1.5e-4)
    parser.add_argument("--psvae_props", nargs="+", default=["qed", "logp"])

    parser.add_argument(
        "--init_mode",
        default="group_a_random",
        choices=[
            "group_a_random",
            "group_b_dataset",
            "group_c_llm",
            "group_d_llm_iterative",
        ],
    )

    parser.add_argument("--version", default="sweet_gated_corrected_v1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--pop_size", type=int, default=120)
    parser.add_argument("--n_gen", type=int, default=20)
    parser.add_argument("--elite_size", type=int, default=20)
    parser.add_argument("--cross_prob", type=float, default=0.35)
    parser.add_argument("--mut_prob", type=float, default=0.12)
    parser.add_argument("--mut_sigma", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--tournament_size", type=int, default=3)

    parser.add_argument("--seed_pool_size", type=int, default=200)
    parser.add_argument("--diverse_seed", action="store_true")
    parser.add_argument("--background_filter_by_evaluator", action="store_true")
    parser.add_argument("--dataset_filter_by_evaluator", action="store_true")
    parser.add_argument("--dataset_rank_by_fitness", action="store_true")
    parser.add_argument("--dataset_sweetdb_only", action="store_true")
    parser.add_argument("--dataset_sweetdb_high_potency", action="store_true")
    parser.add_argument("--dataset_scaffold_assignments", default=None)
    parser.add_argument("--dataset_holdout_fold", type=int, default=0)
    parser.add_argument("--dataset_logsw_quantile", type=float, default=0.70)
    parser.add_argument("--seed_min_p_sweet", type=float, default=0.60)
    parser.add_argument("--seed_augment_sigma", type=float, default=0.05)

    parser.add_argument("--lambda_ood", type=float, default=0.35)
    parser.add_argument(
        "--lambda_reg_uncertainty",
        type=float,
        default=0.0,
        help="Penalty applied to ensemble logSw std: pred_logSw = mean - lambda * std.",
    )
    parser.add_argument("--lambda_desc", type=float, default=0.50)
    parser.add_argument("--lambda_llm", type=float, default=0.10)
    parser.add_argument("--ood_k", type=int, default=10)
    parser.add_argument("--logsw_score_cap", type=float, default=3.5)
    parser.add_argument(
        "--objective",
        default="legacy",
        choices=["legacy", "constrained_sweetness"],
    )
    parser.add_argument("--p_sweet_threshold", type=float, default=0.70)
    parser.add_argument("--logsw_success_threshold", type=float, default=2.30)
    parser.add_argument("--ood_p95", type=float, default=None)

    parser.add_argument("--success_p_sweet", type=float, default=0.70)
    parser.add_argument("--success_ood_percentile", type=float, default=95.0)
    parser.add_argument("--success_logsw", type=float, default=None)
    parser.add_argument("--success_logsw_quantile", type=float, default=0.75)

    parser.add_argument("--final_min_logsw", type=float, default=1.0)
    parser.add_argument("--use_llm_filter", action="store_true")

    parser.add_argument("--bound_low_q", type=float, default=0.01)
    parser.add_argument("--bound_high_q", type=float, default=0.99)

    parser.add_argument("--decode_final", action="store_true")
    parser.add_argument("--decode_archive", action="store_true")
    parser.add_argument("--save_population_history", action="store_true")
    parser.add_argument("--archive_topk", type=int, default=20)
    parser.add_argument("--max_atom_num", type=int, default=80)
    parser.add_argument("--add_edge_th", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument("--llm_feedback_interval", type=int, default=5)
    parser.add_argument("--llm_feedback_topn", type=int, default=20)
    parser.add_argument("--llm_inject_ratio", type=float, default=0.25)
    parser.add_argument("--llm_online_enable", action="store_true")
    parser.add_argument(
        "--llm_helper_dir",
        default="/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_corrected/generate_sweet_smiles",
    )
    parser.add_argument("--llm_base_url", default=None)
    parser.add_argument("--llm_api_key", default=None)
    parser.add_argument("--llm_generator_base_url", default=None)
    parser.add_argument("--llm_generator_api_key", default=None)
    parser.add_argument("--llm_judge_base_url", default=None)
    parser.add_argument("--llm_judge_api_key", default=None)
    parser.add_argument("--llm_generator_model", default="gpt-5.5")
    parser.add_argument("--llm_judge_model", default="gemini-3.1-pro-preview")
    parser.add_argument("--llm_generator_temperature", type=float, default=0.55)
    parser.add_argument("--llm_judge_temperature", type=float, default=0.0)
    parser.add_argument("--llm_generator_max_tokens", type=int, default=1800)
    parser.add_argument("--llm_judge_max_tokens", type=int, default=700)
    parser.add_argument("--llm_judge_batch_size", type=int, default=10)
    parser.add_argument("--llm_online_candidates", type=int, default=20)
    parser.add_argument("--llm_inject_size", type=int, default=10)
    parser.add_argument("--llm_judge_keep", type=int, default=10)
    parser.add_argument("--llm_min_prior", type=float, default=0.45)
    parser.add_argument("--llm_max_risk", type=float, default=0.45)
    parser.add_argument("--llm_stagnation_window", type=int, default=3)
    parser.add_argument("--llm_stagnation_score_delta", type=float, default=0.12)
    parser.add_argument("--llm_stagnation_no_improve", type=int, default=2)
    parser.add_argument("--llm_inject_min_score_gain", type=float, default=0.0)
    parser.add_argument("--llm_trigger_unique_ratio", type=float, default=0.72)
    parser.add_argument("--llm_local_refine_steps", type=int, default=3)
    parser.add_argument("--llm_local_refine_samples", type=int, default=32)
    parser.add_argument("--llm_local_refine_sigma", type=float, default=0.10)
    parser.add_argument("--llm_local_refine_sigma_decay", type=float, default=1.0)
    parser.add_argument("--llm_local_refine_min_novelty", type=float, default=0.08)
    parser.add_argument("--llm_local_refine_novelty_weight", type=float, default=0.08)
    parser.add_argument("--llm_inject_min_novelty", type=float, default=0.08)
    parser.add_argument("--llm_inject_min_population_quantile", type=float, default=0.0)

    parser.add_argument("--enforce_unique_smiles", action="store_true")
    parser.add_argument("--unique_elite_candidates", type=int, default=30)
    parser.add_argument("--unique_refill_attempts", type=int, default=30)
    parser.add_argument("--unique_refill_sigma", type=float, default=0.08)
    parser.add_argument("--unique_refill_from_seed_prob", type=float, default=0.75)
    parser.add_argument(
        "--unique_refill_pool",
        choices=("current", "group_seed", "background"),
        default="current",
        help="Fair default uses only mutations around the current population.",
    )
    parser.add_argument("--unique_target_ratio", type=float, default=1.0)
    parser.add_argument("--unique_refill_min_p_sweet", type=float, default=0.50)
    parser.add_argument("--unique_refill_min_logsw", type=float, default=1.80)
    parser.add_argument("--unique_refill_max_ood_ratio", type=float, default=1.00)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    out_dir = os.path.join(
        args.output_root,
        f"{args.init_mode}_{args.version}_seed{args.seed}"
    )
    ensure_dir(out_dir)

    print("=" * 80)
    print("Sweet gated latent GA, corrected 4 groups")
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("out_dir:", out_dir)
    print("device:", device)
    print("=" * 80)

    background_latent_path = os.path.join(args.ood_dir, "background_latent.npy")
    if not os.path.exists(background_latent_path):
        raise FileNotFoundError(f"background_latent.npy not found: {background_latent_path}")

    background_latent = np.load(background_latent_path).astype(np.float32)

    if background_latent.ndim != 2:
        raise ValueError(f"background_latent must be 2D, got {background_latent.shape}")

    if background_latent.shape[1] != args.latent_dim:
        raise ValueError(
            f"latent dim mismatch: background={background_latent.shape[1]}, args={args.latent_dim}"
        )

    lb = np.quantile(background_latent, args.bound_low_q, axis=0).astype(np.float32)
    ub = np.quantile(background_latent, args.bound_high_q, axis=0).astype(np.float32)

    ref_std = np.std(background_latent, axis=0).astype(np.float32)
    ref_std = np.maximum(ref_std, 1e-6)
    mut_sigma_vec = ref_std * args.mut_sigma

    evaluator = SweetGatedEvaluator(
        predictor_dir=args.predictor_dir,
        ood_dir=args.ood_dir,
        latent_dim=args.latent_dim,
        device=device,
        lambda_ood=args.lambda_ood,
        lambda_reg_uncertainty=args.lambda_reg_uncertainty,
        ood_k=args.ood_k,
        logsw_score_cap=args.logsw_score_cap,
        objective=args.objective,
        p_sweet_threshold=args.p_sweet_threshold,
        logsw_success_threshold=args.logsw_success_threshold,
        ood_threshold=args.ood_p95,
    )

    sweetdb_labels_path = os.path.join(args.latent_dir, "sweetdb_labels.npy")

    if args.objective == "constrained_sweetness":
        success_logsw = float(args.logsw_success_threshold)
    elif args.success_logsw is not None:
        success_logsw = float(args.success_logsw)
    elif os.path.exists(sweetdb_labels_path):
        sweet_y = np.load(sweetdb_labels_path).astype(float)
        success_logsw = float(np.quantile(sweet_y, args.success_logsw_quantile))
    else:
        success_logsw = 1.0

    if args.success_ood_percentile >= 99:
        success_ood = evaluator.ood_p99
    elif args.success_ood_percentile >= 95:
        success_ood = evaluator.ood_p95
    else:
        success_ood = evaluator.ood_p95

    print("=" * 80)
    print("Success definition for GA progress")
    print("P_sweet >=", args.success_p_sweet)
    print("pred_logSw >=", success_logsw)
    print("D_OOD <=", success_ood)
    print("logSw score cap =", args.logsw_score_cap)
    print("=" * 80)

    # PS-VAE is required for corrected final decoding and re-encoding.
    model_psvae, tokenizer = load_psvae(args, device)

    llm_score_map = load_optional_llm_score_csv(args.llm_score_csv)

    population, seed_source, population_source = build_initial_population(args, evaluator, background_latent)
    population = np.clip(population, lb, ub).astype(np.float32)

    llm_seed_latent_for_refill = None
    if args.llm_latent_path is not None and os.path.exists(args.llm_latent_path):
        try:
            llm_seed_latent_for_refill = np.load(args.llm_latent_path).astype(np.float32)
        except Exception as e:
            print(f"[WARN] failed to load LLM refill latent pool: {e}")

    print("=" * 80)
    print("Initial population")
    print("seed_source:", seed_source)
    print("population:", population.shape)
    print("=" * 80)

    start_time = time.time()

    progress_records = []
    topk_archive_records = []
    best_candidates_over_time = []

    best_score_so_far = -1e18
    best_pred_logsw_so_far = -1e18
    best_record_so_far = None
    no_improve = 0
    injection_records = []
    unique_constraint_records = []
    population_history_dir = os.path.join(out_dir, "population_history")
    if args.save_population_history:
        ensure_dir(population_history_dir)

    for gen in range(args.n_gen):
        pending_injection = None
        pending_injection_rows = []
        metrics = evaluator.evaluate(population)

        if args.save_population_history:
            generation_number = gen + 1
            np.save(
                os.path.join(population_history_dir, f"gen_{generation_number:03d}.npy"),
                population.astype(np.float32),
            )
            pd.DataFrame({
                "idx": np.arange(len(population), dtype=int),
                "source_type": population_source.astype(str),
                "score_ga": metrics["score_ga"],
                "p_sweet": metrics["p_sweet"],
                "pred_logsw": metrics["pred_logsw"],
                "d_ood": metrics["d_ood"],
            }).to_csv(
                os.path.join(population_history_dir, f"gen_{generation_number:03d}_metrics.csv"),
                index=False,
            )

        score = metrics["score_ga"]
        p_sweet = metrics["p_sweet"]
        pred_logsw = metrics["pred_logsw"]
        pred_logsw_clipped = metrics["pred_logsw_clipped"]
        d_ood = metrics["d_ood"]

        sorted_idx = np.argsort(score)[::-1]

        best_idx = int(sorted_idx[0])
        best_score = float(score[best_idx])
        best_pred_logsw = float(pred_logsw[best_idx])

        top10_idx = sorted_idx[:min(10, len(sorted_idx))]
        top10_mean_score = float(np.mean(score[top10_idx]))
        top10_mean_pred_logsw = float(np.mean(pred_logsw[top10_idx]))

        success_mask = (
            (p_sweet >= args.success_p_sweet)
            & (pred_logsw >= success_logsw)
            & (d_ood <= success_ood)
        )

        success_count = int(np.sum(success_mask))
        success_rate = float(success_count / len(population))

        if best_score > best_score_so_far:
            best_score_so_far = best_score
            best_record_so_far = {
                "generation": int(gen),
                "idx": int(best_idx),
                "score_ga": float(best_score),
                "p_sweet": float(p_sweet[best_idx]),
                "pred_logsw": float(pred_logsw[best_idx]),
                "pred_logsw_clipped": float(pred_logsw_clipped[best_idx]),
                "d_ood": float(d_ood[best_idx]),
                "d_ood_norm": float(metrics["d_ood_norm"][best_idx]),
            }
            no_improve = 0
        else:
            no_improve += 1

        best_pred_logsw_so_far = max(best_pred_logsw_so_far, best_pred_logsw)

        progress_records.append({
            "generation": int(gen + 1),
            "evaluations": int((gen + 1) * args.pop_size),
            "elapsed_time_sec": float(time.time() - start_time),

            "mean_score": float(np.mean(score)),
            "best_score": float(best_score),
            "best_score_so_far": float(best_score_so_far),
            "top10_mean_score": float(top10_mean_score),

            "mean_pred_logsw": float(np.mean(pred_logsw)),
            "mean_pred_logsw_clipped": float(np.mean(pred_logsw_clipped)),
            "best_pred_logsw": float(best_pred_logsw),
            "best_pred_logsw_so_far": float(best_pred_logsw_so_far),
            "top10_mean_pred_logsw": float(top10_mean_pred_logsw),

            "mean_p_sweet": float(np.mean(p_sweet)),
            "best_p_sweet": float(p_sweet[best_idx]),

            "mean_d_ood": float(np.mean(d_ood)),
            "best_d_ood": float(d_ood[best_idx]),

            "success_count": int(success_count),
            "success_rate": float(success_rate),
            "no_improve": int(no_improve),
        })

        best_candidates_over_time.append({
            "generation": int(gen + 1),
            "score_ga": float(best_score),
            "p_sweet": float(p_sweet[best_idx]),
            "pred_logsw": float(pred_logsw[best_idx]),
            "pred_logsw_mean": float(metrics.get("pred_logsw_mean", pred_logsw)[best_idx]),
            "pred_logsw_std": float(metrics.get("pred_logsw_std", np.zeros_like(pred_logsw))[best_idx]),
            "pred_logsw_clipped": float(pred_logsw_clipped[best_idx]),
            "d_ood": float(d_ood[best_idx]),
            "d_ood_norm": float(metrics["d_ood_norm"][best_idx]),
            "success": bool(success_mask[best_idx]),
        })

        archive_idx = sorted_idx[:min(args.archive_topk, len(sorted_idx))]

        archive_smiles = [None] * len(archive_idx)
        archive_valid = np.zeros(len(archive_idx), dtype=bool)

        if args.decode_archive:
            archive_pop = population[archive_idx]
            archive_smiles, archive_valid = decode_population(model_psvae, archive_pop, device, args)

        current_archive_rows = []

        for rank, idx in enumerate(archive_idx):
            row = {
                "generation": int(gen + 1),
                "rank": int(rank),
                "idx": int(idx),
                "smiles": archive_smiles[rank],
                "valid": bool(archive_valid[rank]),
                "score_ga": float(score[idx]),
                "p_sweet": float(p_sweet[idx]),
                "pred_logsw": float(pred_logsw[idx]),
                "pred_logsw_mean": float(metrics.get("pred_logsw_mean", pred_logsw)[idx]),
                "pred_logsw_std": float(metrics.get("pred_logsw_std", np.zeros_like(pred_logsw))[idx]),
                "pred_logsw_clipped": float(pred_logsw_clipped[idx]),
                "d_ood": float(d_ood[idx]),
                "d_ood_norm": float(metrics["d_ood_norm"][idx]),
                "success": bool(success_mask[idx]),
            }
            topk_archive_records.append(row)
            current_archive_rows.append(row)

        print(
            f"[Gen {gen:03d}] "
            f"best_score={best_score:.4f} | mean_score={np.mean(score):.4f} | "
            f"best_logSw={best_pred_logsw:.4f} | mean_p={np.mean(p_sweet):.4f} | "
            f"mean_ood={np.mean(d_ood):.4f} | success={success_count}/{len(population)} | "
            f"no_improve={no_improve}"
        )

        generation_number = gen + 1
        if args.init_mode == "group_d_llm_iterative" and args.llm_feedback_interval > 0:
            recent_progress = progress_records[-args.llm_stagnation_window:]
            recent_top10 = [float(row["top10_mean_score"]) for row in recent_progress]
            score_stagnant = (
                len(recent_top10) >= args.llm_stagnation_window
                and max(recent_top10) - min(recent_top10)
                <= args.llm_stagnation_score_delta
            )
            latest_unique_ratio = 1.0
            if unique_constraint_records:
                latest_unique_ratio = (
                    float(unique_constraint_records[-1]["unique_after_refill"])
                    / float(args.pop_size)
                )
            diversity_triggered = (
                latest_unique_ratio <= args.llm_trigger_unique_ratio
            )
            reflection_triggered = (
                no_improve >= args.llm_stagnation_no_improve
                or score_stagnant
                or diversity_triggered
            )
            if (
                generation_number < args.n_gen
                and generation_number % args.llm_feedback_interval == 0
                and reflection_triggered
            ):
                tmp_archive_df = pd.DataFrame(topk_archive_records)
                feedback_path = export_llm_feedback(
                    out_dir=out_dir,
                    generation=generation_number,
                    archive_df=tmp_archive_df,
                    top_n=args.llm_feedback_topn,
                )
                if feedback_path is not None:
                    print(f"[Group D] feedback exported for LLM: {feedback_path}")

                if args.llm_online_enable:
                    # Reflection decoding/encoding may consume Python, NumPy, and
                    # Torch RNG state. Isolate it so a rejected injection cannot
                    # silently alter the subsequent GA trajectory.
                    python_rng_state = random.getstate()
                    numpy_rng_state = np.random.get_state()
                    torch_rng_state = torch.random.get_rng_state()
                    cuda_rng_state = (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    )
                    try:
                        pending_injection, pending_injection_rows = llm_online_reflection_injection(
                            generation=generation_number,
                            population=population,
                            metrics=metrics,
                            sorted_idx=sorted_idx,
                            model_psvae=model_psvae,
                            tokenizer=tokenizer,
                            device=device,
                            args=args,
                            out_dir=out_dir,
                            existing_smiles=archive_smiles,
                        )
                    finally:
                        random.setstate(python_rng_state)
                        np.random.set_state(numpy_rng_state)
                        torch.random.set_rng_state(torch_rng_state)
                        if cuda_rng_state is not None:
                            torch.cuda.set_rng_state_all(cuda_rng_state)
                else:
                    new_latent, expected_path = maybe_load_iterative_llm_latent(args, generation_number)
                    if new_latent is not None:
                        print(f"[Group D] queued iterative LLM latent: {expected_path}, shape={new_latent.shape}")
                        pending_injection = new_latent[:args.llm_judge_keep]
                        pending_injection_rows = [
                            {"source": "saved_llm_reflection"}
                            for _ in range(len(pending_injection))
                        ]
                    else:
                        print(f"[Group D] no iterative latent found yet: {expected_path}")
            elif (
                generation_number < args.n_gen
                and generation_number % args.llm_feedback_interval == 0
            ):
                print(
                    f"[Group D] reflection skipped at gen {generation_number}: "
                    f"search still improving (top10 span={max(recent_top10) - min(recent_top10):.4f}, "
                    f"unique_ratio={latest_unique_ratio:.3f})."
                )

        if no_improve >= args.patience:
            print(f"[Early stop] no improvement for {args.patience} generations.")
            break

        if gen == args.n_gen - 1:
            break

        elites, elite_sources, elite_unique_report = select_unique_smiles_elites(
            population=population,
            population_source=population_source,
            sorted_idx=sorted_idx,
            model_psvae=model_psvae,
            device=device,
            args=args,
        )
        new_pop = list(elites)
        new_sources = list(elite_sources)

        while len(new_pop) < args.pop_size:
            p1 = tournament_selection_max(population, score, tourn_size=args.tournament_size)
            p2 = tournament_selection_max(population, score, tourn_size=args.tournament_size)

            c1, c2 = arithmetic_crossover(p1, p2, args.cross_prob, lb, ub)

            c1 = gaussian_mutation(c1, args.mut_prob, mut_sigma_vec, lb, ub)
            c2 = gaussian_mutation(c2, args.mut_prob, mut_sigma_vec, lb, ub)

            new_pop.append(c1)
            new_sources.append("ga_mutation_crossover")
            if len(new_pop) < args.pop_size:
                new_pop.append(c2)
                new_sources.append("ga_mutation_crossover")

        population = np.asarray(new_pop, dtype=np.float32)
        population_source = np.asarray(new_sources, dtype=object)

        if pending_injection is not None and len(pending_injection) > 0:
            pending_injection = np.clip(np.asarray(pending_injection, dtype=np.float32), lb, ub)
            pending_injection, refinement_report = locally_refine_llm_latents(
                seed_latents=pending_injection,
                current_population=population,
                evaluator=evaluator,
                ref_std=ref_std,
                lb=lb,
                ub=ub,
                args=args,
                out_dir=out_dir,
                generation=generation_number,
            )
            offspring_idx = np.arange(args.elite_size, len(population))
            offspring_metrics = evaluator.evaluate(population[offspring_idx])
            replace_order = offspring_idx[np.argsort(offspring_metrics["score_ga"])]
            injection_metrics = evaluator.evaluate(pending_injection)
            candidate_order = np.argsort(injection_metrics["score_ga"])[::-1]
            population_score_floor = float(np.quantile(
                evaluator.evaluate(population)["score_ga"],
                args.llm_inject_min_population_quantile,
            ))
            accepted_pairs = []
            for candidate_i, replace_i in zip(candidate_order, replace_order):
                if len(accepted_pairs) >= args.llm_inject_size:
                    break
                offspring_local_i = int(np.where(offspring_idx == replace_i)[0][0])
                incumbent_score = float(offspring_metrics["score_ga"][offspring_local_i])
                candidate_score = float(injection_metrics["score_ga"][candidate_i])
                if candidate_score < incumbent_score + args.llm_inject_min_score_gain:
                    continue
                if candidate_score < population_score_floor:
                    continue
                if (
                    float(refinement_report.iloc[candidate_i]["novelty_to_population"])
                    < args.llm_inject_min_novelty
                ):
                    continue
                accepted_pairs.append((int(candidate_i), int(replace_i)))

            for candidate_i, replace_i in accepted_pairs:
                population[replace_i] = pending_injection[candidate_i]
                population_source[replace_i] = "llm_injection_accepted"

            n_replace = len(accepted_pairs)
            replace_idx = [replace_i for _, replace_i in accepted_pairs]
            injection_records.append({
                "generation": int(generation_number),
                "generated_basic_gate_count": int(len(pending_injection_rows)),
                "strict_bpe_injected_count": int(n_replace),
                "replace_indices": replace_idx,
                "acceptance_rule": (
                    "locally_refined candidate_score >= incumbent_score + min_gain "
                    "and population quantile floor and novelty >= min_novelty"
                ),
            })
            print(
                f"[Group D] accepted {n_replace}/{len(pending_injection)} "
                f"strict-BPE LLM candidates after gen {generation_number}."
            )

        population, population_source, unique_report = enforce_unique_smiles_population(
            population=population,
        population_source=population_source,
        model_psvae=model_psvae,
        evaluator=evaluator,
        device=device,
            args=args,
            lb=lb,
            ub=ub,
            background_latent=background_latent,
            llm_seed_latent=llm_seed_latent_for_refill,
        )
        if unique_report is not None:
            rec = {
                "generation": int(generation_number),
                **elite_unique_report,
                **unique_report,
            }
            unique_constraint_records.append(rec)
            print(
                "[Unique SMILES] "
                f"gen={gen} duplicates={unique_report['duplicates_before_refill']} "
                f"refilled={unique_report['refilled']} "
                f"unique_after={unique_report['unique_after_refill']}"
            )

    # ========================================================
    # Final decode and corrected re-score
    # ========================================================

    final_metrics = evaluator.evaluate(population)

    print("Decoding final population...")
    final_smiles, final_valid = decode_population(model_psvae, population, device, args)

    base_rows = []
    for i in range(len(population)):
        base_rows.append({
            "generation": "final",
            "idx": i,
            "source_type": str(population_source[i]),
            "smiles": final_smiles[i],
            "valid": bool(final_valid[i]),
            "score_ga": float(final_metrics["score_ga"][i]),
            "p_sweet": float(final_metrics["p_sweet"][i]),
            "pred_logsw": float(final_metrics["pred_logsw"][i]),
            "pred_logsw_mean": float(final_metrics.get("pred_logsw_mean", final_metrics["pred_logsw"])[i]),
            "pred_logsw_std": float(final_metrics.get("pred_logsw_std", np.zeros_like(final_metrics["pred_logsw"]))[i]),
            "pred_logsw_clipped": float(final_metrics["pred_logsw_clipped"][i]),
            "d_ood": float(final_metrics["d_ood"][i]),
            "d_ood_norm": float(final_metrics["d_ood_norm"][i]),
        })

    final_raw_df = pd.DataFrame(base_rows)
    final_raw_df.to_csv(os.path.join(out_dir, "final_population_latent_score.csv"), index=False)

    print("Re-encoding and final rescoring decoded candidates...")
    final_corrected_df = final_rescore_candidates(
        df=final_raw_df,
        population=population,
        model_psvae=model_psvae,
        tokenizer=tokenizer,
        evaluator=evaluator,
        device=device,
        args=args,
        llm_score_map=llm_score_map,
    )

    final_corrected_df.to_csv(os.path.join(out_dir, "final_population_corrected.csv"), index=False)
    final_corrected_df.to_csv(os.path.join(out_dir, "final_population.csv"), index=False)
    np.save(os.path.join(out_dir, "final_population_latent.npy"), population.astype(np.float32))

    progress_df = pd.DataFrame(progress_records)
    progress_df.to_csv(os.path.join(out_dir, "progress_metrics.csv"), index=False)
    progress_df.to_csv(os.path.join(out_dir, "progress.csv"), index=False)

    archive_df = pd.DataFrame(topk_archive_records)
    archive_df.to_csv(os.path.join(out_dir, "topk_archive.csv"), index=False)

    best_time_df = pd.DataFrame(best_candidates_over_time)
    best_time_df.to_csv(os.path.join(out_dir, "best_candidates_over_time.csv"), index=False)

    combined_parts = []
    if len(archive_df) > 0:
        combined_parts.append(archive_df.copy())
    combined_parts.append(final_corrected_df.copy())

    combined = pd.concat(combined_parts, ignore_index=True, sort=False)

    if "canonical_smiles" not in combined.columns:
        combined["canonical_smiles"] = combined["smiles"].apply(canonicalize_smiles)

    combined = combined.drop_duplicates(subset=["canonical_smiles"], keep="first")

    if "final_score" in combined.columns:
        combined = combined.sort_values("final_score", ascending=False)
    else:
        combined = combined.sort_values("score_ga", ascending=False)

    top50 = combined.head(50).copy()
    top50.to_csv(os.path.join(out_dir, "top50_candidates_corrected.csv"), index=False)
    top50.to_csv(os.path.join(out_dir, "top_candidates.csv"), index=False)

    reliable_df = final_corrected_df[final_corrected_df["reliable_candidate"] == True].copy()
    reliable_df = reliable_df.sort_values("final_score", ascending=False)
    reliable_df.to_csv(os.path.join(out_dir, "reliable_candidates.csv"), index=False)

    unique_reliable_df = reliable_df.copy()
    if len(unique_reliable_df) > 0:
        if "canonical_smiles" not in unique_reliable_df.columns:
            unique_reliable_df["canonical_smiles"] = unique_reliable_df["smiles"].apply(canonicalize_smiles)
        unique_reliable_df = unique_reliable_df[
            unique_reliable_df["canonical_smiles"].notna() & (unique_reliable_df["canonical_smiles"] != "")
        ].copy()
        unique_reliable_df = unique_reliable_df.drop_duplicates(subset=["canonical_smiles"], keep="first")
        unique_reliable_df = unique_reliable_df.sort_values("final_score", ascending=False)
    unique_reliable_df.to_csv(os.path.join(out_dir, "unique_reliable_candidates.csv"), index=False)

    valid_smiles = [s for s in final_smiles if s is not None]
    unique_smiles = {canonicalize_smiles(s) for s in valid_smiles if canonicalize_smiles(s) is not None}
    diversity = compute_diversity(valid_smiles)
    validity = float(np.mean(final_valid)) if len(final_valid) > 0 else 0.0

    final_success_count = int(final_corrected_df["reliable_candidate"].sum())
    final_success_rate = float(final_success_count / len(final_corrected_df))
    unique_reliable_count = int(len(unique_reliable_df))
    unique_reliable_rate = float(unique_reliable_count / len(final_corrected_df)) if len(final_corrected_df) else 0.0

    summary = {
        "task": "Constrained Sweetness Potency Optimization",
        "objective": args.objective,
        "ga_fitness": (
            "score = pred_logSw + 0.50*P_sweet - 0.80*max(0, p_threshold-P_sweet) "
            "- 0.50*max(0, D_OOD-OOD_p95)/OOD_p95"
            if args.objective == "constrained_sweetness"
            else "score_ga = P_sweet * min(pred_logSw, logsw_score_cap) - lambda_ood * D_OOD_norm"
        ),
        "final_scoring": "decode -> strict re-encode -> re-score -> descriptor filter -> final_score",

        "init_mode": args.init_mode,
        "seed_source": seed_source,
        "version": args.version,
        "seed": args.seed,

        "pop_size": args.pop_size,
        "n_gen_requested": args.n_gen,
        "n_gen_finished": int(len(progress_df)),
        "elite_size": args.elite_size,
        "cross_prob": args.cross_prob,
        "mut_prob": args.mut_prob,
        "mut_sigma": args.mut_sigma,

        "lambda_ood": args.lambda_ood,
        "lambda_reg_uncertainty": args.lambda_reg_uncertainty,
        "lambda_desc": args.lambda_desc,
        "lambda_llm": args.lambda_llm,
        "logsw_score_cap": args.logsw_score_cap,

        "success_p_sweet": args.success_p_sweet,
        "success_logsw_for_progress": success_logsw,
        "success_ood": success_ood,
        "llm_seed_policy": "LLM prior + basic chemistry gate only; GA fitness is not used to reject LLM seeds",
        "llm_online_injections": injection_records,
        "unique_smiles_constraint": bool(args.enforce_unique_smiles),
        "unique_smiles_constraint_records": unique_constraint_records,

        "best_score_ga_final": float(final_raw_df["score_ga"].max()),
        "mean_score_ga_final": float(final_raw_df["score_ga"].mean()),

        "best_final_score": float(final_corrected_df["final_score"].max()),
        "mean_final_score": float(final_corrected_df["final_score"].mean()),
        "top10_mean_final_score": float(final_corrected_df.head(10)["final_score"].mean()),

        "best_pred_logsw_reencoded": float(
            np.nanmax(final_corrected_df["pred_logsw_reencoded"].values)
        ),
        "mean_pred_logsw_reencoded": float(
            np.nanmean(final_corrected_df["pred_logsw_reencoded"].values)
        ),

        "reliable_candidate_count": final_success_count,
        "reliable_candidate_rate": final_success_rate,
        "unique_reliable_candidate_count": unique_reliable_count,
        "unique_reliable_candidate_rate": unique_reliable_rate,

        "validity_final": validity,
        "valid_count_final": int(np.sum(final_valid)),
        "unique_smiles_count_final": int(len(unique_smiles)),
        "unique_smiles_ratio_final": float(len(unique_smiles) / max(len(valid_smiles), 1)),
        "diversity_final": diversity,

        "best_record_so_far": best_record_so_far,

        "time_sec_total": float(time.time() - start_time),
        "n_evaluations_total": int(len(progress_df) * args.pop_size),
    }

    save_json(summary, os.path.join(out_dir, "summary.json"))
    save_json(vars(args), os.path.join(out_dir, "config.json"))
    if injection_records:
        pd.DataFrame(injection_records).to_csv(
            os.path.join(out_dir, "llm_injection_summary.csv"),
            index=False,
        )
    if unique_constraint_records:
        pd.DataFrame(unique_constraint_records).to_csv(
            os.path.join(out_dir, "unique_smiles_constraint_summary.csv"),
            index=False,
        )

    if len(progress_df) > 0:
        plot_progress(progress_df, out_dir)

    try:
        plot_latent_space(background_latent, population, out_dir, seed=args.seed)
    except Exception as e:
        print("[WARN] latent-space plot failed:", str(e))

    print("=" * 80)
    print("Finished.")
    print("Output directory:", out_dir)
    print("Best GA score:", summary["best_score_ga_final"])
    print("Best final score:", summary["best_final_score"])
    print("Reliable candidates:", final_success_count, "/", len(final_corrected_df))
    print("Final validity:", validity)
    print("Final diversity:", diversity)
    print("=" * 80)


if __name__ == "__main__":
    main()
