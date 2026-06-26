#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import sys
import csv
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.serialization
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.rdchem import BondType

RDLogger.DisableLog("rdApp.*")


# ======================
# PS-VAE root
# ======================

PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))

from pl_models import PSVAEModel
from utils.chem_utils import smiles2molecule, GeneralVocab
from data.mol_bpe import Tokenizer

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def canonicalize_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def calc_rdkit_props(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None

    try:
        props = {
            "logP": float(Descriptors.MolLogP(mol)),
            "MolWt": float(Descriptors.MolWt(mol)),
            "TPSA": float(Descriptors.TPSA(mol)),
            "HBD": int(Descriptors.NumHDonors(mol)),
            "HBA": int(Descriptors.NumHAcceptors(mol)),
            "RotBonds": int(Descriptors.NumRotatableBonds(mol)),
            "QED": float(QED.qed(mol)),
        }
        return props
    except Exception:
        return None


def read_smiles_file(path, smiles_col="smiles", max_mols=None):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found. Available columns: {list(df.columns)}")
        raw = df[smiles_col].astype(str).tolist()

    elif path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found. Available columns: {list(df.columns)}")
        raw = df[smiles_col].astype(str).tolist()

    else:
        raw = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                raw.append(s.split()[0])

    smiles = []
    seen = set()

    for s in raw:
        can = canonicalize_smiles(s)
        if can is None:
            continue
        if can in seen:
            continue
        seen.add(can)
        smiles.append(can)

        if max_mols is not None and len(smiles) >= max_mols:
            break

    return smiles


def load_psvae(ckpt, device):
    print(f"[INFO] Loading PS-VAE checkpoint: {ckpt}", flush=True)

    old_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return old_torch_load(*args, **kwargs)

    torch.load = patched_torch_load

    try:
        model = PSVAEModel.load_from_checkpoint(ckpt, map_location=device)
    finally:
        torch.load = old_torch_load

    model.eval()
    model.to(device)

    print("[INFO] PS-VAE loaded.", flush=True)
    return model


def smiles_to_latent(model, smi, device, kekulize=False):
    try:
        mol = smiles2molecule(smi, kekulize=kekulize)
        if mol is None:
            mol = Chem.MolFromSmiles(smi)

        if mol is None:
            return None

        mol = Chem.RemoveHs(mol)

        with torch.no_grad():
            z = model.get_z_from_mol(mol)
            if z.dim() > 1:
                z = z.squeeze(0)
            z = z.detach().cpu().numpy().astype(np.float32)

        if not np.all(np.isfinite(z)):
            return None

        return z

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser("Encode ZINC SMILES into PS-VAE latent and calculate RDKit logP")

    parser.add_argument("--ckpt", type=str, required=True, help="ZINC-trained PS-VAE checkpoint")
    parser.add_argument("--input", type=str, required=True, help="ZINC smiles csv/txt/tsv")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--max_mols", type=int, default=None)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kekulize", action="store_true", help="Use kekulize=True in smiles2molecule")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    print("\n========== CONFIG ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f"[INFO] device = {device}")

    model = load_psvae(args.ckpt, device)

    print("[INFO] Loading SMILES...")
    smiles = read_smiles_file(args.input, smiles_col=args.smiles_col, max_mols=args.max_mols)
    print(f"[INFO] Unique canonical SMILES loaded: {len(smiles)}")

    latents = []
    rows = []
    failed = []

    for idx, smi in enumerate(tqdm(smiles, desc="Encoding ZINC")):
        props = calc_rdkit_props(smi)
        if props is None:
            failed.append({"smiles": smi, "reason": "rdkit_props_failed"})
            continue

        z = smiles_to_latent(model, smi, device=device, kekulize=args.kekulize)
        if z is None:
            failed.append({"smiles": smi, "reason": "encode_failed"})
            continue

        latent_idx = len(latents)
        latents.append(z)

        row = {
            "latent_idx": latent_idx,
            "smiles": smi,
            "canonical_smiles": smi,
        }
        row.update(props)
        rows.append(row)

    if len(latents) == 0:
        raise RuntimeError("No molecules were successfully encoded.")

    latents = np.vstack(latents).astype(np.float32)
    meta_df = pd.DataFrame(rows)

    latent_path = os.path.join(args.out_dir, "zinc_logp_latent.npy")
    meta_path = os.path.join(args.out_dir, "zinc_logp_meta.csv")
    y_path = os.path.join(args.out_dir, "zinc_logp_y.npy")
    failed_path = os.path.join(args.out_dir, "zinc_logp_failed.csv")
    summary_path = os.path.join(args.out_dir, "zinc_logp_encode_summary.json")

    np.save(latent_path, latents)
    np.save(y_path, meta_df["logP"].values.astype(np.float32))
    meta_df.to_csv(meta_path, index=False)

    if len(failed) > 0:
        pd.DataFrame(failed).to_csv(failed_path, index=False)

    summary = {
        "ckpt": args.ckpt,
        "input": args.input,
        "n_input_unique_smiles": len(smiles),
        "n_encoded": int(len(latents)),
        "n_failed": int(len(failed)),
        "latent_shape": list(latents.shape),
        "latent_path": latent_path,
        "meta_path": meta_path,
        "y_path": y_path,
        "failed_path": failed_path if len(failed) > 0 else None,
        "logP_stats": {
            "min": float(meta_df["logP"].min()),
            "max": float(meta_df["logP"].max()),
            "mean": float(meta_df["logP"].mean()),
            "median": float(meta_df["logP"].median()),
            "std": float(meta_df["logP"].std()),
        }
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== DONE ==========")
    print(f"[INFO] Encoded latent: {latent_path}")
    print(f"[INFO] Metadata:       {meta_path}")
    print(f"[INFO] y logP:         {y_path}")
    print(f"[INFO] Summary:        {summary_path}")
    print(f"[INFO] Encoded: {len(latents)} / {len(smiles)}")


if __name__ == "__main__":
    main()