#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.serialization

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import BondType

RDLogger.DisableLog("rdApp.*")

PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))

from pl_models import PSVAEModel
from utils.chem_utils import smiles2molecule, molecule2smiles, GeneralVocab
from data.mol_bpe import Tokenizer

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(ckpt, device):
    old_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return old_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        model = PSVAEModel.load_from_checkpoint(ckpt, map_location=device)
    finally:
        torch.load = old_torch_load
    model.eval().to(device)
    return model


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def read_smiles(path, n_samples, seed):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line.split()[0])
    rng = random.Random(seed)
    if n_samples and n_samples < len(rows):
        rows = rng.sample(rows, n_samples)
    return rows


def decode_once(model, z, device, max_atom_num, add_edge_th, temperature):
    try:
        with torch.no_grad():
            graph = model.inference_single_z(
                z,
                max_atom_num=max_atom_num,
                add_edge_th=add_edge_th,
                temperature=temperature,
            )
            smi = molecule2smiles(graph)
        can = canonicalize(smi)
        if can is None:
            return None, "invalid_smiles"
        return can, "ok"
    except Exception as exc:
        msg = str(exc).strip().replace("\n", " ")
        return None, msg[:180] if msg else type(exc).__name__


def main():
    parser = argparse.ArgumentParser("Audit PS-VAE encode-decode validity on ZINC training molecules")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--train_set", default="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_zinc/train/train.txt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--max_atom_num", type=int, default=80)
    parser.add_argument("--add_edge_th", type=float, default=0.55)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device)
    smiles = read_smiles(args.train_set, args.n_samples, args.seed)

    records = []
    error_counter = Counter()
    encode_ok = 0
    total_attempts = 0
    valid_attempts = 0
    any_valid = 0
    exact_match = 0
    canonical_match = 0

    for idx, smi in enumerate(smiles):
        input_can = canonicalize(smi)
        try:
            mol = smiles2molecule(smi, kekulize=True)
            if mol is None:
                raise ValueError("input_mol_none")
            with torch.no_grad():
                z = model.get_z_from_mol(mol).detach().to(device)
            encode_ok += 1
        except Exception as exc:
            reason = str(exc).strip().replace("\n", " ")[:180] or type(exc).__name__
            error_counter[f"encode:{reason}"] += 1
            records.append({
                "idx": idx,
                "input_smiles": smi,
                "input_canonical": input_can,
                "encode_ok": False,
                "attempt": -1,
                "decoded_smiles": None,
                "valid": False,
                "canonical_match": False,
                "exact_match": False,
                "rdkit_logP": None,
                "status": reason,
            })
            continue

        latent_has_valid = False
        latent_has_exact = False
        latent_has_canonical = False
        for attempt in range(args.attempts):
            total_attempts += 1
            decoded, status = decode_once(
                model, z, device, args.max_atom_num, args.add_edge_th, args.temperature
            )
            valid = decoded is not None
            if valid:
                valid_attempts += 1
                latent_has_valid = True
                if decoded == input_can:
                    latent_has_canonical = True
                if decoded == smi:
                    latent_has_exact = True
                mol_dec = Chem.MolFromSmiles(decoded)
                rdkit_logp = float(Descriptors.MolLogP(mol_dec)) if mol_dec is not None else None
            else:
                error_counter[f"decode:{status}"] += 1
                rdkit_logp = None

            records.append({
                "idx": idx,
                "input_smiles": smi,
                "input_canonical": input_can,
                "encode_ok": True,
                "attempt": attempt,
                "decoded_smiles": decoded,
                "valid": valid,
                "canonical_match": bool(valid and decoded == input_can),
                "exact_match": bool(valid and decoded == smi),
                "rdkit_logP": rdkit_logp,
                "status": status,
            })

        if latent_has_valid:
            any_valid += 1
        if latent_has_exact:
            exact_match += 1
        if latent_has_canonical:
            canonical_match += 1

        if (idx + 1) % 20 == 0:
            print(
                f"[AUDIT] processed={idx + 1}, encode_ok={encode_ok}, "
                f"attempt_valid={valid_attempts}/{total_attempts}, any_valid={any_valid}/{encode_ok}",
                flush=True,
            )

    summary = {
        "ckpt": args.ckpt,
        "train_set": args.train_set,
        "n_samples": len(smiles),
        "attempts_per_latent": args.attempts,
        "max_atom_num": args.max_atom_num,
        "add_edge_th": args.add_edge_th,
        "temperature": args.temperature,
        "encode_ok": int(encode_ok),
        "encode_rate": float(encode_ok / len(smiles)) if smiles else 0.0,
        "decode_attempts": int(total_attempts),
        "valid_decode_attempts": int(valid_attempts),
        "single_attempt_validity": float(valid_attempts / total_attempts) if total_attempts else 0.0,
        "latent_any_valid_count": int(any_valid),
        "latent_any_valid_rate": float(any_valid / encode_ok) if encode_ok else 0.0,
        "latent_canonical_match_count": int(canonical_match),
        "latent_canonical_match_rate": float(canonical_match / encode_ok) if encode_ok else 0.0,
        "latent_exact_match_count": int(exact_match),
        "latent_exact_match_rate": float(exact_match / encode_ok) if encode_ok else 0.0,
        "top_errors": error_counter.most_common(20),
    }

    pd.DataFrame(records).to_csv(out_dir / "reconstruction_audit_records.csv", index=False)
    (out_dir / "reconstruction_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
