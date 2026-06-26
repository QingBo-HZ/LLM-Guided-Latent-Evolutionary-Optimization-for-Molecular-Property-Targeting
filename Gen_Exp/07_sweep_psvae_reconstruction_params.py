#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.serialization
from rdkit import Chem, RDLogger
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


def decode_valid(model, z, max_atom_num, add_edge_th, temperature):
    try:
        with torch.no_grad():
            mol = model.inference_single_z(
                z,
                max_atom_num=max_atom_num,
                add_edge_th=add_edge_th,
                temperature=temperature,
            )
            smi = molecule2smiles(mol)
        can = canonicalize(smi)
        if can is None:
            return None, "invalid_smiles"
        return can, "ok"
    except Exception as exc:
        msg = str(exc).strip().replace("\n", " ")
        return None, msg[:160] if msg else type(exc).__name__


def parse_float_list(s):
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def parse_int_list(s):
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def main():
    parser = argparse.ArgumentParser("Sweep PS-VAE reconstruction decode params on freshly encoded ZINC molecules")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--train_set", default="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/data/my_zinc/train/train.txt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--max_atom_nums", default="60,80,100")
    parser.add_argument("--add_edge_ths", default="0.30,0.45,0.55,0.65")
    parser.add_argument("--temperatures", default="0.20,0.30,0.50,0.70")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device)
    smiles = read_smiles(args.train_set, args.n_samples, args.seed)

    z_rows = []
    for idx, smi in enumerate(smiles):
        try:
            mol = smiles2molecule(smi, kekulize=True)
            with torch.no_grad():
                z = model.get_z_from_mol(mol).detach().to(device)
            z_rows.append((idx, smi, canonicalize(smi), z))
        except Exception as exc:
            print(f"[WARN] encode failed idx={idx}: {exc}", flush=True)
    print(f"[INFO] encoded {len(z_rows)}/{len(smiles)} molecules", flush=True)

    results = []
    examples = []
    max_atom_nums = parse_int_list(args.max_atom_nums)
    add_edge_ths = parse_float_list(args.add_edge_ths)
    temperatures = parse_float_list(args.temperatures)

    for max_atom_num in max_atom_nums:
        for add_edge_th in add_edge_ths:
            for temperature in temperatures:
                valid_attempts = 0
                total_attempts = 0
                any_valid = 0
                error_counter = Counter()
                for idx, smi, input_can, z in z_rows:
                    latent_valid = False
                    for attempt in range(args.attempts):
                        total_attempts += 1
                        decoded, status = decode_valid(model, z, max_atom_num, add_edge_th, temperature)
                        if decoded is not None:
                            valid_attempts += 1
                            latent_valid = True
                            if len(examples) < 200:
                                examples.append({
                                    "max_atom_num": max_atom_num,
                                    "add_edge_th": add_edge_th,
                                    "temperature": temperature,
                                    "idx": idx,
                                    "input_smiles": smi,
                                    "decoded_smiles": decoded,
                                })
                        else:
                            error_counter[status] += 1
                    if latent_valid:
                        any_valid += 1

                row = {
                    "max_atom_num": max_atom_num,
                    "add_edge_th": add_edge_th,
                    "temperature": temperature,
                    "attempts": args.attempts,
                    "encoded": len(z_rows),
                    "total_attempts": total_attempts,
                    "valid_attempts": valid_attempts,
                    "single_attempt_validity": valid_attempts / total_attempts if total_attempts else 0.0,
                    "latent_any_valid_count": any_valid,
                    "latent_any_valid_rate": any_valid / len(z_rows) if z_rows else 0.0,
                    "top_errors": json.dumps(error_counter.most_common(5), ensure_ascii=False),
                }
                results.append(row)
                print(
                    f"[SWEEP] atom={max_atom_num} edge={add_edge_th:.2f} temp={temperature:.2f} "
                    f"valid={row['single_attempt_validity']:.3f} any={row['latent_any_valid_rate']:.3f}",
                    flush=True,
                )

    df = pd.DataFrame(results).sort_values(
        ["single_attempt_validity", "latent_any_valid_rate"], ascending=[False, False]
    )
    df.to_csv(out_dir / "reconstruction_param_sweep.csv", index=False)
    pd.DataFrame(examples).to_csv(out_dir / "reconstruction_param_sweep_examples.csv", index=False)
    summary = {
        "ckpt": args.ckpt,
        "n_samples": len(smiles),
        "encoded": len(z_rows),
        "attempts": args.attempts,
        "best": df.iloc[0].to_dict() if len(df) else None,
    }
    (out_dir / "reconstruction_param_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
