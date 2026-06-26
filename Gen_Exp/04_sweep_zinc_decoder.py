#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "64")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import sys
import json
import time
import argparse
import random
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
from utils.chem_utils import molecule2smiles, GeneralVocab
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


def load_psvae(ckpt, device):
    old_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return old_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        model = PSVAEModel.load_from_checkpoint(ckpt, map_location=device)
    finally:
        torch.load = old_torch_load
    model.to(device)
    model.eval()
    return model


def canonicalize_smiles(smi):
    try:
        if smi is None:
            return None
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def calc_logp(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return float(Descriptors.MolLogP(mol))
    except Exception:
        return None


def decode_one(model, z, device, max_atom_num, add_edge_th, temperature):
    try:
        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32, device=device)
            graph = model.inference_single_z(
                z_t,
                max_atom_num=max_atom_num,
                add_edge_th=add_edge_th,
                temperature=temperature,
            )
            mol = model.return_data_to_mol(graph)
            smi = molecule2smiles(mol)
        return canonicalize_smiles(smi)
    except Exception:
        return None


def parse_float_list(s):
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def parse_int_list(s):
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def main():
    parser = argparse.ArgumentParser("Sweep ZINC PS-VAE decoder parameters on latent samples")
    parser.add_argument("--ckpt", default="/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt")
    parser.add_argument("--latent", default="/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/train/zinc_logp_latent.npy")
    parser.add_argument("--out_dir", default="/root/autodl-tmp/sweeteners_evolve/Gen_Exp/Zinc_logP/decoder_sweep")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--max_atom_nums", default="40,60,80")
    parser.add_argument("--add_edge_ths", default="0.30,0.45,0.55,0.70")
    parser.add_argument("--temperatures", default="0.10,0.30,0.50")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] loading model: {args.ckpt}", flush=True)
    model = load_psvae(args.ckpt, device)

    latent = np.load(args.latent).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(latent))
    idx = rng.choice(len(latent), size=n, replace=False)
    sample = latent[idx]

    max_atom_nums = parse_int_list(args.max_atom_nums)
    add_edge_ths = parse_float_list(args.add_edge_ths)
    temperatures = parse_float_list(args.temperatures)

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "sample_indices": idx.tolist(),
        "max_atom_nums_list": max_atom_nums,
        "add_edge_ths_list": add_edge_ths,
        "temperatures_list": temperatures,
    })
    (out_dir / "decoder_sweep_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    examples = []
    start = time.time()
    total = len(max_atom_nums) * len(add_edge_ths) * len(temperatures)
    done = 0

    for max_atom_num in max_atom_nums:
        for add_edge_th in add_edge_ths:
            for temperature in temperatures:
                done += 1
                decoded = []
                for z in sample:
                    smi = decode_one(
                        model,
                        z,
                        device,
                        max_atom_num=max_atom_num,
                        add_edge_th=add_edge_th,
                        temperature=temperature,
                    )
                    decoded.append(smi)

                valid = [s for s in decoded if s is not None]
                unique_valid = sorted(set(valid))
                logps = [calc_logp(s) for s in valid]
                logps = [x for x in logps if x is not None]
                row = {
                    "max_atom_num": max_atom_num,
                    "add_edge_th": add_edge_th,
                    "temperature": temperature,
                    "n_samples": n,
                    "decode_success": len(valid),
                    "validity": len(valid) / n if n else 0.0,
                    "unique_valid": len(unique_valid),
                    "uniqueness": len(unique_valid) / len(valid) if valid else 0.0,
                    "mean_rdkit_logp": float(np.mean(logps)) if logps else None,
                    "median_rdkit_logp": float(np.median(logps)) if logps else None,
                    "elapsed_sec": time.time() - start,
                }
                rows.append(row)
                for smi in unique_valid[:5]:
                    examples.append({**row, "example_smiles": smi})
                print(
                    f"[{done}/{total}] max_atom={max_atom_num} edge={add_edge_th:.2f} temp={temperature:.2f} "
                    f"valid={len(valid)}/{n} ({row['validity']:.3f}) unique={len(unique_valid)}",
                    flush=True,
                )

    df = pd.DataFrame(rows).sort_values(["validity", "unique_valid"], ascending=[False, False])
    df.to_csv(out_dir / "decoder_sweep_results.csv", index=False)
    pd.DataFrame(examples).to_csv(out_dir / "decoder_sweep_examples.csv", index=False)
    print("\n[INFO] best settings:")
    print(df.head(10).to_string(index=False))
    print(f"[INFO] saved: {out_dir / 'decoder_sweep_results.csv'}")


if __name__ == "__main__":
    main()
