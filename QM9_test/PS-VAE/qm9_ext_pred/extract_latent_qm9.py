#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
import numpy as np
import torch
from tqdm import tqdm

PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))

print("[DEBUG] sys.path appended", flush=True)
print("[DEBUG] importing PSVAEModel...", flush=True)
from pl_models import PSVAEModel

print("[DEBUG] importing chem utils...", flush=True)
from utils.chem_utils import smiles2molecule, GeneralVocab
from data.mol_bpe import Tokenizer
from rdkit.Chem.rdchem import BondType
import torch.serialization

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
torch.serialization.add_safe_globals(SAFE_GLOBALS)
print(f"[DEBUG] registered safe globals: {[str(x) for x in SAFE_GLOBALS]}", flush=True)

QM9_PROPS = ["homo", "lumo", "gap", "u0", "u298", "h298", "g298"]


def load_model(ckpt_path: str, gpu: int):
    print(f"[DEBUG] load_model ckpt={ckpt_path}", flush=True)

    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print(f"[DEBUG] using device={device}", flush=True)
    print("[DEBUG] loading checkpoint...", flush=True)
    model = PSVAEModel.load_from_checkpoint(ckpt_path)
    print("[DEBUG] checkpoint loaded", flush=True)

    model.to(device)
    model.eval()
    print("[DEBUG] model moved to device and set eval", flush=True)
    return model, device


def main(args):
    print("[DEBUG] entering main()", flush=True)
    print(f"[DEBUG] args = {args}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[DEBUG] ensured out_dir = {args.out_dir}", flush=True)

    model, device = load_model(args.ckpt, args.gpu)

    x_list = []
    y_list = []
    meta_rows = []

    print(f"[DEBUG] opening csv: {args.csv_path}", flush=True)
    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        print(f"[DEBUG] csv header = {reader.fieldnames}", flush=True)

        required_cols = ["mol_id", "smiles"] + QM9_PROPS
        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing column '{col}' in {args.csv_path}. Found: {reader.fieldnames}")

        count_total = 0
        count_valid = 0
        count_fail_mol = 0
        count_fail_z = 0

        for row in tqdm(reader, desc=f"Extracting latent from {os.path.basename(args.csv_path)}"):
            count_total += 1

            smi = row["smiles"].strip()
            mol = smiles2molecule(smi, kekulize=True)
            if mol is None:
                count_fail_mol += 1
                continue

            try:
                with torch.no_grad():
                    z = model.get_z_from_mol(mol).detach().cpu().numpy()
            except Exception as e:
                count_fail_z += 1
                if count_fail_z <= 5:
                    print(f"[WARN] get_z_from_mol failed for smiles={smi}, err={e}", flush=True)
                continue

            props = [float(row[p]) for p in QM9_PROPS]

            x_list.append(z)
            y_list.append(props)
            count_valid += 1

            meta_rows.append({
                "mol_id": row["mol_id"],
                "smiles": smi,
                **{p: row[p] for p in QM9_PROPS}
            })

            if count_total % 100 == 0:
                print(
                    f"[DEBUG] processed={count_total}, valid={count_valid}, "
                    f"fail_mol={count_fail_mol}, fail_z={count_fail_z}",
                    flush=True
                )

    print("[DEBUG] finished csv loop", flush=True)
    print(f"[DEBUG] total={count_total}, valid={count_valid}, fail_mol={count_fail_mol}, fail_z={count_fail_z}", flush=True)

    if len(x_list) == 0:
        raise RuntimeError(f"No valid molecules extracted from {args.csv_path}")

    X = np.stack(x_list, axis=0).astype(np.float32)
    Y = np.stack(y_list, axis=0).astype(np.float32)

    split_name = args.split_name

    x_path = os.path.join(args.out_dir, f"x_{split_name}.npy")
    y_path = os.path.join(args.out_dir, f"y_{split_name}.npy")
    meta_path = os.path.join(args.out_dir, f"meta_{split_name}.csv")

    np.save(x_path, X)
    np.save(y_path, Y)

    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mol_id", "smiles"] + QM9_PROPS)
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"[DEBUG] Saved X: {x_path}, shape={X.shape}", flush=True)
    print(f"[DEBUG] Saved Y: {y_path}, shape={Y.shape}", flush=True)
    print(f"[DEBUG] Saved meta: {meta_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained PS-VAE checkpoint")
    parser.add_argument("--csv_path", type=str, required=True, help="Input CSV with smiles and QM9 labels")
    parser.add_argument("--split_name", type=str, required=True, choices=["train", "valid", "test"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, use -1 for CPU")
    args = parser.parse_args()
    main(args)