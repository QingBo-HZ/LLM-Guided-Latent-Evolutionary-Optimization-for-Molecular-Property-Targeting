#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import sys
import csv
import argparse
import numpy as np
import torch

PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))

from pl_models import PSVAEModel
from utils.chem_utils import molecule2smiles, GeneralVocab
from data.mol_bpe import Tokenizer
from rdkit.Chem.rdchem import BondType
import torch.serialization

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
torch.serialization.add_safe_globals(SAFE_GLOBALS)
print(f"[DEBUG] registered safe globals: {[str(x) for x in SAFE_GLOBALS]}", flush=True)


def load_model(ckpt_path: str, gpu: int):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print(f"[DEBUG] loading checkpoint from: {ckpt_path}", flush=True)
    model = PSVAEModel.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    return model, device


def main(args):
    model, device = load_model(args.ckpt, args.gpu)

    Z = np.load(args.z_path)
    if Z.ndim == 1:
        Z = Z[None, :]

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "smiles"])

        for i, z in enumerate(Z):
            z_t = torch.tensor(z, dtype=torch.float32, device=device)
            try:
                graph = model.inference_single_z(
                    z_t,
                    max_atom_num=args.max_atom_num,
                    add_edge_th=args.add_edge_th,
                    temperature=args.temperature
                )
                mol = model.return_data_to_mol(graph)
                smi = molecule2smiles(mol)
            except Exception as e:
                print(f"[WARN] decode failed at idx={i}, err={e}", flush=True)
                smi = None

            writer.writerow([i, smi])

    print(f"Saved decoded molecules -> {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--z_path", type=str, required=True, help="npy file of latent vectors")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_atom_num", type=int, default=60)
    parser.add_argument("--add_edge_th", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    main(args)