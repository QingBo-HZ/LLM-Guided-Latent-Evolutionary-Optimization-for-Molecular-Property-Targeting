#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import traceback
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType

RDLogger.DisableLog("rdApp.*")

# =========================
# PS-VAE 根目录
# =========================
PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))

from pl_models import PSVAEModel
from utils.chem_utils import smiles2molecule, GeneralVocab
from data.mol_bpe import Tokenizer
import torch.serialization

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)


TEST_SMILES = [
    "c1ccccc1",
    "C1=CC=CC=C1",
    "N#Cc1ncnc(=O)[nH]1",
    "O=c1nc[nH]c(=O)[nH]1",
    "N#Cc1nc(=O)nc[nH]1",
    "NC(=O)CNC(=O)C=O",
    "N=CC(=O)C1=NC(=O)O1",
    "O=CN=C1C=NC(=O)O1",
    "N#CNC=O",
    "COC(=O)C#N",
    "N#CC=O",
    "NCC(N)=O",
    "O=CNCO",
    "CN(C=O)C=O",
    "CC(=O)N=C=O",
    "NC1=NC(=O)OC1=O",
    "N#CC1=NC(=O)NC1=O",
]


def count_aromatic_bonds(mol):
    if mol is None:
        return None
    return sum(1 for b in mol.GetBonds() if b.GetBondType() == BondType.AROMATIC)


def count_aromatic_atoms(mol):
    if mol is None:
        return None
    return sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())


def bond_types(mol):
    if mol is None:
        return []
    return sorted(set(str(b.GetBondType()) for b in mol.GetBonds()))


def atom_symbols(mol):
    if mol is None:
        return []
    return sorted(set(a.GetSymbol() for a in mol.GetAtoms()))


def has_aromatic_bond(mol):
    if mol is None:
        return False
    return any(b.GetBondType() == BondType.AROMATIC for b in mol.GetBonds())


def force_kekulize_mol(smi: str):
    """
    Robustly convert aromatic SMILES to PS-VAE-compatible Kekulé mol.

    Return:
        mol, reason
    """
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None, "MolFromSmiles failed"

    try:
        Chem.RemoveStereochemistry(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as e:
        return None, f"Kekulize failed at first pass: {repr(e)}"

    try:
        kek_smi = Chem.MolToSmiles(
            mol,
            canonical=True,
            kekuleSmiles=True
        )
    except Exception as e:
        return None, f"MolToSmiles(kekuleSmiles=True) failed: {repr(e)}"

    mol2 = Chem.MolFromSmiles(kek_smi)
    if mol2 is None:
        return None, f"MolFromSmiles(kekule_smiles) failed: {kek_smi}"

    try:
        Chem.RemoveStereochemistry(mol2)
        Chem.Kekulize(mol2, clearAromaticFlags=True)
    except Exception as e:
        return None, f"Kekulize failed at second pass: {repr(e)}"

    aromatic_bond_count = count_aromatic_bonds(mol2)
    aromatic_atom_count = count_aromatic_atoms(mol2)

    if aromatic_bond_count and aromatic_bond_count > 0:
        return None, f"AROMATIC bonds remaining after force kekulize: {aromatic_bond_count}"

    if aromatic_atom_count and aromatic_atom_count > 0:
        return None, f"AROMATIC atoms remaining after force kekulize: {aromatic_atom_count}"

    return mol2, "ok"


def try_encode(model, mol):
    try:
        if mol is None:
            return False, None, "mol is None"

        # The encoder preprocessing now normalizes aromatic molecules internally.
        # Keep aromatic diagnostics in print_mol_info(), but do not block here.
        with torch.no_grad():
            z = model.get_z_from_mol(mol).detach().cpu().numpy()

        return True, z.shape, ""

    except Exception as e:
        return False, None, repr(e) + "\n" + traceback.format_exc(limit=6)


def inspect_vocab(model):
    print("\n========== CHECKPOINT TOKENIZER / CHEM_VOCAB ==========")

    tok = getattr(model, "tokenizer", None)
    print(f"tokenizer type: {type(tok)}")

    if tok is None:
        print("[ERROR] model.tokenizer is None")
        return

    chem_vocab = getattr(tok, "chem_vocab", None)
    print(f"chem_vocab type: {type(chem_vocab)}")

    if chem_vocab is None:
        print("[ERROR] tokenizer.chem_vocab is None")
        return

    print("\nidx2bond:")
    idx2bond = getattr(chem_vocab, "idx2bond", [])
    for i, b in enumerate(idx2bond):
        print(f"  {i}: {b}")

    print("\nbond2idx keys:")
    bond2idx = getattr(chem_vocab, "bond2idx", {})
    for k, v in bond2idx.items():
        print(f"  {k}: {v}")

    print("\nSupports BondType.AROMATIC?")
    print(BondType.AROMATIC in bond2idx)

    print("\nnum_bond_type:")
    try:
        print(chem_vocab.num_bond_type())
    except Exception as e:
        print(f"[WARN] cannot call num_bond_type(): {repr(e)}")


def print_mol_info(title, mol):
    print(f"\n[{title}]")
    print(f"mol is None: {mol is None}")

    if mol is None:
        return

    try:
        print(f"canonical: {Chem.MolToSmiles(mol, canonical=True)}")
    except Exception as e:
        print(f"canonical failed: {repr(e)}")

    try:
        print(f"kekule smiles: {Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)}")
    except Exception as e:
        print(f"kekule smiles failed: {repr(e)}")

    print(f"aromatic bonds: {count_aromatic_bonds(mol)}")
    print(f"aromatic atoms: {count_aromatic_atoms(mol)}")
    print(f"bond types: {bond_types(mol)}")
    print(f"atom symbols: {atom_symbols(mol)}")


def test_one_smiles(model, smi):
    print("\n" + "=" * 100)
    print(f"SMILES: {smi}")

    # 1. RDKit raw
    mol_raw = Chem.MolFromSmiles(smi)
    print_mol_info("RDKit MolFromSmiles", mol_raw)

    # 2. 原工程 smiles2molecule(kekulize=True)
    print("\n[Original smiles2molecule(kekulize=True)]")
    mol_orig = None
    try:
        mol_orig = smiles2molecule(smi, kekulize=True)
        print_mol_info("Original smiles2molecule result", mol_orig)

        ok, shape, err = try_encode(model, mol_orig)
        print(f"encode ok: {ok}")
        print(f"z shape: {shape}")
        if not ok:
            print(f"encode error:\n{err}")

    except Exception as e:
        print(f"smiles2molecule exception: {repr(e)}")
        print(traceback.format_exc(limit=6))

    # 3. force kekulize clear aromatic flags
    print("\n[force_kekulize_mol(clearAromaticFlags=True + kekuleSmiles reparse)]")
    mol_force, reason = force_kekulize_mol(smi)
    print(f"reason: {reason}")
    print_mol_info("force_kekulize_mol result", mol_force)

    ok, shape, err = try_encode(model, mol_force)
    print(f"encode ok: {ok}")
    print(f"z shape: {shape}")
    if not ok:
        print(f"encode error:\n{err}")


def load_smiles_file(path):
    smiles_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = s.split(",")[1] if "," in s and s.lower().startswith("row_idx") is False else s
            s = s.strip().split()[0]
            if s and s.lower() not in ["smiles", "canonical_smiles"]:
                smiles_list.append(s)
    return smiles_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smiles", type=str, default=None)
    parser.add_argument("--smi_file", type=str, default=None)
    parser.add_argument("--max_test", type=int, default=50)
    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f"[INFO] loading model from: {args.ckpt}")
    print(f"[INFO] device: {device}")

    model = PSVAEModel.load_from_checkpoint(args.ckpt, map_location=device)
    model.to(device)
    model.eval()

    inspect_vocab(model)

    smiles_list = []

    if args.smiles:
        smiles_list.append(args.smiles)

    if args.smi_file:
        smiles_list.extend(load_smiles_file(args.smi_file))

    if not smiles_list:
        smiles_list = TEST_SMILES

    smiles_list = smiles_list[:args.max_test]

    print("\n========== TEST ENCODING ==========")
    print(f"n_test = {len(smiles_list)}")

    for smi in smiles_list:
        test_one_smiles(model, smi)


if __name__ == "__main__":
    main()