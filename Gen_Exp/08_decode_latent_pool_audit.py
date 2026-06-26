#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, json, random, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch, torch.serialization
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType
RDLogger.DisableLog('rdApp.*')

PSVAE_ROOT='/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE'
sys.path.append(os.path.join(PSVAE_ROOT,'src'))
from pl_models import PSVAEModel
from utils.chem_utils import molecule2smiles, GeneralVocab
from data.mol_bpe import Tokenizer
SAFE_GLOBALS=[Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization,'add_safe_globals'):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_model(ckpt, device):
    old=torch.load
    def patched(*a, **kw):
        kw['weights_only']=False
        return old(*a, **kw)
    torch.load=patched
    try:
        m=PSVAEModel.load_from_checkpoint(ckpt, map_location=device)
    finally:
        torch.load=old
    return m.eval().to(device)

def canonicalize(smi):
    try:
        mol=Chem.MolFromSmiles(str(smi))
        if mol is None: return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None

def decode_once(model, z, device, max_atom_num, add_edge_th, temperature):
    try:
        with torch.no_grad():
            zt=torch.tensor(z, dtype=torch.float32, device=device)
            mol=model.inference_single_z(zt, max_atom_num=max_atom_num, add_edge_th=add_edge_th, temperature=temperature)
            smi=molecule2smiles(model.return_data_to_mol(mol))
        can=canonicalize(smi)
        if can is None: return None, 'invalid_smiles'
        return can, 'ok'
    except Exception as e:
        msg=str(e).strip().replace('\n',' ')
        return None, msg[:160] if msg else type(e).__name__

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE/ckpts/lightning_logs/version_8_zinc/checkpoints/epoch=19-step=124740.ckpt')
    ap.add_argument('--latent', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_samples', type=int, default=200)
    ap.add_argument('--attempts', type=int, default=3)
    ap.add_argument('--max_atom_num', type=int, default=80)
    ap.add_argument('--add_edge_th', type=float, default=0.45)
    ap.add_argument('--temperature', type=float, default=0.30)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    set_seed(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device=torch.device(f'cuda:{args.gpu}' if args.gpu>=0 and torch.cuda.is_available() else 'cpu')
    model=load_model(args.ckpt, device)
    latent=np.load(args.latent).astype(np.float32)
    rng=np.random.default_rng(args.seed)
    n=min(args.n_samples, len(latent))
    idx=rng.choice(len(latent), size=n, replace=False)
    rows=[]; err=Counter(); valid_attempts=0; total_attempts=0; any_valid=0
    for rank,i in enumerate(idx):
        latent_valid=False
        first=None
        for a in range(args.attempts):
            total_attempts += 1
            smi,status=decode_once(model, latent[i], device, args.max_atom_num, args.add_edge_th, args.temperature)
            ok=smi is not None
            if ok:
                valid_attempts += 1
                latent_valid=True
                if first is None: first=smi
            else:
                err[status]+=1
            rows.append({'sample_rank':rank,'latent_idx':int(i),'attempt':a,'valid':ok,'smiles':smi,'status':status})
        any_valid += int(latent_valid)
        if (rank+1)%50==0:
            print(f'[AUDIT] {rank+1}/{n} attempt_valid={valid_attempts}/{total_attempts} any={any_valid}/{rank+1}', flush=True)
    summary={
        'latent': args.latent, 'n_samples': n, 'attempts_per_latent': args.attempts,
        'decode_attempts': total_attempts, 'valid_decode_attempts': valid_attempts,
        'single_attempt_validity': valid_attempts/total_attempts if total_attempts else 0.0,
        'latent_any_valid_count': any_valid,
        'latent_any_valid_rate': any_valid/n if n else 0.0,
        'max_atom_num': args.max_atom_num, 'add_edge_th': args.add_edge_th, 'temperature': args.temperature,
        'top_errors': err.most_common(20),
    }
    pd.DataFrame(rows).to_csv(out/'latent_pool_decode_records.csv', index=False)
    (out/'latent_pool_decode_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if __name__=='__main__': main()
