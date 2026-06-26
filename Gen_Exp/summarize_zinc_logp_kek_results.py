#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import pandas as pd

KEYS = [
    'decode_latent_validity_final',
    'decode_success_final',
    'archive_unique_valid_molecules',
    'archive_rdkit_success_unique',
    'archive_rdkit_success_rate_over_unique',
    'best_rdkit_logP',
    'best_rdkit_abs_error',
    'top10_rdkit_abs_error_mean',
    'diversity_unique_valid',
    'latent_pred_success_rate_final',
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', required=True)
    ap.add_argument('--versions', nargs='+', required=True)
    ap.add_argument('--out_prefix', required=True)
    args = ap.parse_args()
    root = Path(args.results_root)
    rows = []
    for version_dir in args.versions:
        p = root / version_dir / 'summary_decode_aware.json'
        if not p.exists():
            raise FileNotFoundError(p)
        d = json.loads(p.read_text(encoding='utf-8'))
        row = {'version_dir': version_dir, 'version': d.get('version'), 'seed': d.get('seed'), 'pop_size': d.get('pop_size'), 'n_gen_completed': d.get('n_gen_completed'), 'best_smiles_rdkit': d.get('best_smiles_rdkit')}
        for k in KEYS:
            row[k] = d.get(k)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = root / f'{args.out_prefix}_summary.csv'
    df.to_csv(out, index=False)
    agg_rows = []
    for k in KEYS:
        if k in df.columns and pd.api.types.is_numeric_dtype(df[k]):
            agg_rows.append({'metric': k, 'mean': df[k].mean(), 'std': df[k].std()})
    agg = pd.DataFrame(agg_rows)
    agg_out = root / f'{args.out_prefix}_mean_std.csv'
    agg.to_csv(agg_out, index=False)
    print(df.to_string(index=False))
    print('\nMEAN/STD')
    print(agg.to_string(index=False))
    print(f'\nsaved {out}')
    print(f'saved {agg_out}')

if __name__ == '__main__':
    main()
