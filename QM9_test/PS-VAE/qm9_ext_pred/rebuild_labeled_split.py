#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import pandas as pd


REQUIRED_COLS = [
    "mol_id", "smiles",
    "homo", "lumo", "gap",
    "u0", "u298", "h298", "g298"
]


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.master_csv)

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in master csv")

    # 按 smiles 建索引
    # 如果有重复 smiles，这里默认保留第一条
    df_unique = df.drop_duplicates(subset=["smiles"], keep="first").copy()
    smi2row = {row["smiles"]: row for _, row in df_unique.iterrows()}

    with open(args.split_file, "r", encoding="utf-8") as f:
        split_smiles = [line.strip() for line in f if line.strip()]

    rows = []
    missing = []

    for smi in split_smiles:
        if smi in smi2row:
            row = smi2row[smi]
            rows.append({k: row[k] for k in REQUIRED_COLS})
        else:
            missing.append(smi)

    out_csv = os.path.join(args.out_dir, args.out_name)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved labeled split -> {out_csv}")
    print(f"Matched: {len(rows)}")
    print(f"Missing: {len(missing)}")

    if len(missing) > 0:
        miss_path = os.path.join(args.out_dir, args.out_name.replace(".csv", "_missing.txt"))
        with open(miss_path, "w", encoding="utf-8") as f:
            for smi in missing:
                f.write(smi + "\n")
        print(f"Missing smiles saved to -> {miss_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", type=str, required=True, help="Full QM9 csv with labels")
    parser.add_argument("--split_file", type=str, required=True, help="Split txt/csv containing one smiles per line")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--out_name", type=str, required=True, help="Output labeled csv name")
    args = parser.parse_args()
    main(args)