#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


def canonicalize(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    smiles = smiles.strip()
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def make_unique_submission(input_csv: Path, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv)
    if "audit_stage" not in df.columns:
        df["audit_stage"] = stage
    df["canonical_smiles_for_docking"] = df["smiles"].map(canonicalize)
    df = df.dropna(subset=["canonical_smiles_for_docking"]).copy()
    unique = (
        df.groupby("canonical_smiles_for_docking", as_index=False)
        .agg(
            n_records=("ID", "count"),
            example_original_id=("ID", "first"),
            audit_stages=("audit_stage", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .sort_values(["audit_stages", "canonical_smiles_for_docking"])
        .reset_index(drop=True)
    )
    prefix = {"all": "DOCK", "final": "FINAL", "generation": "GEN"}.get(stage, stage.upper())
    unique["dock_id"] = [f"{prefix}_{i:05d}" for i in range(1, len(unique) + 1)]
    unique = unique[["dock_id", "canonical_smiles_for_docking", "n_records", "audit_stages", "example_original_id"]]
    unique = unique.rename(columns={"canonical_smiles_for_docking": "smiles"})

    mapping = df.merge(
        unique.rename(columns={"smiles": "canonical_smiles_for_docking"})[
            ["dock_id", "canonical_smiles_for_docking"]
        ],
        on="canonical_smiles_for_docking",
        how="left",
    )
    keep_cols = ["ID", "dock_id", "smiles", "canonical_smiles_for_docking", "audit_stage"]
    for col in ["method", "seed", "generation", "rank"]:
        if col in mapping.columns:
            keep_cols.append(col)
    mapping = mapping[keep_cols].sort_values(["dock_id", "ID"])
    return unique, mapping


def write_readme(out_dir: Path, manifest: dict[str, int]) -> None:
    readme = f"""# SweetDB v8 Docking Submission Package

Generated from:

`/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA/sweet_ga_results_0622_v8_hard_metrics/docking_audit`

## What to submit

For the broadest real-docking audit, submit:

`submit_unique_smiles_all.csv`

This file contains one row per unique canonical SMILES, so repeated molecules across generations/seeds are docked once.

## Counts

- Original all audit records: {manifest['all_original_records']}
- Unique all docking molecules: {manifest['all_unique_smiles']}
- Original final audit records: {manifest['final_original_records']}
- Unique final docking molecules: {manifest['final_unique_smiles']}
- Original generation audit records: {manifest['generation_original_records']}
- Unique generation docking molecules: {manifest['generation_unique_smiles']}

## Required returned docking format

Return a CSV with either `dock_id` or `ID`, plus a real Vina score column:

```csv
dock_id,vina_kcal_mol
DOCK_00001,-7.34
DOCK_00002,-6.91
```

Accepted score column names in the merge script:

- `vina_kcal_mol`
- `vina_affinity_kcal_mol`
- `affinity`
- `score`
- `vina`

## Mapping back to original experiment records

Use:

`record_to_docking_id_all.csv`

This maps every original generation/final audit ID back to the unique `dock_id`.
"""
    (out_dir / "README_docking_submission.md").write_text(readme, encoding="utf-8")


def zip_dir(out_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(out_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "all": audit_dir / "all_docking_input.csv",
        "final": audit_dir / "final_docking_input.csv",
        "generation": audit_dir / "generation_docking_input.csv",
    }
    manifest: dict[str, int] = {}
    for stage, input_csv in specs.items():
        unique, mapping = make_unique_submission(input_csv, stage)
        unique.to_csv(out_dir / f"submit_unique_smiles_{stage}.csv", index=False)
        mapping.to_csv(out_dir / f"record_to_docking_id_{stage}.csv", index=False)
        manifest[f"{stage}_original_records"] = int(len(mapping))
        manifest[f"{stage}_unique_smiles"] = int(len(unique))

    write_readme(out_dir, manifest)
    (out_dir / "submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = out_dir / "SweetDB_v8_unique_smiles_docking_submission.zip"
    zip_dir(out_dir, zip_path)
    print(json.dumps({"out_dir": str(out_dir), "zip": str(zip_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
