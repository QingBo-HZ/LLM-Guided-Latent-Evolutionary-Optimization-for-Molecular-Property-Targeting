#!/usr/bin/env python3
import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from rdkit import Chem

from export_sweetdb_all_to_mol2 import safe_name, validate_mol2


def output_path(row_number, row, mol2_dir):
    return mol2_dir / (
        f"row_{row_number:04d}_ID_{int(row['ID']):04d}_"
        f"{safe_name(row['Name'])}.mol2"
    )


def repair(row_number, row, mol2_dir, obabel):
    target = output_path(row_number, row, mol2_dir)
    with tempfile.TemporaryDirectory(prefix="sweetdb_obabel_repair_") as tmp:
        smiles_path = Path(tmp) / "input.smi"
        smiles_path.write_text(
            f"{row['Smiles']} {target.stem}\n", encoding="utf-8"
        )
        completed = subprocess.run(
            [
                str(obabel),
                "-ismi",
                str(smiles_path),
                "-omol2",
                "-O",
                str(target),
                "-h",
                "--gen3d",
                "--partialcharge",
                "gasteiger",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0 or not target.exists():
            raise RuntimeError(
                f"Open Babel gen3d failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
    source = Chem.MolFromSmiles(str(row["Smiles"]))
    expected_atoms = Chem.AddHs(source).GetNumAtoms()
    atom_count = validate_mol2(target, expected_atoms)
    return {
        "csv_row_1based": row_number,
        "csv_index_0based": row_number - 1,
        "ID": int(row["ID"]),
        "Name": row["Name"],
        "logSw": row["logSw"],
        "Smiles": row["Smiles"],
        "mol2_filename": target.name,
        "atom_count_with_hydrogens": atom_count,
        "embed_method": "OpenBabel_gen3d_repair",
        "optimization": "OpenBabel_gen3d_default_geometry_refinement",
        "optimization_status": None,
        "partial_charge_method": "Gasteiger (Open Babel)",
        "status": "success_openbabel_gen3d_repair",
    }


def rewrite_zip(output_dir):
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, output_dir.name / path.relative_to(output_dir))
    return zip_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--obabel", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    mol2_dir = output_dir / "mol2"
    obabel = Path(args.obabel).resolve()
    source = pd.read_csv(input_csv)
    manifest_path = output_dir / "SweetDB_mol2_manifest.csv"
    failures_path = output_dir / "SweetDB_mol2_failures.csv"
    manifest = pd.read_csv(manifest_path)
    failures = pd.read_csv(failures_path)

    repaired = []
    remaining = []
    for _, failure in failures.iterrows():
        row_number = int(failure["csv_row_1based"])
        row = source.iloc[row_number - 1]
        try:
            repaired.append(repair(row_number, row, mol2_dir, obabel))
            print(f"repaired row={row_number} ID={int(row['ID'])}", flush=True)
        except Exception as exc:
            record = failure.to_dict()
            record["error"] = (
                f"{record.get('error', '')}; OpenBabel repair: "
                f"{type(exc).__name__}: {exc}"
            )
            remaining.append(record)

    combined = pd.concat(
        [manifest, pd.DataFrame(repaired)], ignore_index=True
    ).sort_values("csv_row_1based")
    combined.to_csv(manifest_path, index=False)
    pd.DataFrame(remaining, columns=failures.columns).to_csv(
        failures_path, index=False
    )
    metadata_path = output_dir / "README_conversion.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["successful_mol2"] = len(combined)
    metadata["failed_rows"] = len(remaining)
    metadata["repair_method"] = (
        "Open Babel gen3d with explicit hydrogens for RDKit embedding failures"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))
    zip_path = rewrite_zip(output_dir)
    print(json.dumps(metadata, indent=2))
    print(f"ZIP: {zip_path}")
    if remaining:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
