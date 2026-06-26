#!/usr/bin/env python3
import argparse
import json
import math
import multiprocessing as mp
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from export_sweetdb_all_to_mol2 import (
    convert_one,
    safe_name,
    validate_mol2,
)


def output_path_for(row_number, row, mol2_dir):
    return mol2_dir / (
        f"row_{row_number:04d}_ID_{int(row['ID']):04d}_"
        f"{safe_name(row['Name'])}.mol2"
    )


def fast_convert(row_number, row_dict, mol2_dir_text, obabel_text):
    row = pd.Series(row_dict)
    mol2_dir = Path(mol2_dir_text)
    output_path = output_path_for(row_number, row, mol2_dir)
    source = Chem.MolFromSmiles(str(row["Smiles"]))
    if source is None:
        raise ValueError("RDKit SMILES parsing failed")
    heavy_atoms = source.GetNumHeavyAtoms()
    mol = Chem.AddHs(source)
    params = AllChem.ETKDGv3()
    params.randomSeed = 30300000 + int(row["ID"])
    params.maxIterations = 200
    params.useRandomCoords = True
    params.pruneRmsThresh = -1.0
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ValueError(f"fast random-coordinate 3D embedding failed: {status}")

    optimization = "none_large_molecule"
    optimization_status = None
    max_iters = 80 if heavy_atoms >= 65 else 250
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            optimization_status = int(
                AllChem.MMFFOptimizeMolecule(
                    mol, mmffVariant="MMFF94s", maxIters=max_iters
                )
            )
            optimization = f"MMFF94s_limited_{max_iters}"
        elif AllChem.UFFHasAllMoleculeParams(mol):
            optimization_status = int(
                AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
            )
            optimization = f"UFF_limited_{max_iters}"
    except Exception as exc:
        optimization = f"limited_optimization_warning:{type(exc).__name__}"

    mol.SetProp("_Name", output_path.stem)
    mol.SetProp("SweetDB_ID", str(int(row["ID"])))
    mol.SetProp("SweetDB_CSV_Row_1Based", str(row_number))
    mol.SetProp("SweetDB_Name", str(row["Name"]))
    mol.SetProp("SweetDB_logSw", str(row["logSw"]))
    mol.SetProp("SweetDB_SMILES", str(row["Smiles"]))
    with tempfile.TemporaryDirectory(prefix="sweetdb_fast_mol2_") as tmp:
        sdf_path = Path(tmp) / f"{output_path.stem}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        completed = subprocess.run(
            [
                obabel_text,
                str(sdf_path),
                "-O",
                str(output_path),
                "--partialcharge",
                "gasteiger",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"Open Babel failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
    atoms = validate_mol2(output_path, mol.GetNumAtoms())
    return {
        "csv_row_1based": row_number,
        "csv_index_0based": row_number - 1,
        "ID": int(row["ID"]),
        "Name": row["Name"],
        "logSw": row["logSw"],
        "Smiles": row["Smiles"],
        "mol2_filename": output_path.name,
        "atom_count_with_hydrogens": atoms,
        "embed_method": "ETKDGv3_random_coords_limited",
        "optimization": optimization,
        "optimization_status": optimization_status,
        "partial_charge_method": "Gasteiger (Open Babel)",
        "status": "success_fast_fallback",
    }


def child_worker(queue, mode, row_number, row_dict, mol2_dir, obabel):
    try:
        row = pd.Series(row_dict)
        if mode == "standard":
            result = convert_one(
                row_number, row, Path(mol2_dir), Path(obabel)
            )
        else:
            result = fast_convert(
                row_number, row_dict, mol2_dir, obabel
            )
        queue.put(("ok", result))
    except Exception as exc:
        queue.put(("error", type(exc).__name__, str(exc)))


def run_with_timeout(mode, row_number, row, mol2_dir, obabel, timeout):
    context = mp.get_context("fork")
    queue = context.Queue()
    process = context.Process(
        target=child_worker,
        args=(
            queue,
            mode,
            row_number,
            row.to_dict(),
            str(mol2_dir),
            str(obabel),
        ),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(10)
        return None, f"{mode} timed out after {timeout}s"
    if queue.empty():
        return None, f"{mode} exited without a result (code {process.exitcode})"
    payload = queue.get()
    if payload[0] == "ok":
        return payload[1], None
    return None, f"{payload[1]}: {payload[2]}"


def existing_record(row_number, row, output_path):
    source = Chem.MolFromSmiles(str(row["Smiles"]))
    if source is None:
        raise ValueError("RDKit SMILES parsing failed")
    expected = Chem.AddHs(source).GetNumAtoms()
    atoms = validate_mol2(output_path, expected)
    return {
        "csv_row_1based": row_number,
        "csv_index_0based": row_number - 1,
        "ID": int(row["ID"]),
        "Name": row["Name"],
        "logSw": row["logSw"],
        "Smiles": row["Smiles"],
        "mol2_filename": output_path.name,
        "atom_count_with_hydrogens": atoms,
        "embed_method": "ETKDGv3_existing_validated",
        "optimization": "MMFF94s_or_UFF_existing",
        "optimization_status": None,
        "partial_charge_method": "Gasteiger (Open Babel)",
        "status": "success_existing_validated",
    }


def write_zip(output_dir):
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
    mol2_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_csv, output_dir / input_csv.name)
    frame = pd.read_csv(input_csv)
    records = []
    failures = []

    for zero_index, row in frame.iterrows():
        row_number = zero_index + 1
        output_path = output_path_for(row_number, row, mol2_dir)
        try:
            if output_path.exists():
                record = existing_record(row_number, row, output_path)
            else:
                heavy_atoms = Chem.MolFromSmiles(str(row["Smiles"])).GetNumHeavyAtoms()
                if heavy_atoms >= 65:
                    record, error = run_with_timeout(
                        "fast", row_number, row, mol2_dir, obabel, 240
                    )
                else:
                    record, error = run_with_timeout(
                        "standard", row_number, row, mol2_dir, obabel, 180
                    )
                    if record is None:
                        record, fallback_error = run_with_timeout(
                            "fast", row_number, row, mol2_dir, obabel, 240
                        )
                        if record is None:
                            error = f"{error}; fallback: {fallback_error}"
                if record is None:
                    raise RuntimeError(error)
            records.append(record)
        except Exception as exc:
            if output_path.exists():
                output_path.unlink()
            failures.append(
                {
                    "csv_row_1based": row_number,
                    "csv_index_0based": zero_index,
                    "ID": row.get("ID"),
                    "Name": row.get("Name"),
                    "logSw": row.get("logSw"),
                    "Smiles": row.get("Smiles"),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if row_number % 10 == 0 or row_number == len(frame):
            print(
                f"[{row_number}/{len(frame)}] success={len(records)} "
                f"failed={len(failures)}",
                flush=True,
            )

    pd.DataFrame(records).sort_values("csv_row_1based").to_csv(
        output_dir / "SweetDB_mol2_manifest.csv", index=False
    )
    pd.DataFrame(failures).to_csv(
        output_dir / "SweetDB_mol2_failures.csv", index=False
    )
    metadata = {
        "source_csv": str(input_csv),
        "source_rows": len(frame),
        "successful_mol2": len(records),
        "failed_rows": len(failures),
        "hydrogens": "explicit",
        "standard_geometry": "ETKDGv3 followed by MMFF94s or UFF",
        "large_or_timeout_fallback": (
            "ETKDGv3 random-coordinate embedding with limited MMFF94s/UFF iterations"
        ),
        "partial_charge_method": "Gasteiger (Open Babel)",
        "filename_scheme": "row_####_ID_####_sanitized-name.mol2",
    }
    (output_dir / "README_conversion.json").write_text(
        json.dumps(metadata, indent=2)
    )
    zip_path = write_zip(output_dir)
    print(json.dumps(metadata, indent=2))
    print(f"ZIP: {zip_path}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
