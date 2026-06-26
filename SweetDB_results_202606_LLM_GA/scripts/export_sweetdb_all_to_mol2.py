#!/usr/bin/env python3
import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def safe_name(value, limit=48):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return (text or "unnamed")[:limit]


def embed_3d(smiles, seed):
    source = Chem.MolFromSmiles(smiles)
    if source is None:
        raise ValueError("RDKit SMILES parsing failed")
    mol = Chem.AddHs(source)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.maxIterations = 1000
    params.pruneRmsThresh = 0.1
    params.useSmallRingTorsions = True
    status = AllChem.EmbedMolecule(mol, params)
    method = "ETKDGv3"
    if status != 0:
        params.useRandomCoords = True
        params.randomSeed = int(seed) + 7919
        status = AllChem.EmbedMolecule(mol, params)
        method = "ETKDGv3_random_coords"
    if status != 0:
        raise ValueError(f"3D embedding failed with status {status}")

    optimization = "none"
    optimization_status = None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            optimization_status = int(
                AllChem.MMFFOptimizeMolecule(
                    mol, mmffVariant="MMFF94s", maxIters=1500
                )
            )
            optimization = "MMFF94s"
        elif AllChem.UFFHasAllMoleculeParams(mol):
            optimization_status = int(AllChem.UFFOptimizeMolecule(mol, maxIters=1500))
            optimization = "UFF"
    except Exception as exc:
        optimization = f"optimization_warning:{type(exc).__name__}"

    conformer = mol.GetConformer()
    if not conformer.Is3D():
        raise ValueError("generated conformer is not marked as 3D")
    for atom_index in range(mol.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
            raise ValueError("non-finite 3D coordinate")
    return mol, method, optimization, optimization_status


def validate_mol2(path, expected_atoms):
    lines = path.read_text(errors="replace").splitlines()
    try:
        molecule_index = lines.index("@<TRIPOS>MOLECULE")
        atom_index = lines.index("@<TRIPOS>ATOM")
    except ValueError as exc:
        raise ValueError("missing required TRIPOS section") from exc
    if molecule_index + 2 >= len(lines):
        raise ValueError("truncated MOL2 header")
    counts = lines[molecule_index + 2].split()
    if not counts:
        raise ValueError("missing MOL2 atom count")
    declared_atoms = int(counts[0])
    if declared_atoms != expected_atoms:
        raise ValueError(
            f"atom count mismatch: expected {expected_atoms}, MOL2 has {declared_atoms}"
        )
    atom_lines = []
    for line in lines[atom_index + 1 :]:
        if line.startswith("@<TRIPOS>"):
            break
        if line.strip():
            atom_lines.append(line)
    if len(atom_lines) != declared_atoms:
        raise ValueError(
            f"ATOM section mismatch: declared {declared_atoms}, found {len(atom_lines)}"
        )
    for line in atom_lines:
        fields = line.split()
        if len(fields) < 9:
            raise ValueError("MOL2 atom line lacks partial-charge column")
        for value in fields[2:5] + [fields[-1]]:
            if not math.isfinite(float(value)):
                raise ValueError("non-finite coordinate or charge")
    return declared_atoms


def convert_one(row_number, row, mol2_dir, obabel):
    source_id = int(row["ID"])
    name = str(row["Name"])
    smiles = str(row["Smiles"])
    stem = (
        f"row_{row_number:04d}_ID_{source_id:04d}_"
        f"{safe_name(name)}"
    )
    output_path = mol2_dir / f"{stem}.mol2"
    mol, embed_method, optimization, optimization_status = embed_3d(
        smiles, seed=20260000 + source_id
    )
    mol.SetProp("_Name", stem)
    mol.SetProp("SweetDB_ID", str(source_id))
    mol.SetProp("SweetDB_CSV_Row_1Based", str(row_number))
    mol.SetProp("SweetDB_Name", name)
    mol.SetProp("SweetDB_logSw", str(row["logSw"]))
    mol.SetProp("SweetDB_SMILES", smiles)

    with tempfile.TemporaryDirectory(prefix="sweetdb_mol2_") as tmp:
        sdf_path = Path(tmp) / f"{stem}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        command = [
            str(obabel),
            str(sdf_path),
            "-O",
            str(output_path),
            "--partialcharge",
            "gasteiger",
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=180
        )
        if completed.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"Open Babel failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
    atom_count = validate_mol2(output_path, mol.GetNumAtoms())
    return {
        "csv_row_1based": row_number,
        "csv_index_0based": row_number - 1,
        "ID": source_id,
        "Name": name,
        "logSw": row["logSw"],
        "Smiles": smiles,
        "mol2_filename": output_path.name,
        "atom_count_with_hydrogens": atom_count,
        "embed_method": embed_method,
        "optimization": optimization,
        "optimization_status": optimization_status,
        "partial_charge_method": "Gasteiger (Open Babel)",
        "status": "success",
    }


def make_zip(output_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != zip_path.resolve():
                archive.write(path, Path(output_dir.name) / path.relative_to(output_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--obabel", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    obabel = Path(args.obabel).resolve()
    mol2_dir = output_dir / "mol2"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    mol2_dir.mkdir(parents=True)

    frame = pd.read_csv(input_csv)
    required = {"ID", "Name", "logSw", "Smiles"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    shutil.copy2(input_csv, output_dir / input_csv.name)
    successes = []
    failures = []
    total = len(frame)
    for zero_index, row in frame.iterrows():
        row_number = zero_index + 1
        try:
            successes.append(
                convert_one(row_number, row, mol2_dir, obabel)
            )
        except Exception as exc:
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
        if row_number % 25 == 0 or row_number == total:
            print(
                f"[{row_number}/{total}] success={len(successes)} "
                f"failed={len(failures)}",
                flush=True,
            )

    success_frame = pd.DataFrame(successes)
    failure_frame = pd.DataFrame(failures)
    success_frame.to_csv(output_dir / "SweetDB_mol2_manifest.csv", index=False)
    failure_frame.to_csv(output_dir / "SweetDB_mol2_failures.csv", index=False)
    metadata = {
        "source_csv": str(input_csv),
        "source_rows": total,
        "successful_mol2": len(successes),
        "failed_rows": len(failures),
        "hydrogens": "explicit",
        "conformer_generation": "RDKit ETKDGv3; random-coordinate retry on failure",
        "geometry_optimization": "MMFF94s when parameterized, otherwise UFF",
        "mol2_writer": f"Open Babel at {obabel}",
        "partial_charge_method": "Gasteiger",
        "filename_scheme": "row_####_ID_####_sanitized-name.mol2",
        "index_preservation": {
            "csv_row_1based": "physical data-row position, header excluded",
            "csv_index_0based": "pandas/original zero-based row index",
            "ID": "original SweetenersDB_v2.0.csv ID column",
        },
    }
    (output_dir / "README_conversion.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    make_zip(output_dir, zip_path)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"ZIP: {zip_path}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
