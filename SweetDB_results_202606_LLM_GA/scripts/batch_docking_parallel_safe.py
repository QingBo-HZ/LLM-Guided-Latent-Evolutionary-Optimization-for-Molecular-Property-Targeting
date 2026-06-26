#!/usr/bin/env python3
"""Safe batch docking with Vina.

This is a corrected wrapper around the local docking pipeline:
- preserves ligand-to-score identity under multiprocessing
- records per-ligand failures instead of aborting the full batch
- reuses pre-generated PDBQT files when available
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import subprocess
import tempfile
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


WORKER_CFG = {}


def obabel_convert(in_path, out_path, in_fmt, out_fmt, opts=()):
    cmd = ["obabel", f"-i{in_fmt}", str(in_path), f"-o{out_fmt}"] + list(opts) + ["-O", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
        raise RuntimeError(result.stderr or result.stdout or f"obabel failed: {' '.join(cmd)}")


def smiles_to_pdbqt(smiles, out_pdbqt):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    rc = AllChem.EmbedMolecule(mol, params)
    if rc != 0:
        params.useRandomCoords = True
        rc = AllChem.EmbedMolecule(mol, params)
    if rc != 0:
        raise RuntimeError("RDKit EmbedMolecule failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".mol", delete=False) as tmp:
        tmp_mol = Path(tmp.name)
    try:
        Chem.MolToMolFile(mol, str(tmp_mol))
        obabel_convert(tmp_mol, out_pdbqt, "mol", "pdbqt", ["-h"])
    finally:
        tmp_mol.unlink(missing_ok=True)


def receptor_pdb_to_pdbqt(in_pdb, out_pdbqt):
    obabel_convert(in_pdb, out_pdbqt, "pdb", "pdbqt", ["-xr", "-h"])


def worker_init(rec_pdbqt, center, box_size, exhaustiveness, n_poses):
    WORKER_CFG["rec"] = str(rec_pdbqt)
    WORKER_CFG["center"] = list(center)
    WORKER_CFG["box_size"] = list(box_size)
    WORKER_CFG["exh"] = exhaustiveness
    WORKER_CFG["n_poses"] = n_poses


def dock_one(payload):
    idx, ligand, ligand_pdbqt = payload
    from vina import Vina

    try:
        v = Vina(sf_name="vina", verbosity=0)
        v.set_receptor(WORKER_CFG["rec"])
        v.set_ligand_from_file(str(ligand_pdbqt))
        v.compute_vina_maps(center=WORKER_CFG["center"], box_size=WORKER_CFG["box_size"])
        v.dock(exhaustiveness=WORKER_CFG["exh"], n_poses=WORKER_CFG["n_poses"])
        score = float(v.energies(WORKER_CFG["n_poses"])[0][0])
        return idx, ligand, score, "ok", ""
    except Exception as exc:
        return idx, ligand, None, "failed", str(exc).replace("\n", " ")[:500]


def read_ligands(smiles_csv):
    ligands = []
    with open(smiles_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            smi = row.get("smiles") or row.get("Smiles")
            if not smi:
                continue
            ligand_id = row.get("ID") or row.get("dock_id") or f"lig_{i:04d}"
            name = row.get("name") or ligand_id
            ligands.append({"ID": ligand_id, "name": name, "smiles": smi})
    return ligands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_csv", required=True)
    parser.add_argument("--receptor_pdb", required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--cz", type=float, required=True)
    parser.add_argument("--sx", type=float, required=True)
    parser.add_argument("--sy", type=float, required=True)
    parser.add_argument("--sz", type=float, required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--work_dir", default="/tmp/docking_batch_safe")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--n_poses", type=int, default=9)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    work = Path(args.work_dir)
    pdbqts_dir = work / "pdbqts"
    pdbqts_dir.mkdir(parents=True, exist_ok=True)
    rec_pdbqt = work / "receptor.pdbqt"
    if not rec_pdbqt.exists() or rec_pdbqt.stat().st_size < 100:
        print(f"[prep] receptor pdbqt: {args.receptor_pdb} -> {rec_pdbqt}", flush=True)
        receptor_pdb_to_pdbqt(Path(args.receptor_pdb), rec_pdbqt)
    else:
        print(f"[prep] receptor pdbqt exists: {rec_pdbqt}", flush=True)

    ligands = read_ligands(args.smiles_csv)
    print(f"[prep] preparing pdbqts for {len(ligands)} ligands...", flush=True)
    payloads = []
    prep_failures = []
    for idx, ligand in enumerate(ligands):
        pdbqt = pdbqts_dir / f"lig_{idx:04d}.pdbqt"
        try:
            if not pdbqt.exists() or pdbqt.stat().st_size < 100:
                smiles_to_pdbqt(ligand["smiles"], pdbqt)
            payloads.append((idx, ligand, pdbqt))
        except Exception as exc:
            prep_failures.append((idx, ligand, str(exc).replace("\n", " ")[:500]))
    print(f"[prep] {len(payloads)} pdbqts ready; prep_failures={len(prep_failures)}", flush=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    center = (args.cx, args.cy, args.cz)
    box_size = (args.sx, args.sy, args.sz)
    print(f"[dock] {len(payloads)} ligands with {args.workers} workers", flush=True)
    with mp.Pool(
        args.workers,
        initializer=worker_init,
        initargs=(rec_pdbqt, center, box_size, args.exhaustiveness, args.n_poses),
    ) as pool:
        for done, result in enumerate(pool.imap_unordered(dock_one, payloads), start=1):
            idx, ligand, score, status, error = result
            rows.append(
                {
                    "ID": ligand["ID"],
                    "name": ligand["name"],
                    "smiles": ligand["smiles"],
                    "vina_kcal_mol": "" if score is None else f"{score:.4f}",
                    "status": status,
                    "error": error,
                }
            )
            if done % 5 == 0 or done == len(payloads):
                ok = sum(1 for r in rows if r["status"] == "ok")
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-8)
                eta = (len(payloads) - done) / max(rate, 1e-8)
                print(f"  [dock {done}/{len(payloads)}] ok={ok} elapsed={elapsed:.1f}s ETA={eta:.0f}s", flush=True)

    for idx, ligand, error in prep_failures:
        rows.append(
            {
                "ID": ligand["ID"],
                "name": ligand["name"],
                "smiles": ligand["smiles"],
                "vina_kcal_mol": "",
                "status": "prep_failed",
                "error": error,
            }
        )

    order = {lig["ID"]: i for i, lig in enumerate(ligands)}
    rows.sort(key=lambda row: order.get(row["ID"], 10**9))
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "name", "smiles", "vina_kcal_mol", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    failed = len(rows) - ok
    elapsed = time.time() - t0
    print(f"[done] ok={ok} failed={failed} total={elapsed:.1f}s -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
