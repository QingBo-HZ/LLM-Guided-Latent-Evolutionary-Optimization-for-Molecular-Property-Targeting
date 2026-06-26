#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import glob
import argparse
import traceback
import numpy as np
import torch
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# =========================
# 你的 PS-VAE 根目录
# =========================
PSVAE_ROOT = "/root/autodl-tmp/sweeteners_evolve/QM9_test/PS-VAE"
sys.path.append(os.path.join(PSVAE_ROOT, "src"))

print("[DEBUG] sys.path appended", flush=True)
print("[DEBUG] importing PSVAEModel...", flush=True)
from pl_models import PSVAEModel

print("[DEBUG] importing chem utils...", flush=True)
from utils.chem_utils import smiles2molecule, GeneralVocab
from data.mol_bpe import Tokenizer
from rdkit.Chem.rdchem import BondType
import torch.serialization

SAFE_GLOBALS = [Tokenizer, GeneralVocab, BondType]
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(SAFE_GLOBALS)
print(f"[DEBUG] registered safe globals: {[str(x) for x in SAFE_GLOBALS]}", flush=True)


# ============================================================
# 0. RDKit / SMILES 工具函数
# ============================================================
def canonicalize_smiles(smi: str):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def rdkit_basic_check(smi: str):
    """
    只用于记录诊断信息，不用于替代 PS-VAE 的 molecule 构建。
    """
    out = {
        "rdkit_parse_ok": 0,
        "rdkit_sanitize_ok": 0,
        "canonical_smiles": None,
        "heavy_atoms": None,
        "atom_symbols": None,
        "formal_charge": None,
        "num_rings": None,
        "num_aromatic_rings": None,
    }

    try:
        mol = Chem.MolFromSmiles(str(smi), sanitize=False)
        if mol is None:
            return out

        out["rdkit_parse_ok"] = 1

        try:
            Chem.SanitizeMol(mol)
            out["rdkit_sanitize_ok"] = 1
        except Exception:
            out["rdkit_sanitize_ok"] = 0

        try:
            out["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            out["canonical_smiles"] = None

        try:
            out["heavy_atoms"] = int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1))
            out["atom_symbols"] = ",".join(sorted(set(a.GetSymbol() for a in mol.GetAtoms())))
            out["formal_charge"] = int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
            out["num_rings"] = int(mol.GetRingInfo().NumRings())
            out["num_aromatic_rings"] = int(
                sum(1 for ring in mol.GetRingInfo().AtomRings()
                    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
            )
        except Exception:
            pass

    except Exception:
        pass

    return out


# ============================================================
# 1. 加载模型
# ============================================================
def load_model(ckpt_path: str, gpu: int):
    print(f"[DEBUG] load_model ckpt={ckpt_path}", flush=True)

    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    print(f"[DEBUG] using device={device}", flush=True)
    print("[DEBUG] loading checkpoint...", flush=True)

    # 保持你的原始加载方式，避免破坏 checkpoint 兼容性
    model = PSVAEModel.load_from_checkpoint(ckpt_path)

    print("[DEBUG] checkpoint loaded", flush=True)

    model.to(device)
    model.eval()
    print("[DEBUG] model moved to device and set eval", flush=True)

    # 尝试打印模型内部 tokenizer 信息，失败也不影响运行
    try:
        attrs = dir(model)
        possible = [a for a in attrs if "token" in a.lower() or "vocab" in a.lower()]
        print(f"[DEBUG] model tokenizer/vocab related attrs: {possible}", flush=True)
    except Exception:
        pass

    return model, device


# ============================================================
# 2. 自动选择输入文件
#    优先级：.smi > .jsonl > .csv
# ============================================================
def resolve_input_file(input_path: str) -> str:
    if os.path.isfile(input_path):
        return input_path

    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    smi_files = sorted(glob.glob(os.path.join(input_path, "*.smi")), key=os.path.getmtime, reverse=True)
    if smi_files:
        print(f"[DEBUG] auto-selected .smi: {smi_files[0]}", flush=True)
        return smi_files[0]

    jsonl_files = sorted(glob.glob(os.path.join(input_path, "*.jsonl")), key=os.path.getmtime, reverse=True)
    if jsonl_files:
        print(f"[DEBUG] auto-selected .jsonl: {jsonl_files[0]}", flush=True)
        return jsonl_files[0]

    csv_files = sorted(glob.glob(os.path.join(input_path, "*.csv")), key=os.path.getmtime, reverse=True)
    if csv_files:
        print(f"[DEBUG] auto-selected .csv: {csv_files[0]}", flush=True)
        return csv_files[0]

    raise FileNotFoundError(
        f"No supported input files found in directory: {input_path}. "
        f"Supported: .smi, .jsonl, .csv"
    )


# ============================================================
# 3. 读取 smiles
# ============================================================
def read_smiles_from_smi(path: str):
    smiles_list = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            s = line.strip()
            if s:
                smiles_list.append({
                    "idx": idx,
                    "smiles": s,
                    "source_file": path,
                    "source_type": "smi",
                })
    return smiles_list


def read_smiles_from_jsonl(path: str, only_accepted: bool = True):
    smiles_list = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            if only_accepted:
                status = obj.get("status", None)
                smi = obj.get("smiles", None)
                if status == "accepted" and isinstance(smi, str) and smi.strip():
                    smiles_list.append({
                        "idx": idx,
                        "smiles": smi.strip(),
                        "source_file": path,
                        "source_type": "jsonl_accepted",
                    })
            else:
                smi = obj.get("smiles", None)
                if isinstance(smi, str) and smi.strip():
                    smiles_list.append({
                        "idx": idx,
                        "smiles": smi.strip(),
                        "source_file": path,
                        "source_type": "jsonl_any",
                    })
                elif isinstance(smi, list):
                    for j, s in enumerate(smi):
                        if isinstance(s, str) and s.strip():
                            smiles_list.append({
                                "idx": f"{idx}_{j}",
                                "smiles": s.strip(),
                                "source_file": path,
                                "source_type": "jsonl_smiles_list",
                            })
    return smiles_list


def read_smiles_from_csv(path: str):
    smiles_list = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "smiles" not in reader.fieldnames:
            raise ValueError(f"CSV must contain 'smiles' column. Found: {reader.fieldnames}")

        for idx, row in enumerate(reader):
            s = row["smiles"].strip()
            if s:
                smiles_list.append({
                    "idx": idx,
                    "smiles": s,
                    "source_file": path,
                    "source_type": "csv",
                })
    return smiles_list


def read_smiles(input_path: str, only_accepted_jsonl: bool = True):
    path = resolve_input_file(input_path)

    if path.endswith(".smi"):
        smiles_list = read_smiles_from_smi(path)
    elif path.endswith(".jsonl"):
        smiles_list = read_smiles_from_jsonl(path, only_accepted=only_accepted_jsonl)
    elif path.endswith(".csv"):
        smiles_list = read_smiles_from_csv(path)
    else:
        raise ValueError(f"Unsupported input file: {path}")

    print(f"[DEBUG] loaded {len(smiles_list)} raw smiles from {path}", flush=True)
    return smiles_list, path


# ============================================================
# 4. 去重
# ============================================================
def deduplicate_smiles(smiles_records, canonical_dedup: bool = True):
    """
    canonical_dedup=True:
      用 RDKit canonical SMILES 去重，更适合 LLM 输出。
    canonical_dedup=False:
      保持原脚本逻辑，按原始文本去重。
    """
    seen = set()
    unique_records = []
    dup_count = 0
    canonical_fail_count = 0

    for rec in smiles_records:
        raw_smi = rec["smiles"]

        if canonical_dedup:
            can = canonicalize_smiles(raw_smi)
            if can is None:
                canonical_fail_count += 1
                key = raw_smi
                rec = dict(rec)
                rec["canonical_smiles"] = None
            else:
                key = can
                rec = dict(rec)
                rec["canonical_smiles"] = can
        else:
            key = raw_smi
            rec = dict(rec)
            rec["canonical_smiles"] = None

        if key in seen:
            dup_count += 1
            continue

        seen.add(key)
        unique_records.append(rec)

    print(
        f"[DEBUG] deduplicated smiles: {len(smiles_records)} -> {len(unique_records)}, "
        f"duplicates={dup_count}, canonical_fail={canonical_fail_count}",
        flush=True
    )
    return unique_records


# ============================================================
# 5. 提取 latent
# ============================================================
def extract_latent(smiles_records, model, kekulize: bool = True):
    x_list = []
    meta_rows = []
    fail_rows = []

    count_total = 0
    count_valid = 0
    count_fail_mol = 0
    count_fail_z = 0

    for rec in tqdm(smiles_records, desc="Extracting latent from LLM smiles"):
        count_total += 1
        smi = rec["smiles"]
        can_smi = rec.get("canonical_smiles", None)

        # 记录 RDKit 诊断信息
        rdkit_info = rdkit_basic_check(smi)

        try:
            mol = smiles2molecule(smi, kekulize=kekulize)
        except Exception as e:
            mol = None
            count_fail_mol += 1
            fail_rows.append({
                "row_idx": rec["idx"],
                "smiles": smi,
                "canonical_smiles": can_smi,
                "stage": "smiles2molecule_exception",
                "error_type": type(e).__name__,
                "error": repr(e),
                "traceback": traceback.format_exc(limit=3),
                "source_file": rec["source_file"],
                "source_type": rec["source_type"],
                **rdkit_info,
            })
            if count_fail_mol <= 5:
                print(f"[WARN] smiles2molecule exception: {smi}, err={e}", flush=True)
            continue

        if mol is None:
            count_fail_mol += 1
            fail_rows.append({
                "row_idx": rec["idx"],
                "smiles": smi,
                "canonical_smiles": can_smi,
                "stage": "smiles2molecule_none",
                "error_type": "MolNone",
                "error": "smiles2molecule returned None",
                "traceback": "",
                "source_file": rec["source_file"],
                "source_type": rec["source_type"],
                **rdkit_info,
            })
            if count_fail_mol <= 5:
                print(f"[WARN] smiles2molecule failed: {smi}", flush=True)
            continue

        try:
            with torch.no_grad():
                z = model.get_z_from_mol(mol).detach().cpu().numpy()

            # 统一成 [latent_dim]，避免某些模型返回 [1, D] 导致 stack 成 [N, 1, D]
            z = np.asarray(z, dtype=np.float32)
            if z.ndim > 1:
                z = np.squeeze(z)
            if z.ndim != 1:
                raise ValueError(f"Unexpected latent shape after squeeze: {z.shape}")

        except Exception as e:
            count_fail_z += 1
            fail_rows.append({
                "row_idx": rec["idx"],
                "smiles": smi,
                "canonical_smiles": can_smi,
                "stage": "get_z_from_mol",
                "error_type": type(e).__name__,
                "error": repr(e),
                "traceback": traceback.format_exc(limit=3),
                "source_file": rec["source_file"],
                "source_type": rec["source_type"],
                **rdkit_info,
            })
            if count_fail_z <= 5:
                print(f"[WARN] get_z_from_mol failed for smiles={smi}, err={e}", flush=True)
            continue

        x_list.append(z)
        count_valid += 1

        meta_rows.append({
            "row_idx": rec["idx"],
            "smiles": smi,
            "canonical_smiles": can_smi if can_smi is not None else rdkit_info.get("canonical_smiles"),
            "source_file": rec["source_file"],
            "source_type": rec["source_type"],
            "latent_dim": int(z.shape[0]),
            **rdkit_info,
        })

        if count_total % 100 == 0:
            print(
                f"[DEBUG] processed={count_total}, valid={count_valid}, "
                f"fail_mol={count_fail_mol}, fail_z={count_fail_z}",
                flush=True
            )

    print("[DEBUG] finished latent extraction", flush=True)
    print(f"[DEBUG] total={count_total}, valid={count_valid}, fail_mol={count_fail_mol}, fail_z={count_fail_z}", flush=True)

    if len(x_list) == 0:
        raise RuntimeError("No valid molecules extracted from input.")

    X = np.stack(x_list, axis=0).astype(np.float32)

    summary = {
        "total": int(count_total),
        "valid": int(count_valid),
        "fail_mol": int(count_fail_mol),
        "fail_z": int(count_fail_z),
        "encoding_success_rate": float(count_valid / count_total) if count_total > 0 else 0.0,
        "latent_shape": list(X.shape),
        "kekulize": bool(kekulize),
    }

    return X, meta_rows, fail_rows, summary


# ============================================================
# 6. 保存输出
# ============================================================
def save_outputs(X, meta_rows, fail_rows, summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    x_path = os.path.join(out_dir, "llm_init_latent.npy")
    smiles_path = os.path.join(out_dir, "llm_valid_smiles.smi")
    meta_path = os.path.join(out_dir, "llm_meta.csv")
    failed_path = os.path.join(out_dir, "llm_failed_smiles.csv")
    summary_path = os.path.join(out_dir, "llm_extract_summary.json")

    np.save(x_path, X)

    with open(smiles_path, "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(row["smiles"] + "\n")

    meta_fields = [
        "row_idx",
        "smiles",
        "canonical_smiles",
        "source_file",
        "source_type",
        "latent_dim",
        "rdkit_parse_ok",
        "rdkit_sanitize_ok",
        "heavy_atoms",
        "atom_symbols",
        "formal_charge",
        "num_rings",
        "num_aromatic_rings",
    ]

    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(meta_rows)

    fail_fields = [
        "row_idx",
        "smiles",
        "canonical_smiles",
        "stage",
        "error_type",
        "error",
        "traceback",
        "source_file",
        "source_type",
        "rdkit_parse_ok",
        "rdkit_sanitize_ok",
        "heavy_atoms",
        "atom_symbols",
        "formal_charge",
        "num_rings",
        "num_aromatic_rings",
    ]

    with open(failed_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fail_rows)

    summary = dict(summary)
    summary.update({
        "latent_path": x_path,
        "valid_smiles_path": smiles_path,
        "meta_path": meta_path,
        "failed_path": failed_path,
    })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DEBUG] Saved latent: {x_path}, shape={X.shape}", flush=True)
    print(f"[DEBUG] Saved smiles: {smiles_path}", flush=True)
    print(f"[DEBUG] Saved meta: {meta_path}", flush=True)
    print(f"[DEBUG] Saved failed smiles: {failed_path}, n={len(fail_rows)}", flush=True)
    print(f"[DEBUG] Saved summary: {summary_path}", flush=True)

    return x_path, smiles_path, meta_path, failed_path, summary_path


# ============================================================
# 7. 主函数
# ============================================================
def main(args):
    print("[DEBUG] entering main()", flush=True)
    print(f"[DEBUG] args = {args}", flush=True)

    model, device = load_model(args.ckpt, args.gpu)

    smiles_records, selected_input = read_smiles(
        input_path=args.input_path,
        only_accepted_jsonl=args.only_accepted_jsonl,
    )

    smiles_records = deduplicate_smiles(
        smiles_records,
        canonical_dedup=not args.no_canonical_dedup,
    )

    X, meta_rows, fail_rows, summary = extract_latent(
        smiles_records,
        model,
        kekulize=not args.no_kekulize,
    )

    summary["selected_input_file"] = selected_input
    summary["dedup_mode"] = "raw_text" if args.no_canonical_dedup else "canonical_smiles"

    save_outputs(X, meta_rows, fail_rows, summary, args.out_dir)

    print("[DEBUG] ALL DONE", flush=True)
    print(f"[DEBUG] selected input file: {selected_input}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained PS-VAE checkpoint"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/smiles_generated",
        help="Input file or directory. Supports .smi, .jsonl, .csv. "
             "If a directory is given, the newest .smi is preferred."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/root/autodl-tmp/sweeteners_evolve/Ablation_1/llm/latent",
        help="Directory to save llm_init_latent.npy and metadata"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id, use -1 for CPU"
    )
    parser.add_argument(
        "--only_accepted_jsonl",
        action="store_true",
        help="When reading .jsonl, only keep records with status == accepted"
    )
    parser.add_argument(
        "--no_canonical_dedup",
        action="store_true",
        help="Use raw SMILES text for deduplication instead of RDKit canonical SMILES."
    )
    parser.add_argument(
        "--no_kekulize",
        action="store_true",
        help="Call smiles2molecule(..., kekulize=False). Default is kekulize=True."
    )

    args = parser.parse_args()
    main(args)