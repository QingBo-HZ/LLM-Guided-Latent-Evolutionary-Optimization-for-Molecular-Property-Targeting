#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gaussian HOMO-LUMO gap parser and Fig. 2f plotting pipeline.

Input:
    Gaussian .log / .out files, organized by method folders.

Outputs:
    1. dft_gap_all_candidates.csv
    2. dft_gap_ranked_all.csv
    3. dft_gap_best_per_method.csv
    4. fig2f_dft_gap_all_candidates.png/pdf
    5. fig2f_dft_gap_best_per_method.png/pdf
    6. fig2f_pred_vs_dft_gap.png/pdf, if predicted gap metadata is provided

Notes:
    Gaussian orbital eigenvalues are normally printed in Hartree.
    This script converts HOMO-LUMO gap from Hartree to eV:
        1 Hartree = 27.211386245988 eV
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HARTREE_TO_EV = 27.211386245988


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_gaussian_files(input_dir):
    gaussian_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".log", ".out", ".gjf.out")):
                gaussian_files.append(os.path.join(root, f))
    gaussian_files = sorted(gaussian_files)
    return gaussian_files


def parse_float_list(s):
    vals = []
    for x in re.findall(r"[-+]?\d+\.\d+(?:[DEde][-+]?\d+)?|[-+]?\d+(?:[DEde][-+]?\d+)", s):
        try:
            vals.append(float(x.replace("D", "E").replace("d", "e")))
        except Exception:
            pass
    return vals


def normalize_method_name(name):
    name = str(name)
    mapping = {
        "1_random_search": "Random Latent Search",
        "random_search": "Random Latent Search",
        "Random_Latent_Search": "Random Latent Search",
        "random": "Random Latent Search",

        "2_smiles_GA": "BRICS-based SMILES GA",
        "BRICS_SMILES_GA": "BRICS-based SMILES GA",
        "brics": "BRICS-based SMILES GA",
        "fragment_ga": "BRICS-based SMILES GA",

        "3_latent_GA_noLLM": "Latent GA",
        "Latent_GA": "Latent GA",
        "latent_ga": "Latent GA",

        "4_latent_GA_LLM": "LLM-Initialized Latent GA",
        "LLM_Initialized_Latent_GA": "LLM-Initialized Latent GA",
        "llm_init": "LLM-Initialized Latent GA",

        "5_Ours": "Iterative LLM-Guided Latent GA",
        "Iterative_LLM_Guided_Latent_GA": "Iterative LLM-Guided Latent GA",
        "ours": "Iterative LLM-Guided Latent GA",
    }

    if name in mapping:
        return mapping[name]

    lower = name.lower()
    if "random" in lower:
        return "Random Latent Search"
    if "brics" in lower or "fragment" in lower or "smiles" in lower:
        return "BRICS-based SMILES GA"
    if "llm" in lower and "guided" in lower:
        return "Iterative LLM-Guided Latent GA"
    if "ours" in lower or "iterative" in lower:
        return "Iterative LLM-Guided Latent GA"
    if "llm" in lower:
        return "LLM-Initialized Latent GA"
    if "latent" in lower:
        return "Latent GA"

    return name.replace("_", " ")


def infer_method_from_path(path, input_dir):
    rel = os.path.relpath(path, input_dir)
    parts = rel.split(os.sep)

    if len(parts) >= 2:
        raw_method = parts[0]
    else:
        raw_method = os.path.basename(os.path.dirname(path))

    return normalize_method_name(raw_method)


def infer_candidate_rank(path):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]

    patterns = [
        r"top[_-]?(\d+)",
        r"rank[_-]?(\d+)",
        r"candidate[_-]?(\d+)",
        r"cand[_-]?(\d+)",
        r"r[_-]?(\d+)",
        r"_(\d+)$",
    ]

    for p in patterns:
        m = re.search(p, name, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

    return None


# ============================================================
# Gaussian parser
# ============================================================

def finalize_orbital_block(blocks, spin, occ, virt):
    if len(occ) > 0 and len(virt) > 0:
        blocks.append({
            "spin": spin,
            "occ": occ.copy(),
            "virt": virt.copy(),
            "homo": float(max(occ)),
            "lumo": float(min(virt)),
            "gap_ha": float(min(virt) - max(occ)),
            "n_occ": int(len(occ)),
            "n_virt": int(len(virt)),
        })


def parse_orbital_blocks(lines, spin="alpha"):
    """
    Parse final orbital eigenvalue block for alpha or beta spin.

    Gaussian examples:
        Alpha  occ. eigenvalues -- -0.350 -0.250 ...
        Alpha virt. eigenvalues --  0.020  0.040 ...
        Beta  occ. eigenvalues -- ...
        Beta virt. eigenvalues -- ...

    We collect blocks and use the last complete block.
    """
    if spin.lower() == "alpha":
        occ_pat = re.compile(r"Alpha\s+occ\.\s+eigenvalues\s+--\s+(.*)", re.IGNORECASE)
        virt_pat = re.compile(r"Alpha\s+virt\.\s+eigenvalues\s+--\s+(.*)", re.IGNORECASE)
    elif spin.lower() == "beta":
        occ_pat = re.compile(r"Beta\s+occ\.\s+eigenvalues\s+--\s+(.*)", re.IGNORECASE)
        virt_pat = re.compile(r"Beta\s+virt\.\s+eigenvalues\s+--\s+(.*)", re.IGNORECASE)
    else:
        raise ValueError("spin must be alpha or beta")

    blocks = []
    curr_occ = []
    curr_virt = []
    in_block = False
    have_virt = False

    for line in lines:
        mo = occ_pat.search(line)
        mv = virt_pat.search(line)

        if mo:
            # New occ after a previous virt means a new orbital print block starts.
            if in_block and have_virt:
                finalize_orbital_block(blocks, spin, curr_occ, curr_virt)
                curr_occ = []
                curr_virt = []
                have_virt = False

            in_block = True
            curr_occ.extend(parse_float_list(mo.group(1)))
            continue

        if mv and in_block:
            have_virt = True
            curr_virt.extend(parse_float_list(mv.group(1)))
            continue

    if in_block and have_virt:
        finalize_orbital_block(blocks, spin, curr_occ, curr_virt)

    return blocks


def parse_scf_energy(lines):
    """
    Parse final SCF energy from lines like:
        SCF Done:  E(RB3LYP) = -xxx A.U. after ...
    """
    energies = []
    pat = re.compile(r"SCF Done:\s+E\([RU]?[A-Za-z0-9]+\)\s+=\s+([-+]?\d+\.\d+)")
    for line in lines:
        m = pat.search(line)
        if m:
            try:
                energies.append(float(m.group(1)))
            except Exception:
                pass
    return energies[-1] if len(energies) > 0 else None


def parse_charge_multiplicity(lines):
    charge = None
    multiplicity = None
    pat = re.compile(r"Charge\s+=\s+(-?\d+)\s+Multiplicity\s+=\s+(\d+)")
    for line in lines:
        m = pat.search(line)
        if m:
            charge = int(m.group(1))
            multiplicity = int(m.group(2))
            break
    return charge, multiplicity


def parse_imaginary_frequencies(lines):
    """
    Count negative frequencies if frequency calculation exists.
    """
    freqs = []
    pat = re.compile(r"Frequencies\s+--\s+(.*)")
    for line in lines:
        m = pat.search(line)
        if m:
            freqs.extend(parse_float_list(m.group(1)))
    n_imag = int(sum(1 for x in freqs if x < 0.0))
    return n_imag, freqs


def parse_gaussian_gap(log_path):
    with open(log_path, "r", errors="ignore") as f:
        lines = f.readlines()

    text = "".join(lines)

    normal_termination = "Normal termination of Gaussian" in text
    error_termination = "Error termination" in text

    scf_energy = parse_scf_energy(lines)
    charge, multiplicity = parse_charge_multiplicity(lines)
    n_imag, freqs = parse_imaginary_frequencies(lines)

    alpha_blocks = parse_orbital_blocks(lines, spin="alpha")
    beta_blocks = parse_orbital_blocks(lines, spin="beta")

    alpha = alpha_blocks[-1] if len(alpha_blocks) > 0 else None
    beta = beta_blocks[-1] if len(beta_blocks) > 0 else None

    if alpha is None and beta is None:
        return {
            "parse_ok": False,
            "error": "No alpha/beta orbital eigenvalue block found",
            "normal_termination": normal_termination,
            "error_termination": error_termination,
            "scf_energy_ha": scf_energy,
            "charge": charge,
            "multiplicity": multiplicity,
            "n_imag_freq": n_imag,
        }

    # Closed-shell or restricted case: alpha block exists and beta often absent.
    if alpha is not None and beta is None:
        homo_ha = alpha["homo"]
        lumo_ha = alpha["lumo"]
        gap_ha = lumo_ha - homo_ha

        return {
            "parse_ok": True,
            "spin_type": "restricted_or_alpha_only",
            "homo_ha": float(homo_ha),
            "lumo_ha": float(lumo_ha),
            "gap_ha": float(gap_ha),
            "gap_ev": float(gap_ha * HARTREE_TO_EV),
            "alpha_homo_ha": float(alpha["homo"]),
            "alpha_lumo_ha": float(alpha["lumo"]),
            "alpha_gap_ha": float(alpha["gap_ha"]),
            "beta_homo_ha": None,
            "beta_lumo_ha": None,
            "beta_gap_ha": None,
            "normal_termination": normal_termination,
            "error_termination": error_termination,
            "scf_energy_ha": scf_energy,
            "charge": charge,
            "multiplicity": multiplicity,
            "n_imag_freq": n_imag,
        }

    # Unrestricted case: use global spin orbital gap.
    if alpha is not None and beta is not None:
        homo_ha = max(alpha["homo"], beta["homo"])
        lumo_ha = min(alpha["lumo"], beta["lumo"])
        gap_ha = lumo_ha - homo_ha

        return {
            "parse_ok": True,
            "spin_type": "unrestricted_alpha_beta",
            "homo_ha": float(homo_ha),
            "lumo_ha": float(lumo_ha),
            "gap_ha": float(gap_ha),
            "gap_ev": float(gap_ha * HARTREE_TO_EV),
            "alpha_homo_ha": float(alpha["homo"]),
            "alpha_lumo_ha": float(alpha["lumo"]),
            "alpha_gap_ha": float(alpha["gap_ha"]),
            "beta_homo_ha": float(beta["homo"]),
            "beta_lumo_ha": float(beta["lumo"]),
            "beta_gap_ha": float(beta["gap_ha"]),
            "normal_termination": normal_termination,
            "error_termination": error_termination,
            "scf_energy_ha": scf_energy,
            "charge": charge,
            "multiplicity": multiplicity,
            "n_imag_freq": n_imag,
        }

    # Rare case: beta only.
    homo_ha = beta["homo"]
    lumo_ha = beta["lumo"]
    gap_ha = lumo_ha - homo_ha

    return {
        "parse_ok": True,
        "spin_type": "beta_only",
        "homo_ha": float(homo_ha),
        "lumo_ha": float(lumo_ha),
        "gap_ha": float(gap_ha),
        "gap_ev": float(gap_ha * HARTREE_TO_EV),
        "alpha_homo_ha": None,
        "alpha_lumo_ha": None,
        "alpha_gap_ha": None,
        "beta_homo_ha": float(beta["homo"]),
        "beta_lumo_ha": float(beta["lumo"]),
        "beta_gap_ha": float(beta["gap_ha"]),
        "normal_termination": normal_termination,
        "error_termination": error_termination,
        "scf_energy_ha": scf_energy,
        "charge": charge,
        "multiplicity": multiplicity,
        "n_imag_freq": n_imag,
    }


# ============================================================
# Optional metadata merge
# ============================================================

def load_optional_metadata(meta_csv):
    if meta_csv is None:
        return None

    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"metadata CSV not found: {meta_csv}")

    meta = pd.read_csv(meta_csv)

    # Standardize possible columns.
    rename = {}
    for c in meta.columns:
        lc = c.lower()
        if lc in ["file", "filename", "log_file", "gaussian_file"]:
            rename[c] = "filename"
        elif lc in ["method_name", "method"]:
            rename[c] = "method"
        elif lc in ["rank", "candidate_rank", "pred_rank", "rank_by_score"]:
            rename[c] = "candidate_rank"
        elif lc in ["smiles", "canonical_smiles"]:
            # keep smiles name if first found
            if "smiles" not in meta.columns:
                rename[c] = "smiles"
        elif lc in ["pred_gap", "predicted_gap", "gap_pred", "pred_gap_ha"]:
            rename[c] = "pred_gap"
        elif lc in ["pred_gap_ev", "predicted_gap_ev"]:
            rename[c] = "pred_gap_ev"

    meta = meta.rename(columns=rename)
    return meta


def merge_metadata(df, meta):
    if meta is None:
        return df

    out = df.copy()

    # Merge by filename if possible.
    if "filename" in meta.columns:
        meta2 = meta.copy()
        meta2["filename"] = meta2["filename"].astype(str).apply(os.path.basename)
        out["filename"] = out["file_path"].apply(os.path.basename)
        out = out.merge(meta2, on="filename", how="left", suffixes=("", "_meta"))
        return out

    # Merge by method + candidate rank if possible.
    if "method" in meta.columns and "candidate_rank" in meta.columns:
        meta2 = meta.copy()
        meta2["method"] = meta2["method"].apply(normalize_method_name)
        out = out.merge(
            meta2,
            on=["method", "candidate_rank"],
            how="left",
            suffixes=("", "_meta")
        )
        return out

    return out


# ============================================================
# Plotting
# ============================================================

def method_order():
    return [
        "Random Latent Search",
        "BRICS-based SMILES GA",
        "Latent GA",
        "LLM-Initialized Latent GA",
        "Iterative LLM-Guided Latent GA",
    ]


def sort_methods(df):
    order = method_order()
    order_map = {m: i for i, m in enumerate(order)}
    df = df.copy()
    df["_method_order"] = df["method"].map(order_map).fillna(999).astype(int)
    df = df.sort_values(["_method_order", "candidate_rank", "gap_ev"]).reset_index(drop=True)
    df = df.drop(columns=["_method_order"])
    return df


def add_bar_labels(ax, fmt="{:.2f}", fontsize=8):
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height):
            ax.annotate(
                fmt.format(height),
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                rotation=0,
                xytext=(0, 2),
                textcoords="offset points",
            )


def plot_all_candidates(df, out_dir, unit="eV"):
    valid = df[df["parse_ok"] == True].copy()
    valid = sort_methods(valid)

    if len(valid) == 0:
        print("[WARN] No valid Gaussian results for all-candidate plot.")
        return

    labels = []
    for _, r in valid.iterrows():
        rank = r["candidate_rank"]
        if pd.isna(rank):
            rank_label = ""
        else:
            rank_label = f"-{int(rank)}"
        labels.append(f"{r['method_short']}{rank_label}")

    y_col = "gap_ev" if unit.lower() == "ev" else "gap_ha"
    y_label = "DFT HOMO-LUMO gap (eV)" if unit.lower() == "ev" else "DFT HOMO-LUMO gap (Hartree)"

    fig_w = max(9, 0.45 * len(valid))
    plt.figure(figsize=(fig_w, 5.2))
    ax = plt.gca()

    ax.bar(range(len(valid)), valid[y_col].values)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Candidate")
    ax.set_title("DFT-calculated HOMO-LUMO gaps of selected candidates")
    ax.grid(axis="y", alpha=0.25)

    add_bar_labels(ax, fmt="{:.2f}" if unit.lower() == "ev" else "{:.4f}", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2f_dft_gap_all_candidates.png"), dpi=600)
    plt.savefig(os.path.join(out_dir, "fig2f_dft_gap_all_candidates.pdf"))
    plt.close()


def plot_best_per_method(best_df, out_dir, unit="eV"):
    valid = best_df[best_df["parse_ok"] == True].copy()
    valid = sort_methods(valid)

    if len(valid) == 0:
        print("[WARN] No valid Gaussian results for best-per-method plot.")
        return

    y_col = "gap_ev" if unit.lower() == "ev" else "gap_ha"
    y_label = "DFT HOMO-LUMO gap (eV)" if unit.lower() == "ev" else "DFT HOMO-LUMO gap (Hartree)"

    labels = valid["method_short"].tolist()

    plt.figure(figsize=(7.8, 5.2))
    ax = plt.gca()

    ax.bar(range(len(valid)), valid[y_col].values)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Optimization method")
    ax.set_title("Best DFT-validated molecule from each method")
    ax.grid(axis="y", alpha=0.25)

    add_bar_labels(ax, fmt="{:.2f}" if unit.lower() == "ev" else "{:.4f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2f_dft_gap_best_per_method.png"), dpi=600)
    plt.savefig(os.path.join(out_dir, "fig2f_dft_gap_best_per_method.pdf"))
    plt.close()


def plot_pred_vs_dft(df, out_dir):
    valid = df[df["parse_ok"] == True].copy()

    pred_col = None
    if "pred_gap_ev" in valid.columns:
        pred_col = "pred_gap_ev"
    elif "pred_gap" in valid.columns:
        # If pred_gap exists but not pred_gap_ev, keep as-is.
        # You can manually ensure units match before using this plot.
        pred_col = "pred_gap"

    if pred_col is None:
        print("[INFO] No predicted gap column found; skip pred-vs-DFT scatter.")
        return

    valid[pred_col] = pd.to_numeric(valid[pred_col], errors="coerce")
    valid = valid.dropna(subset=[pred_col, "gap_ev"])

    if len(valid) < 2:
        print("[INFO] Too few points for pred-vs-DFT scatter.")
        return

    x = valid[pred_col].values.astype(float)
    y = valid["gap_ev"].values.astype(float)

    plt.figure(figsize=(5.6, 5.2))
    ax = plt.gca()

    ax.scatter(x, y, s=45, alpha=0.8)

    mn = min(np.min(x), np.min(y))
    mx = max(np.max(x), np.max(y))
    pad = (mx - mn) * 0.08 if mx > mn else 1.0

    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "--", linewidth=1.1)
    ax.set_xlabel("Predicted gap")
    ax.set_ylabel("DFT HOMO-LUMO gap (eV)")
    ax.set_title("Predicted gap versus DFT-calculated gap")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2f_pred_vs_dft_gap.png"), dpi=600)
    plt.savefig(os.path.join(out_dir, "fig2f_pred_vs_dft_gap.pdf"))
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Parse Gaussian HOMO-LUMO gap and generate Fig. 2f")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing Gaussian .log/.out files."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="Optional metadata CSV with filename or method+candidate_rank and predicted gap/smiles."
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="eV",
        choices=["eV", "Hartree"],
        help="Plot unit."
    )
    parser.add_argument(
        "--require_normal_termination",
        action="store_true",
        help="Only treat normally terminated Gaussian jobs as valid."
    )

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    files = find_gaussian_files(args.input_dir)

    if len(files) == 0:
        raise RuntimeError(f"No Gaussian .log/.out files found in {args.input_dir}")

    print(f"[INFO] Found Gaussian files: {len(files)}")

    rows = []

    method_short_map = {
        "Random Latent Search": "Random",
        "BRICS-based SMILES GA": "BRICS-GA",
        "Latent GA": "Latent-GA",
        "LLM-Initialized Latent GA": "LLM-init",
        "Iterative LLM-Guided Latent GA": "Ours",
    }

    for fp in files:
        method = infer_method_from_path(fp, args.input_dir)
        candidate_rank = infer_candidate_rank(fp)

        print(f"[INFO] Parsing: {fp}")

        result = parse_gaussian_gap(fp)

        parse_ok = bool(result.get("parse_ok", False))

        if args.require_normal_termination and not bool(result.get("normal_termination", False)):
            parse_ok = False
            result["parse_ok"] = False
            result["error"] = "Not normal termination"

        row = {
            "file_path": fp,
            "file_name": os.path.basename(fp),
            "method": method,
            "method_short": method_short_map.get(method, method),
            "candidate_rank": candidate_rank,
        }
        row.update(result)
        row["parse_ok"] = parse_ok

        rows.append(row)

    df = pd.DataFrame(rows)

    meta = load_optional_metadata(args.metadata_csv)
    df = merge_metadata(df, meta)

    # Convert numeric columns.
    for c in [
        "gap_ha", "gap_ev", "homo_ha", "lumo_ha",
        "alpha_homo_ha", "alpha_lumo_ha", "alpha_gap_ha",
        "beta_homo_ha", "beta_lumo_ha", "beta_gap_ha",
        "scf_energy_ha", "candidate_rank"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    all_csv = os.path.join(args.out_dir, "dft_gap_all_candidates.csv")
    df.to_csv(all_csv, index=False)

    valid = df[df["parse_ok"] == True].copy()
    valid = valid.dropna(subset=["gap_ev"])

    ranked = valid.sort_values("gap_ev", ascending=True).reset_index(drop=True)
    ranked["global_dft_rank"] = np.arange(1, len(ranked) + 1)

    ranked_csv = os.path.join(args.out_dir, "dft_gap_ranked_all.csv")
    ranked.to_csv(ranked_csv, index=False)

    if len(valid) > 0:
        best = (
            valid.sort_values("gap_ev", ascending=True)
            .groupby("method", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        best = sort_methods(best)
        best["method_dft_rank"] = np.arange(1, len(best) + 1)
    else:
        best = pd.DataFrame()

    best_csv = os.path.join(args.out_dir, "dft_gap_best_per_method.csv")
    best.to_csv(best_csv, index=False)

    # Summary.
    summary = {
        "input_dir": args.input_dir,
        "out_dir": args.out_dir,
        "n_files_found": int(len(files)),
        "n_parse_ok": int(df["parse_ok"].sum()) if "parse_ok" in df.columns else 0,
        "n_valid_gap": int(len(valid)),
        "all_candidates_csv": all_csv,
        "ranked_all_csv": ranked_csv,
        "best_per_method_csv": best_csv,
        "unit": args.unit,
        "require_normal_termination": bool(args.require_normal_termination),
    }

    if len(ranked) > 0:
        summary["global_best"] = {
            "method": ranked.iloc[0]["method"],
            "file_name": ranked.iloc[0]["file_name"],
            "gap_ev": float(ranked.iloc[0]["gap_ev"]),
            "gap_ha": float(ranked.iloc[0]["gap_ha"]),
        }

    summary_path = os.path.join(args.out_dir, "dft_gap_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Plots.
    plot_all_candidates(df, args.out_dir, unit=args.unit)
    plot_best_per_method(best, args.out_dir, unit=args.unit)
    plot_pred_vs_dft(df, args.out_dir)

    print("\n========== DONE ==========")
    print(f"[INFO] All candidates CSV: {all_csv}")
    print(f"[INFO] Ranked all CSV:     {ranked_csv}")
    print(f"[INFO] Best per method:    {best_csv}")
    print(f"[INFO] Summary:            {summary_path}")

    if len(ranked) > 0:
        print("\n[GLOBAL DFT RANKING]")
        cols = ["global_dft_rank", "method", "file_name", "gap_ev", "gap_ha", "normal_termination"]
        print(ranked[cols].head(20).to_string(index=False))

    if len(best) > 0:
        print("\n[BEST PER METHOD]")
        cols = ["method", "file_name", "gap_ev", "gap_ha", "normal_termination"]
        print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()