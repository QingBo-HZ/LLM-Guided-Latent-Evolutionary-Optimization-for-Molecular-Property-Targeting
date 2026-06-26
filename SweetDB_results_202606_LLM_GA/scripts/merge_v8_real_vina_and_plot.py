#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


METHOD_ORDER = ["A", "B", "C", "D"]
METHOD_LABELS = {
    "A": "Random-Seeded Latent GA",
    "B": "SweetDB-Seeded Latent GA",
    "C": "LLM-Initialized Latent GA",
    "D": "Iterative LLM-Guided Latent GA",
}
COLORS = {
    "A": "#6F768A",
    "B": "#4F7EC7",
    "C": "#E8753A",
    "D": "#2F9B63",
}
MARKERS = {"A": "s", "B": "o", "C": "^", "D": "D"}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}


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


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Arial", "Segoe UI", "sans-serif"],
            "font.size": 18,
            "axes.titlesize": 25,
            "axes.labelsize": 21,
            "axes.linewidth": 1.5,
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 14.5,
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "savefig.facecolor": TOKENS["surface"],
            "savefig.dpi": 600,
        }
    )


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.07, 0.965, title, ha="left", va="top", fontsize=25, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.07, 0.915, subtitle, ha="left", va="top", fontsize=15.5, color=TOKENS["muted"])


def polish(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=1.15, alpha=0.9)
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=0.9, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def normalize_vina_csv(vina_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(vina_csv)
    score_candidates = [
        "vina_kcal_mol",
        "vina_affinity_kcal_mol",
        "affinity",
        "score",
        "vina",
        "real_vina",
        "real_vina_kcal_mol",
    ]
    score_col = next((c for c in score_candidates if c in df.columns), None)
    if score_col is None:
        raise ValueError(f"No Vina score column found. Accepted: {score_candidates}. Got: {list(df.columns)}")
    df = df.rename(columns={score_col: "vina_kcal_mol"})
    df["vina_kcal_mol"] = pd.to_numeric(df["vina_kcal_mol"], errors="coerce")
    if "dock_id" not in df.columns:
        for c in ["Dock_ID", "docking_id", "mol_id", "ligand_id"]:
            if c in df.columns:
                df = df.rename(columns={c: "dock_id"})
                break
    if "ID" not in df.columns:
        for c in ["id", "Name", "name"]:
            if c in df.columns:
                df = df.rename(columns={c: "ID"})
                break
    if "smiles" not in df.columns:
        for c in ["SMILES", "canonical_smiles", "ligand_smiles"]:
            if c in df.columns:
                df = df.rename(columns={c: "smiles"})
                break
    return df


def load_vina_by_original_id(vina: pd.DataFrame, submission_dir: Path) -> pd.DataFrame:
    frames = []
    if "ID" in vina.columns:
        direct = vina[["ID", "vina_kcal_mol"]].copy()
        frames.append(direct)

    if "dock_id" in vina.columns:
        for stage in ["all", "final", "generation"]:
            map_path = submission_dir / f"record_to_docking_id_{stage}.csv"
            if map_path.exists():
                mapping = pd.read_csv(map_path)
                merged = mapping[["ID", "dock_id"]].merge(vina[["dock_id", "vina_kcal_mol"]], on="dock_id", how="inner")
                frames.append(merged[["ID", "vina_kcal_mol"]])

    if "smiles" in vina.columns:
        vina_smiles = vina[["smiles", "vina_kcal_mol"]].copy()
        vina_smiles["canonical_smiles_for_docking"] = vina_smiles["smiles"].map(canonicalize)
        map_path = submission_dir / "record_to_docking_id_all.csv"
        if map_path.exists():
            mapping = pd.read_csv(map_path)
            merged = mapping[["ID", "canonical_smiles_for_docking"]].merge(
                vina_smiles[["canonical_smiles_for_docking", "vina_kcal_mol"]],
                on="canonical_smiles_for_docking",
                how="inner",
            )
            frames.append(merged[["ID", "vina_kcal_mol"]])

    if not frames:
        raise ValueError("Returned Vina CSV must contain one of: dock_id, ID, or smiles.")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["vina_kcal_mol"])
    out = out.drop_duplicates("ID", keep="first")
    return out


def merge_stage(pool_path: Path, vina_by_id: pd.DataFrame, vina_threshold: float) -> pd.DataFrame:
    pool = pd.read_csv(pool_path)
    merged = pool.merge(vina_by_id, on="ID", how="left")
    merged["real_vina_supported"] = merged["vina_kcal_mol"] <= vina_threshold
    merged["gold_real_vina"] = merged["pre_docking_goldlike"].astype(bool) & merged["real_vina_supported"]
    return merged


def summarize_final(final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = (
        final.groupby(["method", "seed"])
        .agg(
            n_scored=("vina_kcal_mol", lambda x: int(x.notna().sum())),
            hard_pass=("pre_docking_goldlike", "sum"),
            vina_supported=("real_vina_supported", "sum"),
            gold_real_vina=("gold_real_vina", "sum"),
            mean_real_vina=("vina_kcal_mol", "mean"),
            best_real_vina=("vina_kcal_mol", "min"),
            mean_pred_logsw=("pred_logsw_reencoded", "mean"),
            mean_p_sweet=("p_sweet_reencoded", "mean"),
        )
        .reset_index()
    )
    summary = (
        by_seed.groupby("method")
        .agg(
            n_seeds=("seed", "count"),
            gold_real_mean=("gold_real_vina", "mean"),
            gold_real_sd=("gold_real_vina", "std"),
            vina_supported_mean=("vina_supported", "mean"),
            vina_supported_sd=("vina_supported", "std"),
            mean_real_vina=("mean_real_vina", "mean"),
            best_real_vina=("best_real_vina", "min"),
            mean_pred_logsw=("mean_pred_logsw", "mean"),
            mean_p_sweet=("mean_p_sweet", "mean"),
        )
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    return by_seed, summary


def plot_final_scatter(final: pd.DataFrame, out_dir: Path, vina_threshold: float) -> None:
    scored = final[final["vina_kcal_mol"].notna()].copy()
    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    add_header(
        fig,
        "Final candidates under real docking",
        f"Top-10 decoded/re-encoded pool per seed; filled points satisfy sweet gate, logSw, OOD, and Vina <= {vina_threshold:.1f}",
    )
    for method in METHOD_ORDER:
        sub = scored[scored["method"] == method].copy()
        if sub.empty:
            continue
        ok = sub["gold_real_vina"].astype(bool)
        ax.scatter(
            sub.loc[~ok, "pred_logsw_reencoded"],
            sub.loc[~ok, "vina_kcal_mol"],
            s=64,
            marker=MARKERS[method],
            facecolors="white",
            edgecolors=COLORS[method],
            linewidths=1.55,
            alpha=0.72,
        )
        ax.scatter(
            sub.loc[ok, "pred_logsw_reencoded"],
            sub.loc[ok, "vina_kcal_mol"],
            s=92,
            marker=MARKERS[method],
            facecolors=COLORS[method],
            edgecolors="#FFFFFF",
            linewidths=0.9,
            alpha=0.9,
            label=METHOD_LABELS[method],
        )
    ax.axhline(vina_threshold, color=TOKENS["ink"], lw=1.5, ls="--", alpha=0.65)
    ax.axvline(2.60, color=TOKENS["ink"], lw=1.5, ls="--", alpha=0.65)
    ax.set_xlabel("Predicted logSw after re-encoding")
    ax.set_ylabel("Real Vina score (kcal/mol)")
    if not scored.empty:
        ymin = np.floor(scored["vina_kcal_mol"].min() * 2) / 2 - 0.2
        ymax = np.ceil(scored["vina_kcal_mol"].max() * 2) / 2 + 0.2
        ax.set_ylim(ymax, ymin)
    polish(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.50), frameon=False, ncol=1, handlelength=1.4)
    fig.subplots_adjust(top=0.84, left=0.13, right=0.76, bottom=0.14)
    save(fig, out_dir, "v8_real_vina_final_scatter")


def plot_gold_count_summary(by_seed: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    add_header(
        fig,
        "Real-docking supported hits",
        "Mean±SD gold hits per seed in the final top-10 external pool",
    )
    x = np.arange(len(METHOD_ORDER))
    means = summary["gold_real_mean"].fillna(0).to_numpy()
    sds = summary["gold_real_sd"].fillna(0).to_numpy()
    ax.bar(x, means, yerr=sds, color=[COLORS[m] for m in METHOD_ORDER], edgecolor=TOKENS["ink"], linewidth=1.15, capsize=5, alpha=0.88)
    for i, method in enumerate(METHOD_ORDER):
        vals = by_seed.loc[by_seed["method"] == method, "gold_real_vina"].to_numpy()
        jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) else []
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=45, color="#2A2F3A", alpha=0.75, zorder=4)
        ax.text(i, means[i] + sds[i] + 0.18, f"{means[i]:.1f}", ha="center", va="bottom", fontsize=14, fontweight="bold", color=TOKENS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m].replace(" Latent GA", "\nLatent GA") for m in METHOD_ORDER])
    ax.set_ylabel("Gold hits per seed")
    ax.set_ylim(bottom=0)
    polish(ax)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(top=0.82, left=0.14, right=0.97, bottom=0.20)
    save(fig, out_dir, "v8_real_vina_gold_count_summary")


def plot_generation_gold_evolution(generation: pd.DataFrame, out_dir: Path) -> None:
    scored = generation[generation["vina_kcal_mol"].notna()].copy()
    if scored.empty:
        return
    per_seed = (
        scored.groupby(["method", "seed", "generation"])
        .agg(gold_real=("gold_real_vina", "sum"))
        .reset_index()
    )
    summary = (
        per_seed.groupby(["method", "generation"])
        .agg(mean_gold=("gold_real", "mean"), sd_gold=("gold_real", "std"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    add_header(
        fig,
        "Real-docking supported evolution",
        "Mean gold hits per seed among generation top-5 docking-audit candidates",
    )
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("generation")
        if sub.empty:
            continue
        x = sub["generation"].to_numpy()
        y = sub["mean_gold"].to_numpy()
        sd = sub["sd_gold"].fillna(0).to_numpy()
        ax.step(x, y, where="mid", lw=2.8, color=COLORS[method], label=METHOD_LABELS[method])
        ax.scatter(x, y, s=56, marker=MARKERS[method], facecolors="white", edgecolors=COLORS[method], linewidths=1.8)
        ax.fill_between(x, np.clip(y - sd, 0, None), y + sd, color=COLORS[method], alpha=0.10, step="mid", linewidth=0)
    for x in [3, 6, 9]:
        ax.axvline(x, color=TOKENS["axis"], lw=1.2, ls=":", zorder=0)
    ax.set_xlim(1, 12)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Real-docking gold hits in top-5")
    polish(ax)
    ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=2.0, fontsize=14.5)
    fig.subplots_adjust(top=0.84, left=0.13, right=0.97, bottom=0.14)
    save(fig, out_dir, "v8_real_vina_generation_gold_evolution")
    per_seed.to_csv(out_dir / "v8_real_vina_generation_by_seed.csv", index=False)
    summary.to_csv(out_dir / "v8_real_vina_generation_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_dir", required=True)
    parser.add_argument("--submission_dir", required=True)
    parser.add_argument("--vina_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vina_threshold", type=float, default=-6.8)
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    submission_dir = Path(args.submission_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    vina = normalize_vina_csv(Path(args.vina_csv))
    vina_by_id = load_vina_by_original_id(vina, submission_dir)
    vina_by_id.to_csv(out_dir / "v8_real_vina_by_original_id.csv", index=False)

    final = merge_stage(audit_dir / "final_docking_candidate_pool.csv", vina_by_id, args.vina_threshold)
    generation = merge_stage(audit_dir / "generation_docking_candidate_pool.csv", vina_by_id, args.vina_threshold)
    final.to_csv(out_dir / "v8_final_real_vina_scored.csv", index=False)
    generation.to_csv(out_dir / "v8_generation_real_vina_scored.csv", index=False)

    by_seed, summary = summarize_final(final)
    by_seed.to_csv(out_dir / "v8_final_real_vina_by_seed.csv", index=False)
    summary.to_csv(out_dir / "v8_final_real_vina_summary.csv", index=False)
    plot_final_scatter(final, out_dir, args.vina_threshold)
    plot_gold_count_summary(by_seed, summary, out_dir)
    plot_generation_gold_evolution(generation, out_dir)

    manifest = {
        "vina_csv": str(args.vina_csv),
        "scored_original_ids": int(vina_by_id["ID"].nunique()),
        "final_scored_rows": int(final["vina_kcal_mol"].notna().sum()),
        "generation_scored_rows": int(generation["vina_kcal_mol"].notna().sum()),
        "vina_threshold": args.vina_threshold,
    }
    (out_dir / "v8_real_vina_merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
