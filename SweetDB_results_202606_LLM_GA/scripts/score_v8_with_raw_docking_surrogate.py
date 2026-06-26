#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path("/home/jqb/Apply_Sweet/Apply_Sweet_results_202605_LLM_GA")
RESULT_ROOT = ROOT / "sweet_ga_results_0622_v8_hard_metrics"
DOCKING_DIR = ROOT / "latent_evaluator_data_manifold_v2" / "docking_surrogate_raw_vina_v1"
OUT_DIR = RESULT_ROOT / "nature_style_panels" / "main_3panels"

METHODS = {
    "group_a_random": ("A", "Random-Seeded Latent GA"),
    "group_b_dataset": ("B", "SweetDB-Seeded Latent GA"),
    "group_c_llm": ("C", "LLM-Initialized Latent GA"),
    "group_d_llm_iterative": ("D", "Iterative LLM-Guided Latent GA"),
}


class LatentRegressor(nn.Module):
    def __init__(self, latent_dim: int = 56, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_ensemble():
    with open(DOCKING_DIR / "latent_docking_ensemble_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    members = []
    for member in summary["models"]:
        member_dir = DOCKING_DIR / member["dir"]
        bundle = safe_load(member_dir / "latent_docking_regressor_bundle.pt")
        model = LatentRegressor(
            latent_dim=int(summary.get("latent_dim", 56)),
            hidden_dim=int(bundle.get("hidden_dim", 128)),
            dropout=float(bundle.get("dropout", 0.1)),
        )
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        members.append(
            {
                "model": model,
                "scaler_z": joblib.load(member_dir / "latent_docking_scaler_z.pkl"),
                "scaler_y": joblib.load(member_dir / "latent_docking_scaler_y.pkl"),
            }
        )
    return summary, members


def predict_support(z: np.ndarray, members):
    preds = []
    with torch.no_grad():
        for member in members:
            z_scaled = member["scaler_z"].transform(z)
            x = torch.tensor(z_scaled, dtype=torch.float32)
            pred_scaled = member["model"](x).detach().cpu().numpy().reshape(-1, 1)
            pred_raw = member["scaler_y"].inverse_transform(pred_scaled).reshape(-1)
            preds.append(pred_raw.astype(np.float32))
    arr = np.stack(preds, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def method_from_dir(path: Path):
    name = path.name
    for prefix, value in METHODS.items():
        if name.startswith(prefix):
            return value
    return None, None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, members = load_ensemble()
    rows = []
    for run_dir in sorted(RESULT_ROOT.glob("group_*_ABCD_v8_hard_metrics_0622_*_seed*")):
        method, label = method_from_dir(run_dir)
        if method is None:
            continue
        corrected_path = run_dir / "final_population_corrected.csv"
        latent_path = run_dir / "final_population_latent.npy"
        if not corrected_path.exists() or not latent_path.exists():
            continue
        corrected = pd.read_csv(corrected_path)
        z = np.load(latent_path)
        if len(corrected) != len(z):
            raise RuntimeError(f"Row/latent mismatch in {run_dir}: {len(corrected)} vs {len(z)}")
        support, support_sd = predict_support(z, members)
        corrected = corrected.copy()
        corrected["method"] = method
        corrected["method_label"] = label
        corrected["run_dir"] = run_dir.name
        corrected["seed"] = corrected["run_dir"].str.extract(r"seed(\d+)$").astype(int)
        corrected["pred_binding_support_raw_vina_surrogate"] = support
        corrected["pred_binding_support_sd_raw_vina_surrogate"] = support_sd
        corrected["pred_vina_kcal_mol_raw_surrogate"] = -support
        corrected["pred_vina_sd_kcal_mol_raw_surrogate"] = support_sd
        rows.append(corrected)
    all_final = pd.concat(rows, ignore_index=True)
    top10 = all_final[pd.to_numeric(all_final["final_rank"], errors="coerce").le(10)].copy()

    # Keep the same external hard-pass definition used in v8 plots.
    top10["pre_docking_hard_pass"] = (
        top10["valid"].astype(str).str.lower().eq("true")
        & top10["reencode_ok"].astype(str).str.lower().eq("true")
        & pd.to_numeric(top10["p_sweet_reencoded"], errors="coerce").ge(0.80)
        & pd.to_numeric(top10["pred_logsw_reencoded"], errors="coerce").ge(2.60)
        & pd.to_numeric(top10["d_ood_reencoded"], errors="coerce").le(7.225)
    )

    all_final.to_csv(OUT_DIR / "v8_all_final_population_with_raw_docking_surrogate.csv", index=False)
    top10.to_csv(OUT_DIR / "v8_final_top10_with_raw_docking_surrogate.csv", index=False)
    with open(OUT_DIR / "raw_docking_surrogate_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_model": str(DOCKING_DIR),
                "target": summary.get("target"),
                "target_transform": summary.get("target_transform"),
                "oof_mae": summary.get("oof_mae"),
                "oof_r2": summary.get("oof_r2"),
                "oof_spearman": summary.get("oof_spearman"),
                "n_final_population_rows": int(len(all_final)),
                "n_final_top10_rows": int(len(top10)),
            },
            handle,
            indent=2,
        )
    print(json.dumps({"all_final": len(all_final), "top10": len(top10)}, indent=2))


if __name__ == "__main__":
    main()
