#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class LatentRegressor(nn.Module):
    def __init__(self, latent_dim=56, hidden_dim=128, dropout=0.10):
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

    def forward(self, z):
        return self.net(z).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def corr(x, y):
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def train_member(x_train, y_train, x_val, y_val, args, device, seed):
    set_seed(seed)
    scaler_z = StandardScaler().fit(x_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    train_x = torch.tensor(
        scaler_z.transform(x_train), dtype=torch.float32, device=device
    )
    train_y = torch.tensor(
        scaler_y.transform(y_train.reshape(-1, 1)).reshape(-1),
        dtype=torch.float32,
        device=device,
    )
    val_x = torch.tensor(
        scaler_z.transform(x_val), dtype=torch.float32, device=device
    )
    model = LatentRegressor(
        latent_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=0.2)
    best_state = None
    best_mae = float("inf")
    wait = 0
    history = []
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), device=device)
        losses = []
        for start in range(0, len(train_x), args.batch_size):
            idx = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred_scaled = model(val_x).detach().cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        val_mae = float(mean_absolute_error(y_val, pred))
        history.append(
            {"epoch": epoch, "train_loss": np.mean(losses), "val_mae": val_mae}
        )
        if val_mae < best_mae - 1e-5:
            best_mae = val_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(val_x).detach().cpu().numpy()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    return model, scaler_z, scaler_y, pred, history, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--docking_csv", required=True)
    parser.add_argument("--fold_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", default="recommended_docking_fitness_0_1")
    parser.add_argument(
        "--negate_target",
        action="store_true",
        help="Train on -target so that larger predictions are always better for GA.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latent_dir = Path(args.latent_dir)
    latent = np.load(latent_dir / "sweetdb_latent.npy").astype(np.float32)
    aligned = pd.read_csv(latent_dir / "sweetdb_regression_dataset_aligned.csv")
    docking = pd.read_csv(args.docking_csv)
    folds = pd.read_csv(args.fold_csv)
    merged = (
        aligned.reset_index(names="aligned_index")
        .merge(docking[["ID", args.target]], on="ID", how="left", validate="one_to_one")
        .merge(
            folds[["index", "fold"]],
            left_on="latent_index",
            right_on="index",
            how="left",
            validate="one_to_one",
        )
        .sort_values("aligned_index")
    )
    if len(latent) != len(merged):
        raise ValueError("Latent and merged row counts differ")
    if merged[[args.target, "fold"]].isna().any().any():
        raise ValueError("Missing docking targets or scaffold folds")
    target = merged[args.target].to_numpy(np.float32)
    if args.negate_target:
        target = -target
    fold_ids = merged["fold"].to_numpy(int)
    oof = np.full(len(target), np.nan, dtype=np.float32)
    model_entries = []
    histories = []
    fold_metrics = []
    for fold in sorted(np.unique(fold_ids)):
        train_idx = fold_ids != fold
        val_idx = fold_ids == fold
        member_dir = output / f"fold_{fold}"
        member_dir.mkdir(exist_ok=True)
        model, scaler_z, scaler_y, pred, history, best_epoch = train_member(
            latent[train_idx], target[train_idx],
            latent[val_idx], target[val_idx],
            args, device, args.seed + int(fold),
        )
        oof[val_idx] = pred
        torch.save(
            {
                "state_dict": model.state_dict(),
                "latent_dim": latent.shape[1],
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "fold": int(fold),
                "target": args.target,
                "best_epoch": int(best_epoch),
            },
            member_dir / "latent_docking_regressor_bundle.pt",
        )
        joblib.dump(scaler_z, member_dir / "latent_docking_scaler_z.pkl")
        joblib.dump(scaler_y, member_dir / "latent_docking_scaler_y.pkl")
        histories.extend({"fold": int(fold), **row} for row in history)
        fold_metrics.append({
            "fold": int(fold),
            "n_train": int(train_idx.sum()),
            "n_val": int(val_idx.sum()),
            "best_epoch": int(best_epoch),
            "mae": float(mean_absolute_error(target[val_idx], pred)),
            "rmse": float(mean_squared_error(target[val_idx], pred) ** 0.5),
            "r2": float(r2_score(target[val_idx], pred)),
            "pearson": corr(target[val_idx], pred),
            "spearman": corr(rankdata(target[val_idx]), rankdata(pred)),
        })
        model_entries.append({"fold": int(fold), "dir": f"fold_{fold}"})
    summary = {
        "target": args.target,
        "target_transform": "negate" if args.negate_target else "identity",
        "n_samples": len(target),
        "latent_dim": latent.shape[1],
        "split": "existing 5-fold scaffold-aware SweetDB assignments",
        "models": model_entries,
        "oof_mae": float(mean_absolute_error(target, oof)),
        "oof_rmse": float(mean_squared_error(target, oof) ** 0.5),
        "oof_r2": float(r2_score(target, oof)),
        "oof_pearson": corr(target, oof),
        "oof_spearman": corr(rankdata(target), rankdata(oof)),
    }
    pd.DataFrame(fold_metrics).to_csv(
        output / "docking_surrogate_fold_metrics.csv", index=False
    )
    pd.DataFrame(histories).to_csv(
        output / "docking_surrogate_training_history.csv", index=False
    )
    merged.assign(
        docking_target=target, docking_oof_prediction=oof
    ).to_csv(output / "docking_surrogate_oof_predictions.csv", index=False)
    with open(output / "latent_docking_ensemble_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
