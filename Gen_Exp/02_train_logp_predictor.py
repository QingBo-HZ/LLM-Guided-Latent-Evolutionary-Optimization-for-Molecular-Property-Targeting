#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

import json
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class LatentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LogPPredictor(nn.Module):
    def __init__(self, dim_feature, hidden_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_feature, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_split(latent_path, meta_path, target_col="logP", split_name="train"):
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"[{split_name}] latent file not found: {latent_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"[{split_name}] meta file not found: {meta_path}")

    X = np.load(latent_path).astype(np.float32)
    meta = pd.read_csv(meta_path)

    if X.ndim != 2:
        raise ValueError(f"[{split_name}] latent must be 2D, got shape={X.shape}")

    if target_col not in meta.columns:
        raise ValueError(
            f"[{split_name}] target column '{target_col}' not found. "
            f"Available columns: {list(meta.columns)}"
        )

    y = meta[target_col].values.astype(np.float32)

    if len(X) != len(y):
        raise ValueError(
            f"[{split_name}] X/y size mismatch: X={len(X)}, y={len(y)}"
        )

    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[finite]
    y = y[finite]
    meta = meta.loc[finite].reset_index(drop=True)

    if len(X) == 0:
        raise RuntimeError(f"[{split_name}] no valid samples after finite filtering.")

    print(
        f"[INFO] {split_name}: n={len(X)}, latent_dim={X.shape[1]}, "
        f"{target_col} mean={y.mean():.4f}, std={y.std():.4f}, "
        f"min={y.min():.4f}, max={y.max():.4f}"
    )

    return X, y, meta


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    try:
        pearson = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        pearson = float("nan")

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson": pearson,
        "n": int(len(y_true)),
    }


def predict(model, loader, device, y_mean, y_std):
    model.eval()

    ys = []
    ps = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            pred_norm = model(x).cpu().numpy().reshape(-1)
            y_norm = y.numpy().reshape(-1)

            pred = pred_norm * y_std + y_mean
            true = y_norm * y_std + y_mean

            ps.append(pred)
            ys.append(true)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    return y_true, y_pred


def save_prediction_csv(out_path, split_name, y_true, y_pred):
    df = pd.DataFrame({
        "split": split_name,
        "true_logP": y_true,
        "pred_logP": y_pred,
    })
    df["abs_error"] = np.abs(df["true_logP"] - df["pred_logP"])
    df.to_csv(out_path, index=False)
    return df


def plot_pred_vs_true(y_true, y_pred, metrics, out_path, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.55)

    mn = min(float(np.min(y_true)), float(np.min(y_pred)))
    mx = max(float(np.max(y_true)), float(np.max(y_pred)))
    pad = (mx - mn) * 0.05 if mx > mn else 1.0

    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "--", linewidth=1.2)

    plt.xlabel("True logP")
    plt.ylabel("Predicted logP")
    plt.title(title)

    txt = (
        f"R² = {metrics['R2']:.3f}\n"
        f"RMSE = {metrics['RMSE']:.3f}\n"
        f"MAE = {metrics['MAE']:.3f}\n"
        f"r = {metrics['Pearson']:.3f}"
    )

    plt.text(
        0.05,
        0.95,
        txt,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    plt.xlim(mn - pad, mx + pad)
    plt.ylim(mn - pad, mx + pad)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser("Train latent-logP predictor with fixed train/valid/test splits")

    parser.add_argument("--train_latent", type=str, required=True)
    parser.add_argument("--train_meta", type=str, required=True)

    parser.add_argument("--valid_latent", type=str, required=True)
    parser.add_argument("--valid_meta", type=str, required=True)

    parser.add_argument("--test_latent", type=str, required=True)
    parser.add_argument("--test_meta", type=str, required=True)

    parser.add_argument("--target_col", type=str, default="logP")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )

    print("\n========== CONFIG ==========")
    print(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f"[INFO] device = {device}")

    # =========================
    # Load fixed splits
    # =========================

    X_train, y_train, meta_train = load_split(
        args.train_latent,
        args.train_meta,
        target_col=args.target_col,
        split_name="train"
    )

    X_valid, y_valid, meta_valid = load_split(
        args.valid_latent,
        args.valid_meta,
        target_col=args.target_col,
        split_name="valid"
    )

    X_test, y_test, meta_test = load_split(
        args.test_latent,
        args.test_meta,
        target_col=args.target_col,
        split_name="test"
    )

    latent_dim = X_train.shape[1]

    if X_valid.shape[1] != latent_dim:
        raise ValueError(f"valid latent dim mismatch: {X_valid.shape[1]} vs train {latent_dim}")
    if X_test.shape[1] != latent_dim:
        raise ValueError(f"test latent dim mismatch: {X_test.shape[1]} vs train {latent_dim}")

    # =========================
    # Normalize by train only
    # =========================

    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True)
    X_std = np.where(X_std < 1e-6, 1.0, X_std)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-6:
        y_std = 1.0

    X_train_n = (X_train - X_mean) / X_std
    X_valid_n = (X_valid - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std

    y_train_n = (y_train - y_mean) / y_std
    y_valid_n = (y_valid - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    print("\n========== NORMALIZATION ==========")
    print(f"[INFO] y_mean(train) = {y_mean:.6f}")
    print(f"[INFO] y_std(train)  = {y_std:.6f}")

    np.save(os.path.join(args.out_dir, "X_mean.npy"), X_mean.astype(np.float32))
    np.save(os.path.join(args.out_dir, "X_std.npy"), X_std.astype(np.float32))
    np.save(os.path.join(args.out_dir, "y_mean.npy"), np.array([y_mean], dtype=np.float32))
    np.save(os.path.join(args.out_dir, "y_std.npy"), np.array([y_std], dtype=np.float32))

    # =========================
    # Data loaders
    # =========================

    train_ds = LatentDataset(X_train_n, y_train_n)
    valid_ds = LatentDataset(X_valid_n, y_valid_n)
    test_ds = LatentDataset(X_test_n, y_test_n)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    # =========================
    # Model
    # =========================

    model = LogPPredictor(
        dim_feature=latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    wait = 0
    history = []

    best_ckpt_path = os.path.join(args.out_dir, "best_logp_predictor.pt")

    # =========================
    # Training
    # =========================

    print("\n========== TRAINING ==========")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        valid_losses = []

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)
                valid_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))

        history.append({
            "epoch": int(epoch),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f}, "
                f"valid_loss={valid_loss:.6f}"
            )

        if valid_loss < best_val:
            best_val = valid_loss
            best_epoch = epoch
            wait = 0

            ckpt = {
                "model_state_dict": model.state_dict(),
                "dim_feature": int(latent_dim),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "target_name": args.target_col,
                "X_mean": X_mean.astype(np.float32),
                "X_std": X_std.astype(np.float32),
                "y_mean": float(y_mean),
                "y_std": float(y_std),
                "seed": int(args.seed),
                "split_mode": "fixed_train_valid_test",
                "train_latent": args.train_latent,
                "valid_latent": args.valid_latent,
                "test_latent": args.test_latent,
            }

            torch.save(ckpt, best_ckpt_path)

        else:
            wait += 1

        if wait >= args.patience:
            print(
                f"[Early Stop] epoch={epoch}, "
                f"best_epoch={best_epoch}, best_valid_loss={best_val:.6f}"
            )
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.out_dir, "training_history.csv"), index=False)

    # =========================
    # Load best model
    # =========================

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # =========================
    # Evaluation
    # =========================

    print("\n========== EVALUATION ==========")

    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    valid_eval_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_eval_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    y_train_true, y_train_pred = predict(model, train_eval_loader, device, y_mean, y_std)
    y_valid_true, y_valid_pred = predict(model, valid_eval_loader, device, y_mean, y_std)
    y_test_true, y_test_pred = predict(model, test_eval_loader, device, y_mean, y_std)

    train_metrics = calc_metrics(y_train_true, y_train_pred)
    valid_metrics = calc_metrics(y_valid_true, y_valid_pred)
    test_metrics = calc_metrics(y_test_true, y_test_pred)

    print("[RESULT] Train metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v}")

    print("[RESULT] Valid metrics:")
    for k, v in valid_metrics.items():
        print(f"  {k}: {v}")

    print("[RESULT] Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # =========================
    # Save outputs
    # =========================

    metrics = {
        "best_epoch": int(best_epoch),
        "best_valid_loss_norm": float(best_val),
        "train": train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
        "dataset": {
            "n_train": int(len(X_train)),
            "n_valid": int(len(X_valid)),
            "n_test": int(len(X_test)),
            "latent_dim": int(latent_dim),
            "target_col": args.target_col,
        },
        "normalization": {
            "y_mean_train": float(y_mean),
            "y_std_train": float(y_std),
        },
        "model": {
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        }
    }

    metrics_path = os.path.join(args.out_dir, "logp_predictor_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    train_pred_df = save_prediction_csv(
        os.path.join(args.out_dir, "logp_predictions_train.csv"),
        "train",
        y_train_true,
        y_train_pred
    )

    valid_pred_df = save_prediction_csv(
        os.path.join(args.out_dir, "logp_predictions_valid.csv"),
        "valid",
        y_valid_true,
        y_valid_pred
    )

    test_pred_df = save_prediction_csv(
        os.path.join(args.out_dir, "logp_predictions_test.csv"),
        "test",
        y_test_true,
        y_test_pred
    )

    all_pred_df = pd.concat(
        [train_pred_df, valid_pred_df, test_pred_df],
        axis=0,
        ignore_index=True
    )
    all_pred_df.to_csv(os.path.join(args.out_dir, "logp_predictions_all.csv"), index=False)

    plot_pred_vs_true(
        y_train_true,
        y_train_pred,
        train_metrics,
        os.path.join(args.out_dir, "logp_pred_vs_true_train.png"),
        "Latent-logP predictor on train set"
    )

    plot_pred_vs_true(
        y_valid_true,
        y_valid_pred,
        valid_metrics,
        os.path.join(args.out_dir, "logp_pred_vs_true_valid.png"),
        "Latent-logP predictor on validation set"
    )

    plot_pred_vs_true(
        y_test_true,
        y_test_pred,
        test_metrics,
        os.path.join(args.out_dir, "logp_pred_vs_true_test.png"),
        "Latent-logP predictor on test set"
    )

    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["valid_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (normalized logP)")
    plt.title("Latent-logP predictor training curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "training_loss_curve.png"), dpi=300)
    plt.close()

    print("\n========== DONE ==========")
    print(f"[INFO] Best checkpoint: {best_ckpt_path}")
    print(f"[INFO] Metrics:         {metrics_path}")
    print(f"[INFO] Output dir:      {args.out_dir}")


if __name__ == "__main__":
    main()