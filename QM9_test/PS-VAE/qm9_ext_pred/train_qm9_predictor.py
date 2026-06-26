#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


QM9_PROPS = ["homo", "lumo", "gap", "u0", "u298", "h298", "g298"]


class Predictor(nn.Module):
    def __init__(self, dim_feature, dim_hidden, num_property, dropout=0.2):
        super(Predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_feature, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(dim_hidden, num_property)

    def forward(self, x):
        hidden = self.mlp(x)
        return self.output(hidden)


class LatentDataset(Dataset):
    def __init__(self, x, y):
        print("[DEBUG] building LatentDataset...", flush=True)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        print(f"[DEBUG] dataset built: x={self.x.shape}, y={self.y.shape}", flush=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def physically_consistent_loss(
    pred,
    target,
    lambda_order=2.0,
    lambda_consistency=2.0,
    lambda_positive=2.0,
    margin=1e-4,
):
    """
    pred / target shape: [B, 7]
    order:
        [homo, lumo, gap, u0, u298, h298, g298]
    """
    # 原始7维监督
    data_loss = torch.mean((pred - target) ** 2)

    homo = pred[:, 0]
    lumo = pred[:, 1]
    gap = pred[:, 2]

    # 约束1：lumo > homo
    order_penalty = torch.relu(homo - lumo + margin).mean()

    # 约束2：gap ≈ lumo - homo
    consistency_penalty = torch.abs(gap - (lumo - homo)).mean()

    # 约束3：gap >= 0
    positive_penalty = torch.relu(-gap + margin).mean()

    total_loss = (
        data_loss
        + lambda_order * order_penalty
        + lambda_consistency * consistency_penalty
        + lambda_positive * positive_penalty
    )

    loss_dict = {
        "total_loss": total_loss.detach(),
        "data_loss": data_loss.detach(),
        "order_penalty": order_penalty.detach(),
        "consistency_penalty": consistency_penalty.detach(),
        "positive_penalty": positive_penalty.detach(),
    }
    return total_loss, loss_dict


def evaluate(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    total_num = 0

    pred_all = []
    true_all = []

    penalty_sums = {
        "data_loss": 0.0,
        "order_penalty": 0.0,
        "consistency_penalty": 0.0,
        "positive_penalty": 0.0,
    }

    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss, loss_dict = physically_consistent_loss(
                pred,
                y,
                lambda_order=args.lambda_order,
                lambda_consistency=args.lambda_consistency,
                lambda_positive=args.lambda_positive,
                margin=args.margin,
            )

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_num += bs

            for k in penalty_sums:
                penalty_sums[k] += loss_dict[k].item() * bs

            pred_all.append(pred.cpu().numpy())
            true_all.append(y.cpu().numpy())

            if step == 0:
                print(f"[DEBUG] evaluate first batch: x={x.shape}, y={y.shape}, pred={pred.shape}", flush=True)

    pred_all = np.concatenate(pred_all, axis=0)
    true_all = np.concatenate(true_all, axis=0)

    mae = np.mean(np.abs(pred_all - true_all), axis=0)

    avg_penalties = {k: v / total_num for k, v in penalty_sums.items()}
    return total_loss / total_num, mae, avg_penalties


def postprocess_prediction_numpy(pred_np, margin=1e-6):
    """
    推理阶段可选物理修正，避免极少数不合理输出。
    pred_np shape: [N, 7] or [7]
    """
    was_1d = False
    if pred_np.ndim == 1:
        pred_np = pred_np[None, :]
        was_1d = True

    out = pred_np.copy()

    homo = out[:, 0]
    lumo = out[:, 1]
    gap = out[:, 2]

    gap = np.maximum(gap, margin)

    bad_mask = lumo <= homo + margin
    lumo[bad_mask] = homo[bad_mask] + gap[bad_mask]

    # 强制三者一致
    gap = lumo - homo

    out[:, 1] = lumo
    out[:, 2] = gap

    return out[0] if was_1d else out


def main(args):
    print("[DEBUG] script start", flush=True)
    print("[DEBUG] enter main()", flush=True)
    print(f"[DEBUG] args = {args}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[DEBUG] save_dir ensured: {args.save_dir}", flush=True)

    print("[DEBUG] loading npy files...", flush=True)
    x_train = np.load(args.x_train)
    print(f"[DEBUG] loaded x_train: {x_train.shape}", flush=True)

    x_valid = np.load(args.x_valid)
    print(f"[DEBUG] loaded x_valid: {x_valid.shape}", flush=True)

    x_test = np.load(args.x_test)
    print(f"[DEBUG] loaded x_test: {x_test.shape}", flush=True)

    y_train = np.load(args.y_train)
    print(f"[DEBUG] loaded y_train: {y_train.shape}", flush=True)

    y_valid = np.load(args.y_valid)
    print(f"[DEBUG] loaded y_valid: {y_valid.shape}", flush=True)

    y_test = np.load(args.y_test)
    print(f"[DEBUG] loaded y_test: {y_test.shape}", flush=True)

    assert x_train.ndim == 2, f"x_train ndim must be 2, got {x_train.ndim}"
    assert x_valid.ndim == 2, f"x_valid ndim must be 2, got {x_valid.ndim}"
    assert x_test.ndim == 2, f"x_test ndim must be 2, got {x_test.ndim}"

    assert y_train.ndim == 2, f"y_train ndim must be 2, got {y_train.ndim}"
    assert y_valid.ndim == 2, f"y_valid ndim must be 2, got {y_valid.ndim}"
    assert y_test.ndim == 2, f"y_test ndim must be 2, got {y_test.ndim}"

    assert y_train.shape[1] == 7, f"y_train second dim must be 7, got {y_train.shape}"
    assert y_valid.shape[1] == 7, f"y_valid second dim must be 7, got {y_valid.shape}"
    assert y_test.shape[1] == 7, f"y_test second dim must be 7, got {y_test.shape}"

    print("[DEBUG] computing normalization...", flush=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-8

    y_train_n = (y_train - y_mean) / y_std
    y_valid_n = (y_valid - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    np.save(os.path.join(args.save_dir, "y_mean.npy"), y_mean)
    np.save(os.path.join(args.save_dir, "y_std.npy"), y_std)
    print("[DEBUG] saved normalization stats", flush=True)

    print("[DEBUG] building datasets...", flush=True)
    train_set = LatentDataset(x_train, y_train_n)
    valid_set = LatentDataset(x_valid, y_valid_n)
    test_set = LatentDataset(x_test, y_test_n)

    print("[DEBUG] building dataloaders...", flush=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print("[DEBUG] dataloaders built", flush=True)

    dim_feature = x_train.shape[1]
    print(f"[DEBUG] dim_feature = {dim_feature}", flush=True)

    model = Predictor(
        dim_feature=dim_feature,
        dim_hidden=args.hidden_dim,
        num_property=7,
        dropout=args.dropout
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[DEBUG] using device = {device}", flush=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    best_valid = float("inf")
    best_epoch = -1
    best_ckpt = os.path.join(args.save_dir, "best_predictor.pt")
    history_csv = os.path.join(args.save_dir, "train_history.csv")

    early_stop_counter = 0

    with open(history_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "valid_loss", "lr",
            "train_data_loss", "train_order_penalty", "train_consistency_penalty", "train_positive_penalty",
            "valid_data_loss", "valid_order_penalty", "valid_consistency_penalty", "valid_positive_penalty",
            *[f"val_mae_{p}" for p in QM9_PROPS]
        ])
    print(f"[DEBUG] history csv initialized: {history_csv}", flush=True)

    print("[DEBUG] starting training loop...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_num = 0

        train_penalty_sums = {
            "data_loss": 0.0,
            "order_penalty": 0.0,
            "consistency_penalty": 0.0,
            "positive_penalty": 0.0,
        }

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss, loss_dict = physically_consistent_loss(
                pred,
                y,
                lambda_order=args.lambda_order,
                lambda_consistency=args.lambda_consistency,
                lambda_positive=args.lambda_positive,
                margin=args.margin,
            )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_num += bs

            for k in train_penalty_sums:
                train_penalty_sums[k] += loss_dict[k].item() * bs

            if step == 0:
                print(
                    f"[DEBUG] epoch {epoch} first train batch: "
                    f"x={x.shape}, y={y.shape}, pred={pred.shape}, loss={loss.item():.6f}",
                    flush=True
                )

        train_loss = total_loss / total_num
        train_penalties = {k: v / total_num for k, v in train_penalty_sums.items()}

        valid_loss, valid_mae, valid_penalties = evaluate(model, valid_loader, device, args)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(valid_loss)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} lr={current_lr:.6e}", flush=True)
        print(
            f"  train penalties: data={train_penalties['data_loss']:.6f}, "
            f"order={train_penalties['order_penalty']:.6f}, "
            f"cons={train_penalties['consistency_penalty']:.6f}, "
            f"pos={train_penalties['positive_penalty']:.6f}",
            flush=True
        )
        print(
            f"  valid penalties: data={valid_penalties['data_loss']:.6f}, "
            f"order={valid_penalties['order_penalty']:.6f}, "
            f"cons={valid_penalties['consistency_penalty']:.6f}, "
            f"pos={valid_penalties['positive_penalty']:.6f}",
            flush=True
        )
        for p, m in zip(QM9_PROPS, valid_mae):
            print(f"  val_mae_norm {p:10s}: {m:.6f}", flush=True)

        with open(history_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, valid_loss, current_lr,
                train_penalties["data_loss"], train_penalties["order_penalty"],
                train_penalties["consistency_penalty"], train_penalties["positive_penalty"],
                valid_penalties["data_loss"], valid_penalties["order_penalty"],
                valid_penalties["consistency_penalty"], valid_penalties["positive_penalty"],
                *valid_mae.tolist()
            ])

        if valid_loss < best_valid - args.min_delta:
            best_valid = valid_loss
            best_epoch = epoch
            early_stop_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "dim_feature": dim_feature,
                "hidden_dim": args.hidden_dim,
                "num_property": 7,
                "dropout": args.dropout,
                "property_names": QM9_PROPS,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid,
                "lambda_order": args.lambda_order,
                "lambda_consistency": args.lambda_consistency,
                "lambda_positive": args.lambda_positive,
                "margin": args.margin
            }, best_ckpt)
            print(f"[DEBUG] Saved best model -> {best_ckpt}", flush=True)
        else:
            early_stop_counter += 1
            print(f"[DEBUG] no improvement count = {early_stop_counter}/{args.patience}", flush=True)

        if early_stop_counter >= args.patience:
            print(f"[DEBUG] Early stopping triggered at epoch {epoch}, best_epoch={best_epoch}, best_valid={best_valid:.6f}", flush=True)
            break

    print("[DEBUG] loading best checkpoint for final test...", flush=True)
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_mae_norm, test_penalties = evaluate(model, test_loader, device, args)
    print(f"\n[Test] normalized loss = {test_loss:.6f}", flush=True)
    print(
        f"[Test] penalties: data={test_penalties['data_loss']:.6f}, "
        f"order={test_penalties['order_penalty']:.6f}, "
        f"cons={test_penalties['consistency_penalty']:.6f}, "
        f"pos={test_penalties['positive_penalty']:.6f}",
        flush=True
    )
    for p, m in zip(QM9_PROPS, test_mae_norm):
        print(f"  test_mae_norm {p:10s}: {m:.6f}", flush=True)

    model.eval()
    pred_all = []
    true_all = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            pred = postprocess_prediction_numpy(pred, margin=args.margin)
            pred_all.append(pred)
            true_all.append(y.numpy())

    pred_all = np.concatenate(pred_all, axis=0)
    true_all = np.concatenate(true_all, axis=0)

    pred_denorm = pred_all * y_std + y_mean
    true_denorm = true_all * y_std + y_mean

    mae_denorm = np.mean(np.abs(pred_denorm - true_denorm), axis=0)
    print("\n[Test] de-normalized MAE:", flush=True)
    for p, m in zip(QM9_PROPS, mae_denorm):
        print(f"  {p:10s}: {m:.6f}", flush=True)

    # 检查物理一致性
    homo_pred = pred_denorm[:, 0]
    lumo_pred = pred_denorm[:, 1]
    gap_pred = pred_denorm[:, 2]

    bad_order = np.sum(lumo_pred <= homo_pred)
    bad_gap = np.sum(gap_pred < 0)
    inconsistent = np.mean(np.abs(gap_pred - (lumo_pred - homo_pred)))

    print(f"\n[Test] physical consistency check:", flush=True)
    print(f"  count(lumo <= homo): {bad_order}", flush=True)
    print(f"  count(gap < 0):      {bad_gap}", flush=True)
    print(f"  mean|gap-(lumo-homo)|: {inconsistent:.6f}", flush=True)

    np.save(os.path.join(args.save_dir, "test_pred.npy"), pred_denorm)
    np.save(os.path.join(args.save_dir, "test_true.npy"), true_denorm)

    print(f"[DEBUG] best_epoch = {ckpt.get('best_epoch', 'NA')}", flush=True)
    print(f"[DEBUG] best_valid_loss = {ckpt.get('best_valid_loss', 'NA')}", flush=True)
    print("[DEBUG] finished successfully", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--x_train", type=str, required=True)
    parser.add_argument("--x_valid", type=str, required=True)
    parser.add_argument("--x_test", type=str, required=True)

    parser.add_argument("--y_train", type=str, required=True)
    parser.add_argument("--y_valid", type=str, required=True)
    parser.add_argument("--y_test", type=str, required=True)

    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_order", type=float, default=2.0)
    parser.add_argument("--lambda_consistency", type=float, default=2.0)
    parser.add_argument("--lambda_positive", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=1e-4)

    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    main(args)