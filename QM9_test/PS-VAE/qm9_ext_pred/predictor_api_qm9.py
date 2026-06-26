#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn


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


class QM9PredictorAPI:
    def __init__(self, predictor_ckpt, mean_path, std_path, device="cpu"):
        self.device = torch.device(device)

        ckpt = torch.load(predictor_ckpt, map_location=self.device)

        self.model = Predictor(
            dim_feature=ckpt["dim_feature"],
            dim_hidden=ckpt["hidden_dim"],
            num_property=ckpt["num_property"],
            dropout=ckpt.get("dropout", 0.0)
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.property_names = ckpt["property_names"]
        self.y_mean = np.load(mean_path)
        self.y_std = np.load(std_path)

    def enforce_physical_constraints(self, pred, margin=1e-6):
        pred = pred.copy()

        homo = pred[:, 0]
        lumo = pred[:, 1]
        gap = pred[:, 2]

        gap = np.maximum(gap, margin)

        bad_mask = lumo <= homo + margin
        lumo[bad_mask] = homo[bad_mask] + gap[bad_mask]

        gap = lumo - homo

        pred[:, 1] = lumo
        pred[:, 2] = gap

        return pred

    def predict_array(self, z):
        """
        z: np.ndarray, shape [D] or [N, D]
        return: np.ndarray, shape [N, 7]
        """
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z[None, :]

        x = torch.tensor(z, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_norm = self.model(x).cpu().numpy()

        pred = pred_norm * self.y_std + self.y_mean
        pred = self.enforce_physical_constraints(pred)

        return pred

    def predict_dict(self, z):
        pred = self.predict_array(z)
        out = []
        for row in pred:
            out.append({k: float(v) for k, v in zip(self.property_names, row)})
        return out if len(out) > 1 else out[0]