"""Detektor Wide & Deep CNN (gaya Zheng dkk., IEEE TII 2018) pada deret harian mentah.

Dipakai sebagai pembanding (ekstensi 4): apakah "level tinggi = mencurigakan" juga dipelajari
model deret-waktu, bukan hanya GBM berfitur. Agar level TERLIHAT oleh model, normalisasi bersifat
GLOBAL (log1p lalu dibagi konstanta dari data latih), bukan per konsumen.

* wide : fully-connected pada deret harian (T hari)
* deep : CNN 2-D pada deret yang dilipat per minggu (W x 7)
Skor dikalibrasi Platt pada split kalibrasi, sama seperti `Detector`.
Membutuhkan PyTorch (CPU cukup); impor ditunda supaya bagian lain repo tidak bergantung padanya.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class SeriesDetector:
    """Antarmuka sama dengan `Detector`: fit(...) lalu score_rows(F, X) -> skor terkalibrasi."""

    uses_series = True

    def __init__(self, name: str, params: dict, seed: int = 0):
        self.name = name
        self.p = {"epochs": 15, "batch": 256, "lr": 1e-3, "wide": 64, "channels": 16,
                  "dropout": 0.3, "val_frac": 0.1, **(params or {})}
        self.seed = seed
        self.net = None
        self.scale = 1.0
        self.platt = None
        self.best_epoch = None

    # ---------------------------------------------------------------- model
    def _build(self, T: int):
        import torch.nn as nn
        W = T // 7
        c = self.p["channels"]

        class WideDeep(nn.Module):
            def __init__(s):
                super().__init__()
                s.W = W
                s.wide = nn.Sequential(nn.Linear(T, self.p["wide"]), nn.ReLU(), nn.Dropout(self.p["dropout"]))
                s.deep = nn.Sequential(
                    nn.Conv2d(1, c, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d((2, 1)), nn.Flatten(),
                    nn.Linear(c * (W // 2) * 7, self.p["wide"]), nn.ReLU(), nn.Dropout(self.p["dropout"]))
                s.head = nn.Linear(2 * self.p["wide"], 1)

            def forward(s, x):
                xw = x[:, : s.W * 7].reshape(-1, 1, s.W, 7)
                return s.head(__import__("torch").cat([s.wide(x), s.deep(xw)], 1)).squeeze(1)

        return WideDeep()

    def _prep(self, X):
        return (np.log1p(np.maximum(np.asarray(X, dtype=np.float32), 0.0)) / self.scale).astype(np.float32)

    def _raw(self, X):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            Z = self._prep(X)
            for i in range(0, len(Z), 4096):
                out.append(torch.sigmoid(self.net(torch.from_numpy(Z[i:i + 4096]))).numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    # ---------------------------------------------------------------- API
    def fit(self, X, y, X_cal, y_cal):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(max(1, int(os.environ.get("TIP_TORCH_THREADS", os.cpu_count() or 1))))
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32)
        self.scale = float(np.log1p(np.maximum(X, 0)).std() + 1e-6)
        idx = rng.permutation(len(y))
        nv = int(self.p["val_frac"] * len(y))
        va, trn = idx[:nv], idx[nv:]
        Ztr, ytr = torch.from_numpy(self._prep(X[trn])), torch.from_numpy(y[trn])
        self.net = self._build(X.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.p["lr"], weight_decay=1e-5)
        prev = float(ytr.mean())
        pw = torch.tensor((1 - prev) / max(prev, 1e-6))
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        best, best_state = -1.0, None
        for ep in range(self.p["epochs"]):
            self.net.train()
            perm = torch.from_numpy(rng.permutation(len(ytr)))
            for i in range(0, len(perm), self.p["batch"]):
                b = perm[i:i + self.p["batch"]]
                opt.zero_grad()
                lossf(self.net(Ztr[b]), ytr[b]).backward()
                opt.step()
            auc = roc_auc_score(y[va], self._raw(X[va]))
            if auc > best:
                best, self.best_epoch = auc, ep
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        raw = self._raw(X_cal)
        if len(np.unique(y_cal)) == 2:
            self.platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw).reshape(-1, 1), y_cal)
        return self

    def score(self, X):
        raw = self._raw(X)
        if self.platt is None:
            return raw
        return self.platt.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]

    def score_rows(self, F_rows, X_rows):
        return self.score(X_rows)
