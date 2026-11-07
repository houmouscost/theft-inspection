"""Fitur per-konsumen, dipisah eksplisit menjadi SHAPE (invarian skala) dan LEVEL (peka skala).

Pemisahan ini adalah inti teori: pencuri yang memotong konsumsi dengan fraksi konstan
(y = (1-r) x) tidak mengubah fitur SHAPE sama sekali; hanya fitur LEVEL yang bergeser.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9
SHAPE_NAMES = ["cv", "skew", "kurt", "acf1", "acf7", "weekend_ratio", "low_day_frac",
               "rel_slope", "half_ratio", "block_cv", "q10_ratio", "q90_ratio",
               "max_ratio", "min_ratio"]
LEVEL_NAMES = ["log_mean", "log_median", "log_std"]
MISS_NAMES = ["miss_frac"]
ALL_NAMES = SHAPE_NAMES + LEVEL_NAMES + MISS_NAMES


def compute_features(X: np.ndarray, miss_frac: np.ndarray | None = None) -> np.ndarray:
    """X: (n,T) -> F: (n, len(ALL_NAMES)) float32. Urutan kolom = ALL_NAMES."""
    X = np.asarray(X, dtype=np.float32)
    n, T = X.shape
    m = X.mean(1) + EPS
    s = X.std(1)
    z = (X - m[:, None]) / (s[:, None] + 1e-4 * m[:, None])   # eps relatif -> invarian skala juga untuk deret hampir konstan

    cv = s / m
    skew = (z ** 3).mean(1)
    kurt = (z ** 4).mean(1)
    acf1 = (z[:, 1:] * z[:, :-1]).mean(1)
    lag = 7 if T > 14 else 1
    acf7 = (z[:, lag:] * z[:, :-lag]).mean(1)
    dow = np.arange(T) % 7
    wk = X[:, dow >= 5].mean(1) / (X[:, dow < 5].mean(1) + EPS)
    low_day = (X < 0.1 * m[:, None]).mean(1)
    t = np.arange(T, dtype=np.float32) - (T - 1) / 2
    rel_slope = (X * t).sum(1) / (t ** 2).sum() / m * T
    half = X[:, T // 2:].mean(1) / (X[:, : T // 2].mean(1) + EPS)
    blk = max(T // 12, 7)
    nb = max(T // blk, 2)
    blocks = X[:, : nb * blk].reshape(n, nb, blk).mean(2)
    block_cv = blocks.std(1) / (blocks.mean(1) + EPS)
    q = np.quantile(X, [0.1, 0.9], axis=1)
    q10, q90 = q[0] / m, q[1] / m
    mx, mn = X.max(1) / m, X.min(1) / m

    lm = np.log(m)
    lmed = np.log(np.median(X, axis=1) + EPS)
    lstd = np.log(s + EPS)
    mf = np.zeros(n, dtype=np.float32) if miss_frac is None else np.asarray(miss_frac, dtype=np.float32)

    F = np.stack([cv, skew, kurt, acf1, acf7, wk, low_day, rel_slope, half, block_cv,
                  q10, q90, mx, mn, lm, lmed, lstd, mf], axis=1).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)


def columns(use_level: bool, use_missing: bool) -> list[int]:
    cols = list(range(len(SHAPE_NAMES)))
    if use_level:
        cols += [ALL_NAMES.index(c) for c in LEVEL_NAMES]
    if use_missing:
        cols += [ALL_NAMES.index(c) for c in MISS_NAMES]
    return cols


def shape_matrix(F: np.ndarray) -> np.ndarray:
    return F[:, : len(SHAPE_NAMES)]
