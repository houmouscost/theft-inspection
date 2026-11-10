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
# Fitur PERUBAHAN dalam deret sendiri (ekstensi 2): invarian skala, peka terhadap onset pencurian.
CHANGE_NAMES = ["chg_max_drop", "chg_tstat", "chg_pos", "chg_last_drop", "chg_min_block"]
ALL_NAMES = SHAPE_NAMES + LEVEL_NAMES + MISS_NAMES + CHANGE_NAMES


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

    C = change_features(X)
    F = np.concatenate([np.stack([cv, skew, kurt, acf1, acf7, wk, low_day, rel_slope, half, block_cv,
                                  q10, q90, mx, mn, lm, lmed, lstd, mf], axis=1), C], axis=1).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=1e6, neginf=-1e6)


def change_features(X: np.ndarray, blk: int = 30, min_seg: int = 3) -> np.ndarray:
    """Statistik titik-ubah pada log rerata bulanan (blok `blk` hari) deret konsumen sendiri.

    Invarian skala: lantai log relatif terhadap rerata konsumen, sehingga y = c*x tidak mengubah fitur.
    * chg_max_drop  : selisih maksimum mean(log sebelum) - mean(log sesudah) atas titik belah k
    * chg_tstat     : versi ternormalisasi (gaya CUSUM/t) dari selisih itu, maksimum atas k
    * chg_pos       : posisi relatif titik belah terbaik (0..1)
    * chg_last_drop : log rerata 6 blok terakhir - log rerata blok sebelumnya
    * chg_min_block : blok terendah relatif terhadap rerata log
    """
    X = np.asarray(X, dtype=np.float32)
    n, T = X.shape
    nb = max(T // blk, 2 * min_seg + 1)
    blk = max(T // nb, 1)
    B = X[:, : nb * blk].reshape(n, nb, blk).mean(2)
    L = np.log(B + 1e-3 * X.mean(1, keepdims=True) + EPS)
    cs = np.cumsum(L, 1)
    tot = cs[:, -1:]
    ks = np.arange(min_seg, nb - min_seg + 1)
    d = cs[:, ks - 1] / ks - (tot - cs[:, ks - 1]) / (nb - ks)
    sd = L.std(1, keepdims=True) + 0.01            # lantai (unit log) -> deret datar tidak memperbesar derau
    t = np.round(d / sd * np.sqrt(ks * (nb - ks) / nb), 5)   # dibulatkan: seri datar -> argmax deterministik
    j = t.argmax(1)
    last = L[:, -6:].mean(1) - L[:, :-6].mean(1)
    mnb = (L - L.mean(1, keepdims=True)).min(1)
    tmax = t.max(1)
    pos = np.where(tmax > 0.5, ks[j] / nb, 0.5)       # posisi tak terdefinisi bila tak ada perubahan
    return np.stack([d.max(1), tmax, pos, last, mnb], axis=1).astype(np.float32)


def columns(use_level: bool, use_missing: bool, use_change: bool = False) -> list[int]:
    cols = list(range(len(SHAPE_NAMES)))
    if use_level:
        cols += [ALL_NAMES.index(c) for c in LEVEL_NAMES]
    if use_missing:
        cols += [ALL_NAMES.index(c) for c in MISS_NAMES]
    if use_change:
        cols += [ALL_NAMES.index(c) for c in CHANGE_NAMES]
    return cols


def shape_matrix(F: np.ndarray) -> np.ndarray:
    return F[:, : len(SHAPE_NAMES)]
