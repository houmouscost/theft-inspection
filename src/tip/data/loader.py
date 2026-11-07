"""Muat -> bersihkan -> pecah. Satu pintu masuk untuk SGCC maupun sintetis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .synthetic import make_synthetic


def read_sgcc_csv(path: str | Path):
    """Baca data.csv SGCC (kolom CONS_NO, FLAG, lalu ~1035 kolom tanggal)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ada. Jalankan `bash scripts/download_sgcc.sh` atau lihat README bagian Data."
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    flag_col = next(c for c in df.columns if c.upper() == "FLAG")
    id_cols = [c for c in df.columns if c.upper() in ("CONS_NO", "FLAG")]
    date_cols = [c for c in df.columns if c not in id_cols]
    dates = pd.to_datetime(pd.Series(date_cols), errors="coerce", format="mixed")
    if dates.isna().any():  # fallback: pertahankan urutan kolom asli
        order = list(range(len(date_cols)))
    else:
        order = list(np.argsort(dates.values, kind="stable"))
    date_cols = [date_cols[i] for i in order]
    X = df[date_cols].to_numpy(dtype=np.float32)
    y = df[flag_col].to_numpy(dtype=np.int8)
    keep = ~np.all(np.isnan(X), axis=1)
    return X[keep], y[keep]


def clean_series(X_raw: np.ndarray, clip_sigma: float = 3.0):
    """Nilai negatif -> NaN, interpolasi linier per konsumen, winsorisasi mean+k*std.

    Return (X_clean, miss_frac). miss_frac dihitung SEBELUM imputasi dan disimpan
    terpisah supaya pengaruh 'shortcut missingness' bisa diablasi.
    """
    X = X_raw.astype(np.float32).copy()
    X[X < 0] = np.nan
    miss_frac = np.isnan(X).mean(axis=1).astype(np.float32)
    df = pd.DataFrame(X)
    df = df.interpolate(axis=1, limit_direction="both")
    X = df.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    if clip_sigma and clip_sigma > 0:
        hi = X.mean(axis=1, keepdims=True) + clip_sigma * X.std(axis=1, keepdims=True)
        X = np.minimum(X, hi)
    return X, miss_frac


def load_dataset(cfg: dict, seed: int = 0):
    """Return X_clean, y, miss_frac."""
    d = cfg["data"]
    if d["source"] == "sgcc":
        X_raw, y = read_sgcc_csv(d["sgcc_path"])
    elif d["source"] == "synthetic":
        s = d["synthetic"]
        X_raw, y = make_synthetic(s["n_consumers"], s["n_days"], s["theft_rate"],
                                  s.get("missing_rate", 0.02), s.get("missing_shortcut", False), seed)
    else:
        raise ValueError(f"data.source tidak dikenal: {d['source']}")
    X, miss = clean_series(X_raw, d.get("clip_sigma", 3.0))
    return X, y, miss


def split_indices(y: np.ndarray, split: dict, seed: int):
    """Pecah per-KONSUMEN, stratifikasi label. (SGCC tidak punya timestamp pencurian,
    jadi split temporal tidak mungkin -- ini keterbatasan yang dilaporkan di paper.)"""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=split["train"], stratify=y, random_state=seed)
    frac_cal = split["calib"] / (split["calib"] + split["test"])
    cal, te = train_test_split(rest, train_size=frac_cal, stratify=y[rest], random_state=seed)
    return tr, cal, te
