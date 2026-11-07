"""Generator data sintetis (untuk uji asap & validasi teori; BUKAN untuk klaim empiris)."""
from __future__ import annotations

import numpy as np


def make_synthetic(n: int, T: int, theft_rate: float, missing_rate: float = 0.02,
                   missing_shortcut: bool = False, seed: int = 0):
    """Return X_raw (n,T) dengan NaN, y (n,) -- 1 = pencuri.

    Beban jujur: level lognormal x musiman x pola akhir pekan x noise lognormal.
    Empat tipe pencurian (mirip taksonomi Jokar et al.): potong konstan dari onset,
    nol pada jendela, potong acak per hari, dan plafon di sebagian mean.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    level = rng.lognormal(mean=2.5, sigma=0.8, size=n)
    season = 1 + 0.25 * np.sin(2 * np.pi * t / 365 + rng.uniform(0, 2 * np.pi, (n, 1)))
    weekend = ((t % 7) >= 5).astype(float)
    weekly = 1 + rng.uniform(0, 0.3, (n, 1)) * (weekend - 0.2)
    sigma = rng.uniform(0.15, 0.5, (n, 1))
    noise = rng.lognormal(0.0, 1.0, (n, T)) ** sigma
    X = level[:, None] * season * weekly * noise

    y = np.zeros(n, dtype=np.int8)
    thieves = rng.choice(n, size=int(round(theft_rate * n)), replace=False)
    y[thieves] = 1
    for k, i in enumerate(thieves):
        typ = k % 4
        r = rng.uniform(0.2, 0.8)
        if typ == 0:                                  # potong konstan sejak onset
            t0 = int(rng.uniform(0, 0.6) * T)
            X[i, t0:] *= (1 - r)
        elif typ == 1:                                # nol pada beberapa jendela
            for _ in range(rng.integers(2, 5)):
                s = int(rng.uniform(0, 0.9) * T)
                X[i, s:s + int(rng.integers(5, max(6, T // 8)))] = 0.0
        elif typ == 2:                                # potong acak per hari
            X[i] *= rng.uniform(1 - r, 1.0, T)
        else:                                         # plafon
            X[i] = np.minimum(X[i], (1 - r) * X[i].mean() * 1.2)

    miss_p = np.where(y == 1, missing_rate * (3 if missing_shortcut else 1), missing_rate)
    mask = rng.random((n, T)) < miss_p[:, None]
    X = X.astype(np.float32)
    X[mask] = np.nan
    return X, y
