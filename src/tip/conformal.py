"""Sertifikat tingkat-lolos (miss) dengan jaminan sampel-hingga.

Dua objek, keduanya DISTRIBUTION-FREE tetapi hanya sah bila kalibrasi dan deployment
dapat dipertukarkan (exchangeable) -- yaitu bila kelas adversary deployment tidak lebih kuat
daripada kelas yang dipakai saat kalibrasi. Percobaan E5 mengukur persis kegagalan itu.

* conformal_upper : kuantil split-conformal => miss-prob pencuri BARU <= q w.p. >= 1-alpha
* hoeffding_upper : batas atas rata-rata miss-prob populasi w.p. >= 1-alpha
"""
from __future__ import annotations

import math

import numpy as np


def conformal_upper(vals: np.ndarray, alpha: float) -> float:
    v = np.sort(np.asarray(vals))
    n = len(v)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(v[k - 1]) if k <= n else 1.0


def hoeffding_upper(vals: np.ndarray, alpha: float, lo: float = 0.0, hi: float = 1.0) -> float:
    n = len(vals)
    return float(np.mean(vals) + (hi - lo) * math.sqrt(math.log(1.0 / alpha) / (2 * n)))


def coverage(pool_cal: np.ndarray, pool_dep: np.ndarray, alpha: float, n_cal: int,
             trials: int, rng: np.random.Generator) -> dict:
    n_cal = min(n_cal, len(pool_cal))
    cov_c, cov_h, ub_c, ub_h = [], [], [], []
    mean_dep = float(np.mean(pool_dep))
    for _ in range(trials):
        s = rng.choice(pool_cal, size=n_cal, replace=False)
        q = conformal_upper(s, alpha)
        u = hoeffding_upper(s, alpha)
        cov_c.append(float(np.mean(pool_dep <= q)))     # peluang pencuri baru tercakup
        cov_h.append(float(mean_dep <= u))
        ub_c.append(q)
        ub_h.append(u)
    return {"cov_conformal": float(np.mean(cov_c)), "cov_hoeffding": float(np.mean(cov_h)),
            "ub_conformal": float(np.mean(ub_c)), "ub_hoeffding": float(np.mean(ub_h)),
            "mean_miss_dep": mean_dep, "n_cal": n_cal, "target": 1 - alpha}
