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
             trials: int, rng: np.random.Generator, tv_adj: float = 0.0) -> dict:
    n_cal = min(n_cal, len(pool_cal))
    cov_c, cov_h, cov_ha, ub_c, ub_h = [], [], [], [], []
    mean_dep = float(np.mean(pool_dep))
    for _ in range(trials):
        s = rng.choice(pool_cal, size=n_cal, replace=False)
        q = conformal_upper(s, alpha)
        u = hoeffding_upper(s, alpha)
        cov_c.append(float(np.mean(pool_dep <= q)))     # peluang pencuri baru tercakup
        cov_h.append(float(mean_dep <= u))
        cov_ha.append(float(mean_dep <= min(1.0, u + tv_adj)))   # batas Hoeffding digeser sebesar TV (Barber dkk.)
        ub_c.append(q)
        ub_h.append(u)
    return {"cov_conformal": float(np.mean(cov_c)), "cov_hoeffding": float(np.mean(cov_h)),
            "ub_conformal": float(np.mean(ub_c)), "ub_hoeffding": float(np.mean(ub_h)),
            "cov_hoeffding_adj": float(np.mean(cov_ha)), "ub_hoeffding_adj": float(min(1.0, np.mean(ub_h) + tv_adj)),
            "mean_miss_dep": mean_dep, "n_cal": n_cal, "target": 1 - alpha}


def shift_diagnostics(pc: np.ndarray, pdp: np.ndarray, bins: int = 20) -> dict:
    """Jarak antara populasi miss-probability kalibrasi dan deployment (argumen Barber dkk. 2023, kasus ambang).

    Kejadian cakupan konformal {m <= q} adalah kejadian AMBANG, sehingga celah cakupan <= KS(pc, pdp) <=
    TV(pc, pdp). Karena m in [0,1], |rata-rata deployment - rata-rata kalibrasi| = |int (F_cal - F_dep)| <= KS
    (eksak pada pool empiris), jadi batas Hoeffding sah terhadap deployment bila digeser sebesar KS.
    `tv_pools_bin` = TV pada `bins` kotak sama lebar (batas bawah TV; hanya diagnostik).
    """
    a, b = np.sort(np.asarray(pc)), np.sort(np.asarray(pdp))
    allv = np.concatenate([a, b])
    ks = float(np.abs(np.searchsorted(a, allv, side="right") / len(a) - np.searchsorted(b, allv, side="right") / len(b)).max())
    ha, _ = np.histogram(a, bins=bins, range=(0, 1))
    hb, _ = np.histogram(b, bins=bins, range=(0, 1))
    return {"ks_pools": ks, "tv_pools_bin": 0.5 * float(np.abs(ha / ha.sum() - hb / hb.sum()).sum()),
            "mean_shift": float(b.mean() - a.mean())}
