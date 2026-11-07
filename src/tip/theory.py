"""Besaran teoretis yang dapat dihitung langsung dari data.

Prop. 1 (batas mimikri): bila pencuri melapor y = (1-r)x dengan SHAPE identik dengan konsumen jujur,
akurasi seimbang detektor mana pun <= (1 + TV(g, h_r)) / 2, dengan g = distribusi log-level jujur,
h_r = distribusi log-level pencuri terlapor = h digeser log(1-r).
Prop. 2 (kokoh-konsisten): kebijakan P(eps) menjamin p_i >= eps*b untuk semua i, dan
    recovered(P_eps) >= (1-eps) * recovered(P_0)  bila pencuri non-adaptif.
Prop. 3 (deterensi, Becker 1968): pencuri risk-neutral tidak mencuri bila p >= 1/(1+phi).
"""
from __future__ import annotations

import numpy as np


def mimicry_tv(log_level_honest: np.ndarray, log_level_thief_true: np.ndarray, r: float,
               bins: int = 60) -> float:
    shift = np.log(max(1.0 - r, 1e-9))
    h = np.asarray(log_level_thief_true) + shift
    g = np.asarray(log_level_honest)
    lo = min(g.min(), h.min()) - 1e-6
    hi = max(g.max(), h.max()) + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    pg, _ = np.histogram(g, bins=edges)
    ph, _ = np.histogram(h, bins=edges)
    return 0.5 * float(np.abs(pg / pg.sum() - ph / ph.sum()).sum())


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Sup_t |F_a(t) - F_b(t)|; balanced accuracy optimal = (1 + KS)/2."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    fa = np.searchsorted(a, allv, side="right") / len(a)
    fb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def deterrence_prob(phi: float) -> float:
    return 1.0 / (1.0 + phi)


def min_eps_for_deterrence(b: float, phi: float) -> float:
    """eps minimum agar lantai acak eps*b saja sudah menderek pencuri (>1 => mustahil)."""
    return deterrence_prob(phi) / b
