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


def _silverman(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sd = x.std(ddof=1)
    return 0.9 * min(sd, iqr / 1.34 if iqr > 0 else sd) * len(x) ** (-0.2)


def _kde_grid(x: np.ndarray, edges: np.ndarray, bw: float) -> np.ndarray:
    """Massa KDE Gaussian per kotak halus (histogram halus + konvolusi FFT); jumlah = 1."""
    from scipy.signal import fftconvolve
    h, _ = np.histogram(x, bins=edges)
    d = edges[1] - edges[0]
    half = int(np.ceil(5 * bw / d))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) * d / bw) ** 2)
    out = fftconvolve(h.astype(float), k / k.sum(), mode="same")
    out = np.clip(out, 0.0, None)
    return out / out.sum()


def mimicry_tv_kde(log_level_honest: np.ndarray, log_level_thief_true: np.ndarray, r: float,
                   bw_mult: float = 1.0, n_bins: int = 8192) -> float:
    """TV(g, h_r) dengan KDE Gaussian untuk g dan h (bandwidth Silverman pada g, dikali bw_mult).

    Estimator histogram `mimicry_tv` membandingkan dua sampel berhingga dan bias ke atas (sampel yang sama
    pun memberi TV > 0 ~ sqrt(bins/n)); KDE pada sampel populasi besar menghilangkan bias itu.
    """
    shift = np.log(max(1.0 - r, 1e-9))
    g = np.asarray(log_level_honest, dtype=float)
    h = np.asarray(log_level_thief_true, dtype=float) + shift
    bw = _silverman(g) * bw_mult
    lo, hi = min(g.min(), h.min()) - 5 * bw, max(g.max(), h.max()) + 5 * bw
    edges = np.linspace(lo, hi, n_bins + 1)
    return 0.5 * float(np.abs(_kde_grid(g, edges, bw) - _kde_grid(h, edges, bw)).sum())


def ba_crossfit(hs: np.ndarray, ts: np.ndarray, rng: np.random.Generator, n_thr: int = 400) -> tuple[float, float]:
    """Akurasi seimbang uji-ambang dengan ambang & arah dipilih pada separuh data dan dinilai pada separuh lain
    (dua arah, cross-fitting). Menghilangkan optimisme maksimum-atas-ambang dari (1+KS)/2.
    Return (estimasi, galat baku binomial)."""
    hs, ts = np.asarray(hs), np.asarray(ts)
    ih, it = rng.permutation(len(hs)), rng.permutation(len(ts))
    fh, ft = np.array_split(ih, 2), np.array_split(it, 2)
    est, var = [], []
    for a in (0, 1):
        A_h, A_t, B_h, B_t = hs[fh[a]], ts[ft[a]], hs[fh[1 - a]], ts[ft[1 - a]]
        cand = np.quantile(np.concatenate([A_h, A_t]), np.linspace(0.0, 1.0, n_thr))
        tpr = (A_t[:, None] > cand[None, :]).mean(0)
        fpr = (A_h[:, None] > cand[None, :]).mean(0)
        j = tpr - fpr
        k = int(np.argmax(np.abs(j)))
        sign = 1.0 if j[k] >= 0 else -1.0
        tb, fb = float((B_t > cand[k]).mean()), float((B_h > cand[k]).mean())
        est.append(0.5 * (1 + sign * (tb - fb)))
        var.append(0.25 * (tb * (1 - tb) / len(B_t) + fb * (1 - fb) / len(B_h)))
    return float(np.mean(est)), float(np.sqrt(np.sum(var)) / 2)


def ks_perm_null(hs: np.ndarray, ts: np.ndarray, rng: np.random.Generator, n_perm: int = 100) -> float:
    """Rata-rata KS dua-sampel bila skor pencuri ~ skor jujur (hipotesis nol). Ini lantai optimisme
    dari memaksimalkan selisih ECDF di atas ambang pada sampel berhingga."""
    allv = np.concatenate([hs, ts])
    nh, nt = len(hs), len(ts)
    lab = np.r_[np.zeros(nh), np.ones(nt)]
    out = np.empty(n_perm)
    for i in range(n_perm):
        l = rng.permutation(lab)[np.argsort(allv, kind="stable")]
        out[i] = np.abs(np.cumsum(l) / nt - np.cumsum(1 - l) / nh).max()
    return float(out.mean())


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
