"""Permainan Stackelberg: utilitas memilih kebijakan, pencuri menjawab terbaik.

Utilitas pencuri i memilih tingkat potongan r dari grid:
    U_i(r) = G_i(r) * (1 - p_i(r) * (1 + phi))
G_i(r) = kWh dicuri; p_i(r) = prob. diinspeksi bila melapor y_r; tertangkap -> curian
disita dan didenda phi * G_i(r). r = 0 (jujur) memberi U = 0.

Karena skor detektor TIDAK bergantung pada kebijakan, matriks skor S[i, j] = s(report(r_j))
dihitung sekali; kebijakan hanya mengubah p = policy.prob(S).
"""
from __future__ import annotations

import numpy as np

from .policies import fit_policy


def solve(S: np.ndarray, G: np.ndarray, honest_scores: np.ndarray, b: float, eps: float,
          tau: float, phi: float, iters: int = 8, rng: np.random.Generator | None = None,
          fixed_idx: int | None = None, init_idx: int | None = None):
    """Return dict(idx, p_th, p_h, converged).

    adaptif   : best-response asinkron (separuh pencuri memperbarui per iterasi) -> meredam siklus
    fixed_idx : pencuri non-adaptif, semua memakai kolom r tetap (uji konsistensi)
    """
    rng = rng or np.random.default_rng(0)
    n_th, R = S.shape
    ar = np.arange(n_th)
    if fixed_idx is not None:
        idx = np.full(n_th, fixed_idx)
        converged = True
    else:
        idx = np.full(n_th, init_idx if init_idx is not None else R // 2)
        converged = False
        for it in range(iters):
            pop = np.concatenate([honest_scores, S[ar, idx]])
            pol = fit_policy(pop, b, eps, tau)
            P = pol.prob(S)
            U = G * (1.0 - P * (1.0 + phi))
            best = U.argmax(1)            # r=0 punya U=0 -> menang saat semua U<=0
            upd = np.ones(n_th, dtype=bool) if it == 0 else rng.random(n_th) < 0.5   # iterasi 1 sinkron
            new = np.where(upd, best, idx)
            if (new == idx).all() and (best == idx).all():
                converged = True
                break
            idx = new
        else:
            converged = bool((idx == best).mean() > 0.98)
    pop = np.concatenate([honest_scores, S[ar, idx]])
    pol = fit_policy(pop, b, eps, tau)
    return {"idx": idx, "p_th": pol.prob(S[ar, idx]), "p_h": pol.prob(honest_scores),
            "converged": converged, "policy": pol}


def outcome_metrics(sol: dict, G: np.ndarray, total_load: float) -> dict:
    ar = np.arange(len(sol["idx"]))
    g = G[ar, sol["idx"]]
    active = sol["idx"] > 0
    p, ph = sol["p_th"], sol["p_h"]
    undetected = float(((1 - p) * g).sum())
    recovered = float((p * g).sum())
    insp_th = float(p[active].sum())
    insp_all = float(p.sum() + ph.sum())
    return {
        "ntl_undetected": undetected / total_load,   # fraksi beban sistem yang hilang & lolos
        "ntl_recovered": recovered / total_load,
        "stolen_total": float(g.sum()) / total_load,
        "deterred": float(1 - active.mean()),
        "recall": insp_th / max(active.sum(), 1),     # ekspektasi fraksi pencuri aktif yang diinspeksi
        "precision": insp_th / max(insp_all, 1e-12),
        "converged": bool(sol["converged"]),
    }
