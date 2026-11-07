"""Keluarga kebijakan inspeksi P(eps, tau) dengan probabilitas inklusi per konsumen.

    p_i = eps * b + (1 - eps) * q_i,     q_i = sigmoid((s_i - t) / tau),   sum_i q_i = b * N

* eps = 0, tau -> 0 : top-k murni (hanya saran ML)           -> konsisten, tapi bisa dikelabui
* eps = 1           : inspeksi acak seragam                   -> kokoh, tapi mengabaikan ML
* eps in (0,1)      : lantai acak  => p_i >= eps * b untuk SIAPA PUN, apa pun strateginya
Anggaran terpenuhi persis dalam ekspektasi: sum_i p_i = b * N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

TAU_HARD = 1e-3


@dataclass
class Policy:
    b: float
    eps: float
    tau: float
    t: float

    def prob(self, scores: np.ndarray) -> np.ndarray:
        tau = max(self.tau, TAU_HARD)
        q = expit((np.asarray(scores) - self.t) / tau)
        return self.eps * self.b + (1.0 - self.eps) * q


def fit_policy(scores: np.ndarray, b: float, eps: float, tau: float) -> Policy:
    """Cari ambang t sehingga sum q_i = b * N (bisection; monoton menurun di t)."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    k = b * n
    tau_e = max(tau, TAU_HARD)
    lo, hi = scores.min() - 20 * tau_e - 1.0, scores.max() + 20 * tau_e + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if expit((scores - mid) / tau_e).sum() > k:
            lo = mid
        else:
            hi = mid
    return Policy(b=b, eps=eps, tau=tau, t=0.5 * (lo + hi))
