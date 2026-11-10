"""Ekstensi 1: kebijakan inspeksi sadar-strategi (Bayesian Stackelberg, MILP).

Utilitas tidak lagi memeringkat menurut skor (keluarga P(eps, tau) monoton), tetapi memilih
probabilitas inspeksi p_k per SEL pengamatan k. Sel ditentukan oleh skor detektor dan/atau
log-level terlapor, dengan batas = kuantil konsumen JUJUR pada split kalibrasi, sehingga p seorang
konsumen hanya bergantung pada laporannya sendiri (tidak ada eksternalitas antarpencuri selain
anggaran). Kebijakan boleh tidak monoton: mis. menginspeksi konsumen yang melapor "terlalu kecil".

Pencuri i memilih aksi j (jenis serangan x kedalaman r; j=0 = jujur) yang memaksimalkan
    U_ij = G_ij * (1 - (1 + phi) * p_{k(i,j)}).
Pemimpin (utilitas) meminimalkan energi curian yang lolos  sum_i G_i,j* (1 - p_{k(i,j*)})
dengan anggaran  sum_honest p + sum_thief p <= b * N  (pencuri ikut memakai anggaran).

Formulasi MILP gaya DOBSS (Paruchuri dkk. 2008) dengan tie-breaking menguntungkan pemimpin
(strong Stackelberg); suku bilinear p*a dilinearkan (McCormick, eksak karena a biner).
Pemangkasan eksak: dalam satu sel p sama, jadi pencuri hanya memilih aksi dengan G terbesar di sel itu.

Kebijakan DILATIH pada pencuri injeksi dari split KALIBRASI dan DIEVALUASI pada pencuri split UJI
(tie-breaking pesimis: di antara aksi ber-U maksimal, pencuri memilih yang paling merugikan utilitas).
"Oracle" = MILP yang dilatih langsung pada pencuri uji (batas atas kelas kebijakan ini).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


# --------------------------------------------------------------------------- sel pengamatan
@dataclass
class Cells:
    mode: str                  # score | level | joint
    edges_s: np.ndarray
    edges_l: np.ndarray

    @property
    def K(self) -> int:
        ns, nl = len(self.edges_s) + 1, len(self.edges_l) + 1
        return {"score": ns, "level": nl, "joint": ns * nl}[self.mode]

    @property
    def n_l(self) -> int:
        return len(self.edges_l) + 1

    def score_bin(self, k: int) -> int:
        return k if self.mode == "score" else (k // self.n_l if self.mode == "joint" else -1)

    def of(self, s, l):
        ks = np.searchsorted(self.edges_s, s, side="right")
        kl = np.searchsorted(self.edges_l, l, side="right")
        if self.mode == "score":
            return ks
        if self.mode == "level":
            return kl
        return ks * (len(self.edges_l) + 1) + kl


def make_cells(hs: np.ndarray, hl: np.ndarray, mode: str, n_s: int, n_l: int, extra_q=()) -> Cells:
    """Batas = kuantil konsumen jujur. `extra_q` menambah batas skor di kuantil atas (mis. 1-b) supaya
    kelas kebijakan memuat top-k pada anggaran b (pembanding harus termasuk dalam kelas)."""
    lv = np.unique(np.r_[np.linspace(0, 1, n_s + 1)[1:-1], list(extra_q)])
    qs = np.unique(np.quantile(hs, lv))
    ql = np.unique(np.quantile(hl, np.linspace(0, 1, n_l + 1)[1:-1]))
    return Cells(mode, qs, ql)


# --------------------------------------------------------------------------- MILP
def _prune(Kth: np.ndarray, G: np.ndarray):
    """Per pencuri: aksi jujur (kolom 0) + aksi ber-G maksimum di tiap sel lain. Return list (i, k, g)."""
    out = []
    for i in range(len(G)):
        out.append((i, int(Kth[i, 0]), 0.0))               # jujur: G = 0, tetap memakai anggaran
        best = {}
        for j in range(1, G.shape[1]):
            k, g = int(Kth[i, j]), float(G[i, j])
            if g > 0 and g > best.get(k, -1.0):
                best[k] = g
        out += [(i, k, g) for k, g in best.items()]
    return out


def fit_milp(Kth: np.ndarray, G: np.ndarray, Kh: np.ndarray, K: int, b: float, phi: float,
             time_limit: float = 120.0, gap: float = 0.01):
    """Return (p[K], info). Kth: (n, J) sel tiap aksi pencuri; G: (n, J) curian; Kh: sel konsumen jujur."""
    t0 = time.time()
    n = len(G)
    scale = float(G.max()) if G.max() > 0 else 1.0
    acts = _prune(Kth, G / scale)
    A = len(acts)
    ai = np.array([a[0] for a in acts]); ak = np.array([a[1] for a in acts]); ag = np.array([a[2] for a in acts])
    c_ = 1.0 + phi
    gmax = np.zeros(n); np.maximum.at(gmax, ai, ag)
    M = c_ * gmax[ai] + 1e-6
    nh = np.bincount(Kh, minlength=K).astype(float)
    # indeks variabel: p | a | z | v
    P0, A0, Z0, V0 = 0, K, K + A, K + 2 * A
    nv = K + 2 * A + n
    ar = np.arange(A)
    rows, lo, hi = [], [], []

    def block(r, c, v, nr, l, h):
        rows.append(coo_matrix((v, (r, c)), shape=(nr, nv)))
        lo.append(l); hi.append(h)

    # (1) sum_j a_ij = 1
    block(ai, A0 + ar, np.ones(A), n, np.ones(n), np.ones(n))
    # (2) v_i + c g p_k >= g
    block(np.r_[ar, ar], np.r_[V0 + ai, P0 + ak], np.r_[np.ones(A), c_ * ag], A, ag, np.full(A, np.inf))
    # (3) v_i + c g p_k + M a <= g + M
    block(np.r_[ar, ar, ar], np.r_[V0 + ai, P0 + ak, A0 + ar], np.r_[np.ones(A), c_ * ag, M], A,
          np.full(A, -np.inf), ag + M)
    # (4) z - a <= 0 ; (5) z - p_k <= 0 ; (6) z - p_k - a >= -1
    block(np.r_[ar, ar], np.r_[Z0 + ar, A0 + ar], np.r_[np.ones(A), -np.ones(A)], A, np.full(A, -np.inf), np.zeros(A))
    block(np.r_[ar, ar], np.r_[Z0 + ar, P0 + ak], np.r_[np.ones(A), -np.ones(A)], A, np.full(A, -np.inf), np.zeros(A))
    block(np.r_[ar, ar, ar], np.r_[Z0 + ar, P0 + ak, A0 + ar], np.r_[np.ones(A), -np.ones(A), -np.ones(A)], A,
          np.full(A, -1.0), np.full(A, np.inf))
    # (7) anggaran
    block(np.zeros(K + A, dtype=int), np.r_[P0 + np.arange(K), Z0 + ar], np.r_[nh, np.ones(A)], 1,
          np.array([-np.inf]), np.array([b * (len(Kh) + n)]))
    cons = LinearConstraint(vstack(rows).tocsr(), np.concatenate(lo), np.concatenate(hi))
    cost = np.zeros(nv); cost[A0:A0 + A] = ag; cost[Z0:Z0 + A] = -ag
    integ = np.zeros(nv); integ[A0:A0 + A] = 1
    lb = np.zeros(nv); ub = np.ones(nv)
    lb[V0:] = -np.inf; ub[V0:] = np.inf
    res = milp(cost, constraints=cons, integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})
    db = getattr(res, "mip_dual_bound", None)
    info = {"status": int(res.status), "n_actions": A, "solve_s": time.time() - t0,
            "mip_gap": float(getattr(res, "mip_gap", np.nan) or np.nan),
            # batas bawah curian lolos (unit asli) untuk SEMUA kebijakan sel ini pada data latih (SSE)
            "dual_bound": float(db) * scale if db is not None and np.isfinite(db) else np.nan}
    if res.x is None:
        return np.full(K, b), info                         # gagal -> acak seragam
    return np.clip(res.x[:K], 0, 1), info


# --------------------------------------------------------------------------- evaluasi
def best_response(p: np.ndarray, Kth: np.ndarray, G: np.ndarray, phi: float):
    """Tie-breaking pesimis: di antara aksi U-maksimal pilih yang meloloskan curian terbanyak."""
    P = p[Kth]
    U = G * (1 - (1 + phi) * P)
    tol = 1e-9 * max(float(G.max()), 1.0)
    cand = U >= U.max(1, keepdims=True) - tol
    loss = np.where(cand, G * (1 - P), -np.inf)
    return loss.argmax(1)


def evaluate(p: np.ndarray, Kth: np.ndarray, G: np.ndarray, Kh: np.ndarray, b: float, phi: float,
             total_load: float, max_iter: int = 20) -> dict:
    """Terapkan kebijakan ke populasi uji. Bila anggaran terlampaui (distribusi sel uji berbeda),
    skala turun p dan hitung ulang best-response sampai anggaran terpenuhi."""
    p = p.copy()
    N = len(Kh) + len(G)
    ar = np.arange(len(G))
    for _ in range(max_iter):
        idx = best_response(p, Kth, G, phi)
        used = p[Kh].sum() + p[Kth[ar, idx]].sum()
        if used <= b * N * (1 + 1e-6):
            break
        p *= b * N / used
    pt = p[Kth[ar, idx]]
    g = G[ar, idx]
    act = idx > 0
    return {"ntl_undetected": float(((1 - pt) * g).sum() / total_load),
            "ntl_recovered": float((pt * g).sum() / total_load),
            "stolen_total": float(g.sum() / total_load),
            "deterred": float(1 - act.mean()),
            "recall": float(pt[act].sum() / max(act.sum(), 1)),
            "budget_used": float(used / N), "idx": idx}


# --------------------------------------------------------------------------- pencarian lokal
def _objective(p, Kth, G, nh, c):
    """Curian lolos (pesimis) dan anggaran terpakai untuk kebijakan p."""
    P = p[Kth]
    U = G * (1.0 - c * P)
    tol = 1e-9 * max(float(G.max()), 1.0)
    loss = np.where(U >= U.max(1, keepdims=True) - tol, G * (1.0 - P), -np.inf)
    j = loss.argmax(1)
    ar = np.arange(len(G))
    return float(loss[ar, j].sum()), float(nh @ p + P[ar, j].sum())


def _rebalance(p, locked, Kth, G, nh, c, B, iters=20):
    """Sel tak terkunci memakai nilai seragam u; cari u terbesar dengan anggaran <= B (bisection)."""
    q = p.copy()
    free = ~locked
    q[free] = 0.0
    f0, used0 = _objective(q, Kth, G, nh, c)
    if used0 > B * (1 + 1e-9):
        return None, np.inf
    if not free.any():
        return q, f0
    lo, hi = 0.0, 1.0
    q[free] = hi
    if _objective(q, Kth, G, nh, c)[1] <= B:
        return q, _objective(q, Kth, G, nh, c)[0]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q[free] = mid
        if _objective(q, Kth, G, nh, c)[1] <= B:
            lo = mid
        else:
            hi = mid
    q[free] = lo
    return q, _objective(q, Kth, G, nh, c)[0]


def fit_local(Kth: np.ndarray, G: np.ndarray, Kh: np.ndarray, K: int, b: float, phi: float,
              max_iter: int = 40, top_cells=None):
    """Pencarian lokal: mulai dari acak seragam (p = b), tiap langkah tetapkan satu sel ke nilai kandidat
    (termasuk ambang deterensi 1/(1+phi)) dan seimbangkan ulang sisa anggaran secara seragam.
    Hanya menerima perbaikan -> tak pernah lebih buruk daripada acak pada data latih."""
    t0 = time.time()
    c = 1.0 + phi
    nh = np.bincount(Kh, minlength=K).astype(float)
    B = b * (len(Kh) + len(G))
    det_level = min(1.0, (1.0 / c) * (1 + 1e-3))
    vals = sorted({0.0, 0.5 * b, b, 2 * b, 4 * b, 8 * b, 0.5 * det_level, det_level, 1.0})
    locked = np.zeros(K, dtype=bool)
    p, f = _rebalance(np.full(K, b), locked, Kth, G, nh, c, B)
    f_start = f
    start = "uniform"
    if top_cells is not None and len(top_cells):   # titik awal kedua: mirip top-k (sel skor teratas p=1)
        lk = np.zeros(K, dtype=bool); lk[top_cells] = True
        q = np.zeros(K); q[top_cells] = 1.0
        q, fq = _rebalance(q, lk, Kth, G, nh, c, B)
        if q is None:   # sel teratas melebihi anggaran -> skala turun
            q = np.zeros(K); q[top_cells] = 1.0
            used = nh @ q
            q[top_cells] = min(1.0, B / max(used, 1e-12)) * 0.999
            q, fq = _rebalance(q, lk, Kth, G, nh, c, B)
        if q is not None and fq < f:
            p, f, locked, start = q, fq, lk, "topk"
    it = 0
    for it in range(max_iter):
        best = (f, None, None)
        for k in range(K):
            lk = locked.copy(); lk[k] = True
            for v in vals:
                q = p.copy(); q[k] = v
                q, fq = _rebalance(q, lk, Kth, G, nh, c, B)
                if q is not None and fq < best[0] - 1e-9 * max(abs(f), 1.0):
                    best = (fq, q, lk)
        if best[1] is None:
            break
        f, p, locked = best
    return np.clip(p, 0, 1), {"status": 0, "solve_s": time.time() - t0, "iters": it + 1, "start": start,
                              "train_obj": f, "train_obj_uniform": f_start, "n_locked": int(locked.sum())}
