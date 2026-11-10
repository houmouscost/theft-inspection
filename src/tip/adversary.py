"""Kelas adversary. Semua menerima beban SEBENARNYA X_true dan mengembalikan beban TERLAPOR.

A1  skala konstan          y = (1-r) x                 (shape tak berubah)
A2  skala bertahap (ramp)  y_t = (1 - r w_t) x_t       (onset acak, naik linier)
A3  mimikri bentuk         y = (1-r) mean(x) * z_nn    z_nn = profil ternormalisasi tetangga
                           jujur terdekat di ruang fitur SHAPE -> hanya LEVEL yang tersisa
A4-L onset eksogen         y_t = (1 - r w_t) x_t,  w_t = clip((t - t0_i)/L, 0, 1)  (L=0: langkah)
                           t0_i DIBERIKAN (pencuri mulai mencuri pada waktu tertentu, bukan pilihan);
                           pencuri memilih L (lewat jenis) dan r. Ekstensi 2.
r = 0 selalu berarti melapor jujur (tidak mencuri, tidak ada risiko).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .features import compute_features, shape_matrix

KINDS = ("A1", "A2", "A3")


def is_onset(kind: str) -> bool:
    return kind.startswith("A4-")


def ramp_weights(T: int, t0: np.ndarray, L: int) -> np.ndarray:
    tt = np.arange(T)[None, :]
    if L <= 0:
        return (tt >= t0[:, None]).astype(np.float32)
    return np.clip((tt - t0[:, None]) / L, 0.0, 1.0).astype(np.float32)


class Adversary:
    def __init__(self, kind: str, X_true: np.ndarray, rng: np.random.Generator,
                 bank_X: np.ndarray | None = None, onset: np.ndarray | None = None):
        if kind not in KINDS and not is_onset(kind):
            raise ValueError(kind)
        self.kind = kind
        self.X = np.asarray(X_true, dtype=np.float32)
        n, T = self.X.shape
        self.mean = self.X.mean(1, keepdims=True)
        if is_onset(kind):
            if onset is None:
                raise ValueError("A4-* membutuhkan onset per pencuri")
            self.w = ramp_weights(T, np.asarray(onset), int(kind.split("-")[1]))
        if kind == "A2":
            t0 = rng.integers(0, max(T // 2, 1), size=n)
            ramp = max(T // 10, 1)
            tt = np.arange(T)[None, :]
            self.w = np.clip((tt - t0[:, None]) / ramp, 0.0, 1.0).astype(np.float32)
        if kind == "A3":
            if bank_X is None:
                raise ValueError("A3 membutuhkan bank_X (konsumen jujur untuk ditiru)")
            Fb = shape_matrix(compute_features(bank_X))
            Fx = shape_matrix(compute_features(self.X))
            mu, sd = Fb.mean(0), Fb.std(0) + 1e-6
            nn = NearestNeighbors(n_neighbors=1).fit((Fb - mu) / sd)
            idx = nn.kneighbors((Fx - mu) / sd, return_distance=False)[:, 0]
            zb = bank_X[idx]
            self.z = (zb / (zb.mean(1, keepdims=True) + 1e-9)).astype(np.float32)

    def report(self, r: float) -> np.ndarray:
        if r <= 0:
            return self.X
        if self.kind == "A1":
            return (1.0 - r) * self.X
        if self.kind == "A2" or is_onset(self.kind):
            return (1.0 - r * self.w) * self.X
        return ((1.0 - r) * self.mean * self.z).astype(np.float32)

    def stolen(self, r: float) -> np.ndarray:
        """kWh yang dicuri per konsumen selama jendela (= beban sebenarnya - terlapor)."""
        return (self.X - self.report(r)).sum(1)


def inject_attack(X: np.ndarray, kind: str, r: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Serangan per-baris dengan r berbeda-beda (untuk adversarial training; A1/A2 saja)."""
    n, T = X.shape
    r = np.asarray(r, dtype=np.float32)[:, None]
    if kind == "A1":
        return (1.0 - r) * X
    if kind == "A2":
        t0 = rng.integers(0, max(T // 2, 1), size=n)[:, None]
        w = np.clip((np.arange(T)[None, :] - t0) / max(T // 10, 1), 0.0, 1.0)
        return ((1.0 - r * w) * X).astype(np.float32)
    if is_onset(kind):   # onset acak 10-90% jendela, ramp sesuai jenis
        t0 = rng.integers(int(0.1 * T), int(0.9 * T), size=n)
        return ((1.0 - r * ramp_weights(T, t0, int(kind.split("-")[1]))) * X).astype(np.float32)
    raise ValueError(f"inject_attack mendukung A1/A2/A4-*, bukan {kind}")
