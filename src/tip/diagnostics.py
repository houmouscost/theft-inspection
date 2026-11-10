"""Diagnostik data SGCC: (i) seleksi label, (ii) kebersihan data, (iii) kekokohan terhadap konsumen berbeban nol.

(i)  Laju pencuri berlabel menurut desil konsumsi dan kelas fraksi-hilang. Pola naik terhadap konsumsi (dan
     lebih tinggi pada fraksi-hilang besar, tidak monoton) konsisten dengan label yang dihasilkan oleh inspeksi yang lebih sering mendatangi konsumen besar
     (Glauner dkk. 2017/2018), sehingga inversi level pada detektor tidak boleh dibaca semata-mata
     sebagai sifat pencuri.
(ii) Hitungan dari berkas mentah: deret nol penuh, baris duplikat, label yang bertentangan, urutan berkas.
(iii) AUC label nyata dengan/tanpa konsumen berbeban nol (latih+uji), untuk memastikan hasil utama
     tidak bergantung pada mereka.
Keluaran: d_selection.csv, d_hygiene.csv, d_zero_load.csv di folder hasil.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .data.loader import read_sgcc_csv, split_indices
from .detector import Detector
from .features import ALL_NAMES

LOGM = ALL_NAMES.index("log_mean")


MISS_EDGES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0001]


def selection_table(X: np.ndarray, y: np.ndarray, miss: np.ndarray) -> pd.DataFrame:
    load = X.mean(1)
    rows = []
    pos = load > 0                                    # desil konsumsi hanya di antara konsumen berbeban positif
    q = np.quantile(load[pos], np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q, load, side="right") - 1, 0, 9)
    for d in range(10):
        m = pos & (dec == d)
        rows.append({"dimension": "consumption_decile", "bin": d + 1, "lo": q[d], "hi": q[d + 1],
                     "n": int(m.sum()), "thieves": int(y[m].sum()), "rate": float(y[m].mean())})
    # fraksi-hilang: lebar-tetap (bebas ikatan; kuintil dengan ikatan banyak -- fraksi hanya berkelipatan 1/T --
    # menghasilkan kotak tak sama besar dan perbandingan ujung yang menyesatkan). Hanya konsumen berbeban positif.
    edges = np.array(MISS_EDGES)
    mb = np.clip(np.searchsorted(edges, miss, side="right") - 1, 0, len(edges) - 2)
    for k in range(len(edges) - 1):
        m = pos & (mb == k)
        rows.append({"dimension": "missing_bin", "bin": k + 1, "lo": float(edges[k]), "hi": float(edges[k + 1]),
                     "n": int(m.sum()), "thieves": int(y[m].sum()), "rate": float(y[m].mean())})
    rho = float(spearmanr(miss[pos], y[pos]).statistic)
    rows.append({"dimension": "spearman_missing_label", "bin": 0, "lo": np.nan, "hi": np.nan,
                 "n": int(pos.sum()), "thieves": int(y[pos].sum()), "rate": rho})
    return pd.DataFrame(rows)


def hygiene_table(path: str) -> pd.DataFrame:
    """Hitungan pada berkas mentah (NaN dianggap sama saat mendeteksi duplikat)."""
    raw = pd.read_csv(path)
    ycol = next(c for c in raw.columns if str(c).strip().upper() == "FLAG")
    vals = raw.drop(columns=[c for c in raw.columns if str(c).strip().upper() in ("CONS_NO", "FLAG")]).to_numpy(dtype=np.float32)
    allnan = np.all(np.isnan(vals), axis=1)
    n_raw, n_allnan, n_allnan_th = len(raw), int(allnan.sum()), int(raw[ycol].to_numpy()[allnan].sum())
    X, y = read_sgcc_csv(path)                         # baris tanpa pembacaan sama sekali sudah dibuang
    n = len(y)
    allzero = np.nanmax(np.where(np.isnan(X), -1.0, X), axis=1) == 0   # semua nilai teramati = 0
    allzero &= ~np.all(np.isnan(X), axis=1)
    df = pd.DataFrame(np.nan_to_num(X, nan=-7777.0))   # penanda NaN agar duplikat memperlakukan NaN sama
    dup = df.duplicated(keep="first").to_numpy()
    # kelompok duplikat via hash baris
    h = pd.util.hash_pandas_object(df, index=False).to_numpy()
    s = pd.DataFrame({"h": h, "y": y, "zero": allzero, "pos": np.nanmax(np.where(np.isnan(X), -1.0, X), axis=1) > 0})
    g = s.groupby("h").agg(n=("y", "size"), ymin=("y", "min"), ymax=("y", "max"), zero=("zero", "max"), pos=("pos", "max"))
    g = g[g.n > 1]
    first_h = int(np.argmax(y == 0))                   # posisi honest pertama
    out = {
        "n_raw_rows": n_raw, "n_allmissing_rows": n_allnan, "n_allmissing_thieves": n_allnan_th,
        "n_rows": n, "n_thieves": int(y.sum()),
        "n_allzero": int(allzero.sum()), "n_allzero_thieves": int(y[allzero].sum()),
        "n_dup_rows": int(dup.sum()), "n_dup_groups": int(len(g)), "n_rows_in_dup_groups": int(g.n.sum()),
        "n_dup_rows_allzero_groups": int(g.loc[g.zero, "n"].sum()) if len(g) else 0,
        "n_rows_in_dup_groups_positive": int(g.loc[g.pos, "n"].sum()) if len(g) else 0,
        "n_dup_groups_positive": int(g.pos.sum()) if len(g) else 0,
        "n_dup_groups_conflict": int((g.ymin != g.ymax).sum()) if len(g) else 0,
        "first_honest_position": first_h,
        "thieves_in_first_block": int(y[: int(y.sum())].sum()),
        "file_sorted_thieves_first": bool(y[: int(y.sum())].all()),
    }
    return pd.DataFrame({"item": list(out), "value": list(out.values())})


def zero_load_table(X, y, miss, F, cfg, seeds) -> pd.DataFrame:
    load = X.mean(1)
    nz = load > 0
    rows = []
    for seed in seeds:
        tr, cal, te = split_indices(y, cfg["data"]["split"], seed)
        for name, use_level in (("gbm_level", True), ("gbm_shape", False)):
            def fit(trn, cln):
                return Detector(name, use_level, False, cfg["detector"], seed).fit(F[trn], y[trn], F[cln], y[cln])
            det = fit(tr, cal)
            s = det.score_rows(F[te], X[te])
            auc_all = roc_auc_score(y[te], s)
            auc_te_nz = roc_auc_score(y[te][nz[te]], s[nz[te]])
            det2 = fit(tr[nz[tr]], cal[nz[cal]])
            s2 = det2.score_rows(F[te][nz[te]], X[te][nz[te]])
            auc_nz = roc_auc_score(y[te][nz[te]], s2)
            h = (y[te] == 0) & nz[te]
            rho = float(spearmanr(s2[(y[te][nz[te]] == 0)], F[te][nz[te]][(y[te][nz[te]] == 0), LOGM]).statistic)
            rows.append({"seed": seed, "variant": name, "auc_all": auc_all, "auc_test_nonzero": auc_te_nz,
                         "auc_train_test_nonzero": auc_nz, "spearman_level_nonzero": rho,
                         "n_test": len(te), "n_test_nonzero": int(nz[te].sum())})
    return pd.DataFrame(rows)


def run(cfg: dict, dirs: dict, X, y, miss, F):
    R = dirs["results"]
    if int(y.sum()) == 0:        # tanpa label (mis. UCI) diagnostik ini tidak bermakna
        return None, None
    sel = selection_table(X, y, miss)
    sel.to_csv(R / "d_selection.csv", index=False)
    zl = zero_load_table(X, y, miss, F, cfg, cfg["seeds"])
    zl.to_csv(R / "d_zero_load.csv", index=False)
    if cfg["data"]["source"] == "sgcc":
        hyg = hygiene_table(cfg["data"]["sgcc_path"])
        hyg.to_csv(R / "d_hygiene.csv", index=False)
    return sel, zl
