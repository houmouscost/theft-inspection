"""Pipeline ujung-ke-ujung: data -> fitur -> latih -> adversary -> permainan -> teori -> sertifikat.

    python -m tip.pipeline --config configs/smoke.yaml --stage all
Tahap: data | experiments | report | all
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from . import conformal, theory
from .adversary import Adversary, inject_attack, is_onset
from .config import load_config, run_dirs
from .data import load_dataset, split_indices
from .detector import Detector
from .features import ALL_NAMES, compute_features
from .game import outcome_metrics, solve, solve_quantal
from .policies import fit_policy

LOGM = ALL_NAMES.index("log_mean")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- tahap 1
def stage_data(cfg: dict, dirs: dict):
    log(f"memuat data ({cfg['data']['source']}) ...")
    X, y, miss = load_dataset(cfg, seed=cfg["seeds"][0])
    log(f"  X={X.shape}  pencuri={int(y.sum())} ({y.mean():.2%})  rerata NaN={miss.mean():.2%}")
    F = compute_features(X, miss)
    np.savez(dirs["artifacts"] / "data.npz", X=X, y=y, miss=miss, F=F)
    log(f"  tersimpan -> {dirs['artifacts'] / 'data.npz'}")


def _load_artifacts(dirs):
    z = np.load(dirs["artifacts"] / "data.npz")
    X, y, miss, F = z["X"], z["y"], z["miss"], z["F"]
    if F.shape[1] != len(ALL_NAMES):   # cache dari versi fitur lama -> hitung ulang (kolom lama tak berubah)
        log(f"  fitur cache {F.shape[1]} kolom != {len(ALL_NAMES)}; menghitung ulang")
        F = compute_features(X, miss)
        np.savez(dirs["artifacts"] / "data.npz", X=X, y=y, miss=miss, F=F)
    return X, y, miss, F


def _onsets(n: int, T: int, seed: int) -> np.ndarray:
    """Onset eksogen A4-*: seragam pada 10-90% jendela, aliran acak tersendiri (tak menggeser rng utama)."""
    return np.random.default_rng(seed).integers(int(0.1 * T), int(0.9 * T), size=n)


def _make_adv(kind, Xs, rng, bank_X, onset):
    return Adversary(kind, Xs, rng, bank_X if kind == "A3" else None, onset if is_onset(kind) else None)


# --------------------------------------------------------------------------- tahap 2
def _adv_features(X, y, miss, cfg, rng):
    a = cfg["adv_train"]
    honest = np.where(y == 0)[0]
    n_pos = int((y == 1).sum())
    if n_pos == 0:   # data tanpa label (mis. UCI): positif latih seluruhnya serangan sintetis
        n_pos = int(round(cfg["experiment"]["inject_rate"] * len(y)))
    n_aug = min(int(a["ratio"] * n_pos), len(honest))
    if n_aug == 0:
        return np.zeros((0, len(ALL_NAMES)), dtype=np.float32)
    sel = rng.choice(honest, size=n_aug, replace=False)
    out = []
    kinds = a["kinds"]
    for k, kind in enumerate(kinds):
        part = sel[k::len(kinds)]
        r = rng.uniform(a["r_range"][0], a["r_range"][1], size=len(part))
        Y = inject_attack(X[part], kind, r, rng)
        out.append(compute_features(Y, miss[part]))
    return np.vstack(out)


def _sample_thieves(te_h, n_th, X, y, F, tr, ex, rng):
    """Pilih konsumen jujur-uji yang dijadikan pencuri (skenario S1/S2).

    * exclude_zero_load (default True): konsumen dengan beban rata-rata 0 tidak bisa mencuri;
      memasukkannya hanya menggelembungkan metrik `deterred` dan menurunkan recall.
    * thief_sampling = uniform (S1)      : acak seragam dari kandidat.
    * thief_sampling = level_matched (S2): bobot = rasio densitas log-level pencuri BERLABEL
      (split latih) terhadap kandidat, sehingga level beban sebenarnya pencuri tersimulasi
      mengikuti pencuri SGCC nyata (yang rata-rata berlevel lebih tinggi).
    """
    cand = te_h
    if ex.get("exclude_zero_load", True):
        cand = te_h[X[te_h].mean(1) > 0]
    n_th = min(n_th, len(cand))
    mode = ex.get("thief_sampling", "uniform")
    if mode == "uniform":
        return rng.choice(cand, size=n_th, replace=False)
    if mode == "level_matched":
        lm = F[:, LOGM]
        ref = lm[tr][(y[tr] == 1) & (X[tr].mean(1) > 0)]
        edges = np.quantile(lm[cand], np.linspace(0, 1, 31))
        edges[0], edges[-1] = -np.inf, np.inf
        h_ref = np.histogram(ref, bins=edges)[0] / len(ref)
        g_c = np.histogram(lm[cand], bins=edges)[0] / len(cand)
        b_of = np.clip(np.searchsorted(edges, lm[cand], side="right") - 1, 0, len(h_ref) - 1)
        w = h_ref[b_of] / np.maximum(g_c[b_of], 1e-12)
        if (w > 0).sum() < n_th:   # terlalu sedikit kandidat di bin berbobot -> lantai kecil agar sampel lengkap
            w = w + 1e-6 * w.max()
        return rng.choice(cand, size=n_th, replace=False, p=w / w.sum())
    raise ValueError(f"thief_sampling tidak dikenal: {mode}")


def _parse_pool(spec: str):
    """'A2' -> (A2, None) = pencuri adaptif; 'A2@0.6' -> (A2, 0.6) = pencuri statis r=0.6."""
    if "@" in spec:
        k, r = spec.split("@")
        return k, float(r)
    return spec, None


def run_seed(cfg: dict, seed: int, X, y, miss, F) -> dict[str, list]:
    ex = cfg["experiment"]
    rng = np.random.default_rng(seed)
    stages = ex.get("stages")      # None = semua; mis. [e4, e5] = hanya diagnostik murah (aliran acak utama tak berubah)

    def on(k: str) -> bool:
        return stages is None or k in stages
    tr, cal, te = split_indices(y, cfg["data"]["split"], seed)
    rows = {"e0": [], "e1": [], "e2": [], "e4": [], "e5": [], "e6": [], "e7": [], "e7p": [], "e7v": [], "e9": []}

    # ---- latih detektor
    dets: dict[str, Detector] = {}
    F_aug = None
    for v in cfg["variants"]:
        Ftr, ytr = F[tr], y[tr]
        if v["adv_train"]:
            if F_aug is None:
                F_aug = _adv_features(X[tr], y[tr], miss[tr], cfg, rng)
            Ftr = np.vstack([Ftr, F_aug])
            ytr = np.concatenate([ytr, np.ones(len(F_aug), dtype=ytr.dtype)])
        if v.get("model", "gbm") == "cnn":     # ekstensi 4: Wide & Deep CNN pada deret mentah
            if v["adv_train"]:
                raise ValueError("adv_train belum didukung untuk model cnn")
            from .deep import SeriesDetector
            dets[v["name"]] = SeriesDetector(v["name"], cfg.get("cnn", {}), seed).fit(X[tr], y[tr], X[cal], y[cal])
            continue
        dets[v["name"]] = Detector(v["name"], v["use_level"], v["use_missing"], cfg["detector"], seed,
                                   use_change=v.get("use_change", False)).fit(Ftr, ytr, F[cal], y[cal])

    # ---- E1: kinerja pada label nyata (skenario S0, tanpa adaptasi)
    yt = y[te]
    for name, det in (dets.items() if len(np.unique(yt)) == 2 and on("e1") else []):   # E1 butuh label nyata
        s = det.score_rows(F[te], X[te])
        hon_te = yt == 0
        # korelasi peringkat skor vs log-level pada konsumen jujur: >0 berarti "level tinggi = curiga"
        base = {"seed": seed, "variant": name, "auc": roc_auc_score(yt, s),
                "pr_auc": average_precision_score(yt, s),
                "spearman_level": float(spearmanr(s[hon_te], F[te][hon_te, LOGM]).statistic)}
        order = np.argsort(-s)
        for b in ex["budgets"]:
            k = max(int(np.ceil(b * len(te))), 1)
            top = yt[order[:k]]
            rows["e1"].append({**base, "budget": b, "recall_at_b": top.sum() / max(yt.sum(), 1),
                               "precision_at_b": top.mean()})

    # ---- populasi S1: konsumen jujur-uji, sebagian dijadikan pencuri adaptif
    te_h = te[y[te] == 0]
    n_th = int(round(ex["inject_rate"] * len(te_h)))
    th_idx = _sample_thieves(te_h, n_th, X, y, F, tr, ex, rng)
    hon_idx = rng.permutation(np.setdiff1d(te_h, th_idx))
    load = X.mean(1)
    rows["e0"].append({"seed": seed, "n_consumers": len(y), "n_days": X.shape[1],
                       "n_labelled_thieves": int(y.sum()), "nan_rate": float(miss.mean()),
                       "zero_load_frac_honest": float((load[y == 0] <= 0).mean()),
                       "median_load_honest": float(np.median(load[y == 0])),
                       "median_load_labelled_thief": float(np.median(load[y == 1])),
                       "n_test": len(te), "n_test_honest": len(te_h), "n_s1_thieves": len(th_idx),
                       "median_load_s1_thief": float(np.median(load[th_idx])),
                       "median_load_s1_honest": float(np.median(load[hon_idx]))})
    X_th, miss_th = X[th_idx], miss[th_idx]
    total_load = float(X[te_h].sum())
    bank_pool = np.where(y[tr] == 0)[0]
    if len(bank_pool) > ex["a3_bank_max"]:
        bank_pool = rng.choice(bank_pool, ex["a3_bank_max"], replace=False)
    bank_X = X[tr][bank_pool]
    r_grid = np.array(ex["r_grid"], dtype=float)
    onset_te = _onsets(len(th_idx), X.shape[1], seed + 101)

    # fitur/stolen per kelas adversary dihitung sekali (tak bergantung detektor/kebijakan)
    keep_X = any(getattr(d, "uses_series", False) for d in dets.values())
    grid_F, grid_G, grid_X = {}, {}, {}
    for kind in ex["adversaries"]:
        adv = _make_adv(kind, X_th, rng, bank_X, onset_te)
        reps = [adv.report(r) for r in r_grid]
        grid_F[kind] = [compute_features(Y, miss_th) for Y in reps]
        grid_X[kind] = reps if keep_X else [None] * len(reps)
        grid_G[kind] = np.stack([adv.stolen(r) for r in r_grid], axis=1)

    def score_grid(det, kind):
        return np.stack([det.score_rows(Fj, Xj) for Fj, Xj in zip(grid_F[kind], grid_X[kind])], axis=1)

    game_vars = [v["name"] for v in cfg["variants"] if v.get("game", True)]
    sol_cache = {}
    # populasi untuk batas mimikri berbasis KDE: g = level semua konsumen jujur; h = level populasi asal pencuri
    # (S1: semua jujur yang memenuhi syarat; sampling lain: pencuri terealisasi)
    load_all = X.mean(1)
    elig_all = (y == 0) & ((load_all > 0) if ex.get("exclude_zero_load", True) else True)
    lg_pop = F[y == 0, LOGM]
    lh_pop = F[elig_all, LOGM] if ex.get("thief_sampling", "uniform") == "uniform" else F[th_idx, LOGM]
    tv_kde_cache: dict = {}
    rr4 = np.random.default_rng(seed + 707)      # aliran acak terpisah untuk E4
    for name in game_vars:
        det = dets[name]
        hs = det.score_rows(F[hon_idx], X[hon_idx])
        for kind in ex["adversaries"]:
            S = score_grid(det, kind)
            G = grid_G[kind]
            base = {"seed": seed, "variant": name, "adversary": kind}
            # E2 adaptif
            for b in (ex["budgets"] if on("e2") else []):
                for phi in ex["penalties"]:
                    for eps in ex["eps_grid"]:
                        for tau in ex["tau_grid"]:
                            sol = solve(S, G, hs, b, eps, tau, phi, ex["br_iters"],
                                        np.random.default_rng(seed))
                            m = outcome_metrics(sol, G, total_load)
                            rows["e2"].append({**base, "mode": "adaptive", "r_static": np.nan,
                                               "budget": b, "penalty": phi, "eps": eps,
                                               "tau": tau, "mean_r": float(r_grid[sol["idx"]].mean()),
                                               **m})
                            sol_cache[(name, kind, b, phi, eps, tau)] = sol
                    # E3 statis (non-adaptif): tidak bergantung phi, ditulis sekali per b
                for eps in ex["eps_grid"]:
                    for tau in ex["tau_grid"]:
                        for rs in ex["r_static"]:
                            j = int(np.argmin(np.abs(r_grid - rs)))
                            sol = solve(S, G, hs, b, eps, tau, 0.0, fixed_idx=j)
                            m = outcome_metrics(sol, G, total_load)
                            rows["e2"].append({**base, "mode": "static", "r_static": rs,
                                               "budget": b, "penalty": np.nan, "eps": eps,
                                               "tau": tau, "mean_r": rs, **m})
            # E4: batas mimikri vs KS empiris (+ estimasi sampel-hingga: TV-KDE, akurasi seimbang cross-fit, null permutasi)
            if on("e4") and (name in ex.get("e4_variants", ["gbm_level", "gbm_shape"]) or len(game_vars) == 1):
                lg, lh = F[hon_idx, LOGM], F[th_idx, LOGM]
                for j, r in enumerate(r_grid):
                    if r <= 0:
                        continue
                    if r not in tv_kde_cache:
                        tv_kde_cache[r] = tuple(theory.mimicry_tv_kde(lg_pop, lh_pop, r, m) for m in (1.0, 0.5, 2.0))
                    ks = theory.ks_stat(hs, S[:, j])
                    ba_cf, ba_se = theory.ba_crossfit(hs, S[:, j], rr4)
                    rows["e4"].append({"seed": seed, "variant": name, "adversary": kind, "r": r,
                                       "tv_bound": theory.mimicry_tv(lg, lh, r),
                                       "tv_kde": tv_kde_cache[r][0], "tv_kde_lo": tv_kde_cache[r][1],
                                       "tv_kde_hi": tv_kde_cache[r][2],
                                       "ks_emp": ks, "balacc_emp": 0.5 * (1 + ks),
                                       "ba_cf": ba_cf, "ba_cf_se": ba_se,
                                       "ks_null": theory.ks_perm_null(hs, S[:, j], rr4),
                                       "n_h": len(hs), "n_t": len(S)})

    # ---- E5: sertifikat miss-rate (kalibrasi vs deployment; pool 'A2@0.6' = pencuri statis)
    c = cfg.get("certificate") or {}
    cert_vars = c.get("variants", [c["variant"]] if "variant" in c else []) if on("e5") else []
    n_cals = c.get("n_cal", 100)
    n_cals = n_cals if isinstance(n_cals, list) else [n_cals]
    ar = np.arange(len(th_idx))
    for vname in cert_vars:
        if vname not in dets:
            continue
        det = dets[vname]
        hs = det.score_rows(F[hon_idx], X[hon_idx])
        pools = {}
        for spec in sorted({k for pair in c["pairs"] for k in pair}):
            kind, rs = _parse_pool(spec)
            S = score_grid(det, kind)
            G = grid_G[kind]
            if rs is None:
                sol = solve(S, G, hs, c["budget"], c["eps"], c["tau"], c["penalty"],
                            ex["br_iters"], np.random.default_rng(seed))
            else:
                j = int(np.argmin(np.abs(r_grid - rs)))
                sol = solve(S, G, hs, c["budget"], c["eps"], c["tau"], c["penalty"], fixed_idx=j)
            act = (sol["idx"] > 0) & (G[ar, sol["idx"]] > 0)
            pools[spec] = (1.0 - sol["p_th"])[act]
        for cal_k, dep_k in c["pairs"]:
            for n_cal in n_cals:
                pc, pdp = pools[cal_k], pools[dep_k]
                if cal_k == dep_k:   # kalibrasi/deployment dipisah acak -> pertukarkan (exchangeable)
                    pm = rng.permutation(pc)
                    pc, pdp = pm[: len(pm) // 2], pm[len(pm) // 2:]
                if len(pc) < 10 or len(pdp) < 10:
                    continue
                sd = conformal.shift_diagnostics(pc, pdp)
                res = conformal.coverage(pc, pdp, c["alpha"], n_cal, c["trials"], rng, tv_adj=sd["ks_pools"])
                rows["e5"].append({"seed": seed, "variant": vname, "cal": cal_k, "dep": dep_k,
                                   "n_cal_req": n_cal, "mean_miss_cal": float(np.mean(pc)), **res, **sd,
                                   "cov_conformal_lb": max(0.0, 1 - c["alpha"] - sd["ks_pools"])})

    # ---- E6: sapuan deterensi (Prop. 3) ke anggaran besar
    d = cfg.get("deterrence") or {}
    if d.get("variant") in dets and on("e6"):
        det = dets[d["variant"]]
        hs = det.score_rows(F[hon_idx], X[hon_idx])
        for kind in d["adversaries"]:
            S = score_grid(det, kind)
            G = grid_G[kind]
            for b in d["budgets"]:
                for phi in d["penalties"]:
                    for eps in d["eps_grid"]:
                        sol = solve(S, G, hs, b, eps, 0.0, phi, d.get("br_iters", ex["br_iters"]),
                                    np.random.default_rng(seed))
                        m = outcome_metrics(sol, G, total_load)
                        rows["e6"].append({"seed": seed, "variant": d["variant"], "adversary": kind,
                                           "budget": b, "penalty": phi, "eps": eps,
                                           "floor_index": eps * b * (1 + phi),
                                           "mean_r": float(r_grid[sol["idx"]].mean()), **m})

    # ---- E7: kebijakan sadar-strategi (Bayesian Stackelberg MILP), ekstensi 1
    sp = cfg.get("strategic") or {}
    if sp.get("variants") and r_grid[0] == 0 and on("e7"):
        from . import stackelberg as sb
        kinds7 = sp.get("adversaries", ex["adversaries"])

        def union(Fg, Xg, Gg):
            """Gabungkan grid per jenis serangan: kolom 0 = jujur, lalu (jenis, r>0)."""
            Fs, Xs, Gs, lab = [Fg[kinds7[0]][0]], [Xg[kinds7[0]][0]], [Gg[kinds7[0]][:, 0]], [("none", 0.0)]
            for kind in kinds7:
                for j in range(1, len(r_grid)):
                    Fs.append(Fg[kind][j]); Xs.append(Xg[kind][j]); Gs.append(Gg[kind][:, j])
                    lab.append((kind, float(r_grid[j])))
            return Fs, Xs, np.stack(Gs, 1), lab

        # pencuri latih: injeksi pada konsumen jujur split KALIBRASI (aturan sampling sama dengan uji)
        cal_h = cal[y[cal] == 0]
        cth = _sample_thieves(cal_h, int(round(ex["inject_rate"] * len(cal_h))), X, y, F, tr, ex, rng)
        chon = np.setdiff1d(cal_h, cth)
        cF, cX, cG = {}, {}, {}
        for kind in kinds7:
            adv = _make_adv(kind, X[cth], rng, bank_X, _onsets(len(cth), X.shape[1], seed + 202))
            reps = [adv.report(r) for r in r_grid]
            cF[kind] = [compute_features(Y, miss[cth]) for Y in reps]
            cX[kind] = reps if keep_X else [None] * len(reps)
            cG[kind] = np.stack([adv.stolen(r) for r in r_grid], axis=1)
        uF_c, uX_c, uG_c, _ = union(cF, cX, cG)
        uF_t, uX_t, uG_t, lab = union(grid_F, grid_X, grid_G)
        r_of = np.array([l[1] for l in lab])
        L_c = np.stack([f[:, LOGM] for f in uF_c], 1)
        L_t = np.stack([f[:, LOGM] for f in uF_t], 1)
        for vname in sp["variants"]:
            if vname not in dets:
                continue
            det = dets[vname]
            hs_c, hl_c = det.score_rows(F[chon], X[chon]), F[chon, LOGM]
            hs_t, hl_t = det.score_rows(F[hon_idx], X[hon_idx]), F[hon_idx, LOGM]
            S_c = np.stack([det.score_rows(f, x) for f, x in zip(uF_c, uX_c)], 1)
            S_t = np.stack([det.score_rows(f, x) for f, x in zip(uF_t, uX_t)], 1)
            base = {"seed": seed, "variant": vname, "adversary": "union"}
            for b in sp.get("budgets", ex["budgets"]):
                for phi in sp.get("penalties", ex["penalties"]):
                    for eps in sp.get("baseline_eps", [0.0, 1.0]):     # pembanding: keluarga P(eps, 0)
                        sol = solve(S_t, uG_t, hs_t, b, eps, 0.0, phi, ex["br_iters"], np.random.default_rng(seed))
                        m = outcome_metrics(sol, uG_t, total_load)
                        rows["e7"].append({**base, "budget": b, "penalty": phi,
                                           "policy": "topk" if eps == 0 else ("random" if eps == 1 else f"mix{eps:g}"),
                                           "mean_r": float(r_of[sol["idx"]].mean()),
                                           "budget_used": float((sol["p_th"].sum() + sol["p_h"].sum()) / (len(hs_t) + len(S_t))),
                                           **{k: v for k, v in m.items() if k != "precision"}})
                    for mode in sp.get("modes", ["score", "level", "joint"]):
                        n_s, n_l = sp.get("n_bins", [6, 6]) if mode == "joint" else (sp.get("n_bins_1d", 12),) * 2
                        extra_q = sorted({1 - bb for bb in sp.get("budgets", ex["budgets"])} | {0.95, 0.99}) \
                            if mode in ("score", "joint") else ()
                        cells = sb.make_cells(hs_c, hl_c, mode, n_s, n_l, extra_q)
                        s_lv = np.r_[0.0, np.unique(np.r_[np.linspace(0, 1, n_s + 1)[1:-1], list(extra_q)])]
                        Kt, Kht = cells.of(S_t, L_t), cells.of(hs_t, hl_t)
                        thr = np.quantile(hs_c, 1 - b)   # sel skor yang seluruhnya di atas kuantil (1-b)
                        top = [k for k in range(cells.K) if cells.score_bin(k) >= 0 and
                               (cells.edges_s[cells.score_bin(k) - 1] if cells.score_bin(k) > 0 else -np.inf) >= thr - 1e-12]
                        fits = [("milp_" + mode, cells.of(S_c, L_c), uG_c, cells.of(hs_c, hl_c))]
                        if mode in sp.get("oracle_modes", ["joint"]):
                            fits.append(("oracle_" + mode, Kt, uG_t, Kht))
                        for pname, Kfit, Gfit, Khfit in fits:
                            if sp.get("solver", "local") == "milp":
                                pv, info = sb.fit_milp(Kfit, Gfit, Khfit, cells.K, b, phi,
                                                       sp.get("time_limit", 120), sp.get("mip_gap", 0.01))
                            else:
                                pv, info = sb.fit_local(Kfit, Gfit, Khfit, cells.K, b, phi, sp.get("max_iter", 40),
                                                        top_cells=top or None)
                            m = sb.evaluate(pv, Kt, uG_t, Kht, b, phi, total_load)
                            idx = m.pop("idx")
                            rows["e7"].append({**base, "budget": b, "penalty": phi, "policy": pname,
                                               "mean_r": float(r_of[idx].mean()), "converged": info["status"] == 0,
                                               **m, **info})
                            if pname.startswith("milp_"):
                                nh = np.bincount(Khfit, minlength=cells.K)
                                for k in range(cells.K):
                                    rows["e7p"].append({"seed": seed, "variant": vname, "budget": b, "penalty": phi,
                                                        "mode": mode, "cell": k, "p": float(pv[k]), "n_honest_cal": int(nh[k]),
                                                        "s_bin": k // (len(cells.edges_l) + 1) if mode == "joint" else (k if mode == "score" else -1),
                                                        "s_q_lo": float(s_lv[cells.score_bin(k)]) if cells.score_bin(k) >= 0 and cells.score_bin(k) < len(s_lv) else np.nan,
                                                        "l_bin": k % (len(cells.edges_l) + 1) if mode == "joint" else (k if mode == "level" else -1)})
                        if mode == "level" and sp.get("interval", False):
                            # aturan interval 3-parameter pada sel level (keluarga lebih sederhana, dapat diterapkan utilitas)
                            pv, info = sb.fit_interval(cells.of(S_c, L_c), uG_c, cells.of(hs_c, hl_c), cells.K, b, phi)
                            m = sb.evaluate(pv, Kt, uG_t, Kht, b, phi, total_load)
                            idx = m.pop("idx")
                            rows["e7"].append({**base, "budget": b, "penalty": phi, "policy": "interval_level",
                                               "mean_r": float(r_of[idx].mean()), "converged": True, **m, **info})

        # validasi solver: pada subsampel kecil pencuri kalibrasi, bandingkan pencarian lokal dengan MILP eksak
        vd = sp.get("validate") or {}
        if vd.get("n_sub") and sp["variants"][0] in dets and seed in vd.get("seeds", [seed]):
            det = dets[sp["variants"][0]]
            hs_c, hl_c = det.score_rows(F[chon], X[chon]), F[chon, LOGM]
            S_c = np.stack([det.score_rows(f, x) for f, x in zip(uF_c, uX_c)], 1)
            rv = np.random.default_rng(seed + 505)
            for rep in range(vd.get("reps", 1)):
                sub = rv.choice(len(cth), size=min(vd["n_sub"], len(cth)), replace=False)
                hsub = rv.choice(len(chon), size=int(round(len(chon) * len(sub) / len(cth))), replace=False)
                for b in vd.get("budgets", [0.02]):
                    for phi in vd.get("penalties", [3.0]):
                        cells = sb.make_cells(hs_c, hl_c, "joint", *sp.get("n_bins", [6, 6]))
                        Kth_, G_ = cells.of(S_c[sub], L_c[sub]), uG_c[sub]
                        Kh_ = cells.of(hs_c[hsub], hl_c[hsub])
                        nh_ = np.bincount(Kh_, minlength=cells.K).astype(float)
                        out = {}
                        for solver in ("local", "milp"):
                            if solver == "local":
                                pv, info = sb.fit_local(Kth_, G_, Kh_, cells.K, b, phi, sp.get("max_iter", 40))
                            else:
                                pv, info = sb.fit_milp(Kth_, G_, Kh_, cells.K, b, phi, vd.get("time_limit", 600), 1e-4)
                            obj, used = sb._objective(pv, Kth_, G_, nh_, 1 + phi)
                            out[solver] = (obj, used, info)
                        uni, _ = sb._objective(np.full(cells.K, b), Kth_, G_, nh_, 1 + phi)
                        rows["e7v"].append({"seed": seed, "rep": rep, "budget": b, "penalty": phi, "n_sub": len(sub),
                                            "obj_uniform": uni, "obj_local": out["local"][0], "obj_milp": out["milp"][0],
                                            "budget_local": out["local"][1] / (len(hsub) + len(sub)),
                                            "budget_milp": out["milp"][1] / (len(hsub) + len(sub)),
                                            "milp_status": out["milp"][2]["status"], "milp_s": out["milp"][2]["solve_s"],
                                            "milp_gap": out["milp"][2]["mip_gap"], "milp_dual_bound": out["milp"][2]["dual_bound"],
                                            "local_s": out["local"][2]["solve_s"]})

    # ---- E9: pencuri berasionalitas terbatas (quantal response) & serangan transfer, ekstensi 3
    bd = cfg.get("bounded") or {}
    if bd.get("variant") in dets and on("e9"):
        det = dets[bd["variant"]]
        hs = det.score_rows(F[hon_idx], X[hon_idx])
        sur = None
        if bd.get("transfer", True):
            # surrogate: famili model sama, data berbeda (50% split latih, seed lain) -> pencuri tak tahu model asli
            rs = np.random.default_rng(seed + 303)
            sub = rs.choice(tr, size=len(tr) // 2, replace=False)
            vsur = next(v for v in cfg["variants"] if v["name"] == bd["variant"])
            sur = Detector("surrogate", vsur["use_level"], vsur["use_missing"],
                           {**cfg["detector"], "learning_rate": bd.get("surrogate_lr", 0.05)}, seed + 303,
                           use_change=vsur.get("use_change", False)).fit(F[sub], y[sub], F[cal], y[cal])
            hs_sur = sur.score_rows(F[hon_idx], X[hon_idx])
        ar = np.arange(len(th_idx))
        for kind in bd.get("adversaries", ex["adversaries"]):
            S = score_grid(det, kind)
            G = grid_G[kind]
            S_sur = score_grid(sur, kind) if sur is not None else None
            for b in bd["budgets"]:
                for phi in bd["penalties"]:
                    for eps in bd["eps_grid"]:
                        base = {"seed": seed, "variant": bd["variant"], "adversary": kind, "budget": b,
                                "penalty": phi, "eps": eps}
                        sol = solve(S, G, hs, b, eps, 0.0, phi, ex["br_iters"], np.random.default_rng(seed))
                        m = outcome_metrics(sol, G, total_load)
                        rows["e9"].append({**base, "thief": "best_response", "lam": np.inf,
                                           "mean_r": float(r_grid[sol["idx"]].mean()), **m})
                        for lam in bd["lambdas"]:
                            q = solve_quantal(S, G, hs, b, eps, 0.0, phi, lam, ex["br_iters"],
                                              np.random.default_rng(seed + 404))
                            rows["e9"].append({**base, "thief": "quantal", "lam": lam,
                                               "mean_r": float((q["pi"] * r_grid[None, :]).sum(1).mean()),
                                               **q["metrics"](total_load)})
                        if S_sur is not None:   # pilih r terhadap surrogate, dinilai dengan detektor asli
                            ss = solve(S_sur, G, hs_sur, b, eps, 0.0, phi, ex["br_iters"], np.random.default_rng(seed))
                            pol = fit_policy(np.concatenate([hs, S[ar, ss["idx"]]]), b, eps, 0.0)
                            tsol = {"idx": ss["idx"], "p_th": pol.prob(S[ar, ss["idx"]]), "p_h": pol.prob(hs),
                                    "converged": ss["converged"]}
                            m = outcome_metrics(tsol, G, total_load)
                            rows["e9"].append({**base, "thief": "transfer", "lam": np.nan,
                                               "mean_r": float(r_grid[ss["idx"]].mean()), **m})
    return rows


def stage_experiments(cfg: dict, dirs: dict):
    X, y, miss, F = _load_artifacts(dirs)
    acc = {"e0": [], "e1": [], "e2": [], "e4": [], "e5": [], "e6": [], "e7": [], "e7p": [], "e7v": [], "e9": []}
    for seed in cfg["seeds"]:
        log(f"seed {seed} ...")
        r = run_seed(cfg, seed, X, y, miss, F)
        for k in acc:
            acc[k].extend(r[k])
    names = {"e0": "e0_data", "e1": "e1_detector", "e2": "e2_game", "e4": "e4_mimicry", "e5": "e5_certificate", "e6": "e6_deterrence", "e7": "e7_strategic", "e7p": "e7_policy_cells", "e7v": "e7_solver_check", "e9": "e9_bounded"}
    for k, rowsk in acc.items():
        if not rowsk:   # eksperimen dimatikan di config -> hapus CSV basi dari run sebelumnya
            (dirs["results"] / f"{names[k]}.csv").unlink(missing_ok=True)
            continue
        df = pd.DataFrame(rowsk)
        df.to_csv(dirs["results"] / f"{names[k]}.csv", index=False)
        log(f"  {names[k]}.csv  ({len(df)} baris)")


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stage", default="all", choices=["data", "experiments", "diag", "report", "all"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    dirs = run_dirs(cfg)
    if a.stage in ("experiments", "all"):   # catat config efektif di samping hasil (dibaca laporan)
        import yaml
        (dirs["results"] / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if a.stage in ("data", "all"):
        stage_data(cfg, dirs)
    if a.stage in ("experiments", "all"):
        stage_experiments(cfg, dirs)
    if a.stage in ("diag", "all"):   # diagnostik data (seleksi label, kebersihan, berbeban nol); butuh data.npz
        from . import diagnostics
        diagnostics.run(cfg, dirs, *_load_artifacts(dirs))
    if a.stage in ("report", "all"):
        from .report import stage_report
        stage_report(cfg, dirs)
    log("selesai.")


if __name__ == "__main__":
    main()
