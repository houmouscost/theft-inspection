"""Pipeline ujung-ke-ujung: data -> fitur -> latih -> adversary -> permainan -> teori -> sertifikat.

    python -m tip.pipeline --config configs/smoke.yaml --stage all
Tahap: data | experiments | report | all
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import conformal, theory
from .adversary import Adversary, inject_attack
from .config import load_config, run_dirs
from .data import load_dataset, split_indices
from .detector import Detector
from .features import ALL_NAMES, compute_features
from .game import outcome_metrics, solve

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
    return z["X"], z["y"], z["miss"], z["F"]


# --------------------------------------------------------------------------- tahap 2
def _adv_features(X, y, miss, cfg, rng):
    a = cfg["adv_train"]
    honest = np.where(y == 0)[0]
    n_aug = min(int(a["ratio"] * (y == 1).sum()), len(honest))
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
    tr, cal, te = split_indices(y, cfg["data"]["split"], seed)
    rows = {"e0": [], "e1": [], "e2": [], "e4": [], "e5": [], "e6": []}

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
        dets[v["name"]] = Detector(v["name"], v["use_level"], v["use_missing"],
                                   cfg["detector"], seed).fit(Ftr, ytr, F[cal], y[cal])

    # ---- E1: kinerja pada label nyata (skenario S0, tanpa adaptasi)
    yt = y[te]
    for name, det in dets.items():
        s = det.score(F[te])
        base = {"seed": seed, "variant": name, "auc": roc_auc_score(yt, s),
                "pr_auc": average_precision_score(yt, s)}
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

    # fitur/stolen per kelas adversary dihitung sekali (tak bergantung detektor/kebijakan)
    grid_F, grid_G = {}, {}
    for kind in ex["adversaries"]:
        adv = Adversary(kind, X_th, rng, bank_X if kind == "A3" else None)
        grid_F[kind] = [compute_features(adv.report(r), miss_th) for r in r_grid]
        grid_G[kind] = np.stack([adv.stolen(r) for r in r_grid], axis=1)

    game_vars = [v["name"] for v in cfg["variants"] if v.get("game", True)]
    sol_cache = {}
    for name in game_vars:
        det = dets[name]
        hs = det.score(F[hon_idx])
        for kind in ex["adversaries"]:
            S = np.stack([det.score(Fj) for Fj in grid_F[kind]], axis=1)
            G = grid_G[kind]
            base = {"seed": seed, "variant": name, "adversary": kind}
            # E2 adaptif
            for b in ex["budgets"]:
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
            # E4: batas mimikri vs KS empiris
            if name in ("gbm_level", "gbm_shape") or len(game_vars) == 1:
                lg, lh = F[hon_idx, LOGM], F[th_idx, LOGM]
                for j, r in enumerate(r_grid):
                    if r <= 0:
                        continue
                    ks = theory.ks_stat(hs, S[:, j])
                    rows["e4"].append({"seed": seed, "variant": name, "adversary": kind, "r": r,
                                       "tv_bound": theory.mimicry_tv(lg, lh, r),
                                       "ks_emp": ks, "balacc_emp": 0.5 * (1 + ks)})

    # ---- E5: sertifikat miss-rate (kalibrasi vs deployment; pool 'A2@0.6' = pencuri statis)
    c = cfg.get("certificate") or {}
    cert_vars = c.get("variants", [c["variant"]] if "variant" in c else [])
    n_cals = c.get("n_cal", 100)
    n_cals = n_cals if isinstance(n_cals, list) else [n_cals]
    ar = np.arange(len(th_idx))
    for vname in cert_vars:
        if vname not in dets:
            continue
        det = dets[vname]
        hs = det.score(F[hon_idx])
        pools = {}
        for spec in sorted({k for pair in c["pairs"] for k in pair}):
            kind, rs = _parse_pool(spec)
            S = np.stack([det.score(Fj) for Fj in grid_F[kind]], axis=1)
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
                res = conformal.coverage(pc, pdp, c["alpha"], n_cal, c["trials"], rng)
                rows["e5"].append({"seed": seed, "variant": vname, "cal": cal_k, "dep": dep_k,
                                   "n_cal_req": n_cal, "mean_miss_cal": float(np.mean(pc)), **res})

    # ---- E6: sapuan deterensi (Prop. 3) ke anggaran besar
    d = cfg.get("deterrence") or {}
    if d.get("variant") in dets:
        det = dets[d["variant"]]
        hs = det.score(F[hon_idx])
        for kind in d["adversaries"]:
            S = np.stack([det.score(Fj) for Fj in grid_F[kind]], axis=1)
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
    return rows


def stage_experiments(cfg: dict, dirs: dict):
    X, y, miss, F = _load_artifacts(dirs)
    acc = {"e0": [], "e1": [], "e2": [], "e4": [], "e5": [], "e6": []}
    for seed in cfg["seeds"]:
        log(f"seed {seed} ...")
        r = run_seed(cfg, seed, X, y, miss, F)
        for k in acc:
            acc[k].extend(r[k])
    names = {"e0": "e0_data", "e1": "e1_detector", "e2": "e2_game", "e4": "e4_mimicry", "e5": "e5_certificate", "e6": "e6_deterrence"}
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
    ap.add_argument("--stage", default="all", choices=["data", "experiments", "report", "all"])
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
    if a.stage in ("report", "all"):
        from .report import stage_report
        stage_report(cfg, dirs)
    log("selesai.")


if __name__ == "__main__":
    main()
