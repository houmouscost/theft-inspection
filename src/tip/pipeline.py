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


def run_seed(cfg: dict, seed: int, X, y, miss, F) -> dict[str, list]:
    ex = cfg["experiment"]
    rng = np.random.default_rng(seed)
    tr, cal, te = split_indices(y, cfg["data"]["split"], seed)
    rows = {"e1": [], "e2": [], "e4": [], "e5": []}

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
    perm = rng.permutation(te_h)
    n_th = int(round(ex["inject_rate"] * len(te_h)))
    th_idx, hon_idx = perm[:n_th], perm[n_th:]
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

    # ---- E5: sertifikat miss-rate
    c = cfg["certificate"]
    if c["variant"] in dets and c["variant"] in game_vars:
        det = dets[c["variant"]]
        hs = det.score(F[hon_idx])
        pools = {}
        need = {k for pair in c["pairs"] for k in pair}
        for kind in need:
            S = np.stack([det.score(Fj) for Fj in grid_F[kind]], axis=1)
            sol = solve(S, grid_G[kind], hs, c["budget"], c["eps"], c["tau"], c["penalty"],
                        ex["br_iters"], np.random.default_rng(seed))
            act = sol["idx"] > 0
            pools[kind] = (1.0 - sol["p_th"])[act]
        for cal_k, dep_k in c["pairs"]:
            pc, pdp = pools[cal_k], pools[dep_k]
            if len(pc) < 10 or len(pdp) < 10:
                continue
            if cal_k == dep_k:   # kalibrasi/deployment dipisah acak -> pertukarkan (exchangeable)
                pm = rng.permutation(pc)
                pc, pdp = pm[: len(pm) // 2], pm[len(pm) // 2:]
            res = conformal.coverage(pc, pdp, c["alpha"], c["n_cal"], c["trials"], rng)
            rows["e5"].append({"seed": seed, "cal": cal_k, "dep": dep_k, **res})
    return rows


def stage_experiments(cfg: dict, dirs: dict):
    X, y, miss, F = _load_artifacts(dirs)
    acc = {"e1": [], "e2": [], "e4": [], "e5": []}
    for seed in cfg["seeds"]:
        log(f"seed {seed} ...")
        r = run_seed(cfg, seed, X, y, miss, F)
        for k in acc:
            acc[k].extend(r[k])
    names = {"e1": "e1_detector", "e2": "e2_game", "e4": "e4_mimicry", "e5": "e5_certificate"}
    for k, rowsk in acc.items():
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
