import numpy as np

from tip.adversary import Adversary
from tip.conformal import conformal_upper, hoeffding_upper
from tip.features import ALL_NAMES, SHAPE_NAMES, compute_features
from tip.game import solve
from tip.policies import fit_policy
from tip.data.synthetic import make_synthetic
from tip.data.loader import clean_series
from tip.theory import ks_stat, mimicry_tv


def _data(n=400, T=120, seed=0):
    X, y = make_synthetic(n, T, 0.1, seed=seed)
    X, miss = clean_series(X)
    return X, y, miss


def test_shape_features_scale_invariant():
    X, _, _ = _data()
    F1 = compute_features(X)
    F2 = compute_features(3.7 * X)
    ns = len(SHAPE_NAMES)
    assert np.allclose(F1[:, :ns], F2[:, :ns], rtol=1e-3, atol=1e-3)
    # level features DO shift
    assert not np.allclose(F1[:, ns], F2[:, ns])


def test_policy_budget_and_floor():
    rng = np.random.default_rng(0)
    s = rng.random(2000)
    for eps in (0.0, 0.3, 1.0):
        for tau in (0.0, 0.05):
            pol = fit_policy(s, b=0.05, eps=eps, tau=tau)
            p = pol.prob(s)
            assert abs(p.sum() - 0.05 * len(s)) < 1e-3 * len(s)
            assert p.min() >= eps * 0.05 - 1e-9 and p.max() <= 1 + 1e-9


def test_adversary_gain_and_mean():
    X, y, _ = _data()
    bank = X[y == 0]
    rng = np.random.default_rng(1)
    for kind in ("A1", "A2", "A3"):
        adv = Adversary(kind, X[y == 1], rng, bank)
        assert np.allclose(adv.stolen(0.0), 0.0)
        assert (adv.stolen(0.5) >= -1e-3).all()
    a1 = Adversary("A1", X[:50], rng)
    assert np.allclose(a1.report(0.4).mean(1), 0.6 * X[:50].mean(1), rtol=1e-4)
    a3 = Adversary("A3", X[y == 1][:30], rng, bank)
    assert np.allclose(a3.report(0.4).mean(1), 0.6 * X[y == 1][:30].mean(1), rtol=1e-3)


def test_floor_guarantee_vs_topk_under_adaptive_thieves():
    """Thief yang best-respond pada top-k murni lolos; lantai eps*b memaksa p>=eps*b."""
    rng = np.random.default_rng(0)
    n_th, R = 100, 5
    hs = rng.random(1000) * 0.5
    S = np.tile(np.linspace(0.9, 0.0, R), (n_th, 1))      # makin besar r -> skor makin rendah
    G = np.tile(np.linspace(0, 1, R), (n_th, 1))
    s0 = solve(S, G, hs, 0.05, 0.0, 0.0, 1.0, iters=10, rng=rng)
    s1 = solve(S, G, hs, 0.05, 0.5, 0.0, 1.0, iters=10, rng=rng)
    assert s1["p_th"].min() >= 0.5 * 0.05 - 1e-9
    assert (s0["p_th"] < 0.5 * 0.05).mean() > 0.5   # top-k murni: mayoritas pencuri tak terlihat


def test_deterrence_with_large_penalty():
    rng = np.random.default_rng(0)
    S = np.full((50, 4), 0.5)
    G = np.tile(np.linspace(0, 1, 4), (50, 1))
    hs = rng.random(500)
    sol = solve(S, G, hs, 0.2, 1.0, 0.0, phi=10.0, iters=5, rng=rng)  # p=0.2 > 1/11
    assert (sol["idx"] == 0).all()


def test_bounds_and_conformal():
    a = np.random.default_rng(0).normal(size=5000)
    assert mimicry_tv(a, a, 0.0) < 0.05
    assert mimicry_tv(a, a, 0.5) > 0.1
    assert ks_stat(a, a) == 0.0
    v = np.random.default_rng(1).random(99)
    assert conformal_upper(v, 0.1) <= 1.0
    assert hoeffding_upper(v, 0.1) > v.mean()
    assert len(ALL_NAMES) == compute_features(_data()[0]).shape[1]


def test_smoke_pipeline(tmp_path):
    from tip.config import load_config
    from tip.pipeline import stage_data, stage_experiments
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "smoke.yaml")
    cfg["name"] = "pytest"
    cfg["seeds"] = [0]
    cfg["data"]["synthetic"]["n_consumers"] = 1200
    cfg["experiment"].update({"budgets": [0.05], "penalties": [1.0], "eps_grid": [0.0, 1.0]})
    cfg["certificate"].update({"budget": 0.05, "trials": 20, "n_cal": [20]})
    cfg["deterrence"].update({"budgets": [0.05, 0.3], "penalties": [3.0], "eps_grid": [0.0, 1.0]})
    cfg["paths"] = {k: str(tmp_path / k) for k in ("artifacts", "results", "paper_generated")}
    from tip.config import run_dirs
    dirs = run_dirs(cfg)
    stage_data(cfg, dirs)
    stage_experiments(cfg, dirs)
    import pandas as pd
    e2 = pd.read_csv(dirs["results"] / "e2_game.csv")
    assert len(e2) > 0 and e2["ntl_undetected"].between(0, 1).all()
    e5 = pd.read_csv(dirs["results"] / "e5_certificate.csv")
    assert {"A2@0.6"} <= set(e5["cal"]) and e5["cov_conformal"].between(0, 1).all()
    e6 = pd.read_csv(dirs["results"] / "e6_deterrence.csv")
    # Prop. 3: eps*b*(1+phi) >= 1  ->  tidak ada pencuri aktif
    assert (e6.loc[e6["floor_index"] >= 1.0 - 1e-9, "deterred"] == 1.0).all()
    from tip.report import stage_report
    stage_report(cfg, dirs)
    assert (dirs["paper"] / "numbers.tex").exists()


def test_thief_sampling_level_matched():
    from tip.pipeline import _sample_thieves, LOGM
    rng = np.random.default_rng(0)
    n = 4000
    X = np.abs(rng.lognormal(0, 1, size=(n, 1)) * np.ones((1, 30))).astype(np.float32)
    X[:200] = 0.0                                      # konsumen berbeban nol
    y = np.zeros(n, dtype=np.int8)
    order = np.argsort(X[:, 0])
    y[order[-400:]] = 1                                # pencuri berlabel = level tinggi
    F = compute_features(X)
    tr = np.arange(0, n, 2)
    te_h = np.setdiff1d(np.arange(1, n, 2), np.where(y == 1)[0])
    ex = {"exclude_zero_load": True, "thief_sampling": "uniform"}
    u = _sample_thieves(te_h, 150, X, y, F, tr, ex, rng)
    assert (X[u].mean(1) > 0).all() and len(np.unique(u)) == 150
    m = _sample_thieves(te_h, 150, X, y, F, tr, {**ex, "thief_sampling": "level_matched"}, rng)
    assert np.median(F[m, LOGM]) > np.median(F[u, LOGM]) + 0.5
