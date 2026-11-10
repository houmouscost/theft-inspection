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


def test_change_features_scale_invariant_and_onset_sensitive():
    from tip.features import CHANGE_NAMES
    X, _, _ = _data(T=360)
    F1, F2 = compute_features(X), compute_features(2.5 * X)
    k = [ALL_NAMES.index(c) for c in CHANGE_NAMES]
    assert np.allclose(F1[:, k], F2[:, k], rtol=1e-3, atol=1e-3)
    adv = Adversary("A4-0", X[:50], np.random.default_rng(0), onset=np.full(50, 180))
    F3 = compute_features(adv.report(0.6))
    drop = ALL_NAMES.index("chg_max_drop")
    assert (F3[:, drop] > F1[:50, drop] + 0.3).mean() > 0.9     # langkah 60% terlihat sebagai penurunan


def test_onset_adversary_ramp():
    X = np.ones((3, 100), dtype=np.float32)
    adv = Adversary("A4-20", X, np.random.default_rng(0), onset=np.array([10, 50, 90]))
    Y = adv.report(0.5)
    assert np.allclose(Y[:, :10], 1.0) and np.isclose(Y[0, 40], 0.5) and np.isclose(Y[1, 60], 0.75)
    assert np.allclose(adv.stolen(0.0), 0.0) and adv.stolen(0.5)[0] > adv.stolen(0.5)[2]


def test_local_policy_never_worse_than_uniform_and_meets_budget():
    from tip import stackelberg as sb
    rng = np.random.default_rng(0)
    n, J, K = 60, 6, 9
    Kth = rng.integers(0, K, size=(n, J)); Kth[:, 0] = rng.integers(0, K, size=n)
    G = np.sort(rng.random((n, J)), axis=1); G[:, 0] = 0.0
    Kh = rng.integers(0, K, size=600)
    nh = np.bincount(Kh, minlength=K).astype(float)
    for phi in (1.0, 3.0):
        p, info = sb.fit_local(Kth, G, Kh, K, 0.05, phi, max_iter=10)
        f, used = sb._objective(p, Kth, G, nh, 1 + phi)
        f_u, _ = sb._objective(np.full(K, 0.05), Kth, G, nh, 1 + phi)
        assert f <= f_u + 1e-9 and used <= 0.05 * (600 + n) * (1 + 1e-6)


def test_quantal_limits():
    from tip.game import solve_quantal
    rng = np.random.default_rng(0)
    n, R = 40, 5
    hs = rng.random(400) * 0.5
    S = np.tile(np.linspace(0.9, 0.0, R), (n, 1))
    G = np.tile(np.linspace(0, 1, R), (n, 1))
    q0 = solve_quantal(S, G, hs, 0.05, 0.0, 0.0, 1.0, lam=0.0, iters=3, rng=rng)
    assert np.allclose(q0["pi"], 1.0 / R)
    qh = solve_quantal(S, G, hs, 0.05, 0.0, 0.0, 1.0, lam=200.0, iters=5, rng=rng)
    assert (qh["pi"][:, -1] > 0.9).all()           # hampir best-response: curi maksimum, skor terendah


def test_diagnostics_selection_and_hygiene(tmp_path):
    """Desil konsumsi menghitung benar; tabel higiene mengenali baris nol, duplikat, label bertentangan, urutan berkas."""
    import pandas as pd
    from tip import diagnostics as D
    rng = np.random.default_rng(0)
    X = rng.gamma(2.0, 1.0, size=(1000, 30)).astype(np.float32) * np.linspace(0.2, 3, 1000)[:, None]
    y = (rng.random(1000) < 0.1 + 0.2 * (np.arange(1000) > 900)).astype(np.int8)
    miss = rng.random(1000).astype(np.float32)
    t = D.selection_table(X, y, miss)
    dec = t[t.dimension == "consumption_decile"]
    assert len(dec) == 10 and dec.n.sum() == 1000 and dec.thieves.sum() == y.sum()
    assert dec.rate.iloc[-1] > dec.rate.iloc[0]                       # laju naik menurut konsumsi pada data buatan
    assert t[t.dimension == "missing_bin"].n.sum() == 1000
    # berkas mentah mini: 2 baris nol identik (label bertentangan), 1 baris positif, 1 baris semua-NaN
    cols = [f"2014/01/{d:02d}" for d in range(1, 6)]
    df = pd.DataFrame([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [np.nan] * 5], columns=cols)
    df.insert(0, "FLAG", [1, 0, 1, 1]); df.insert(0, "CONS_NO", ["a", "b", "c", "d"])
    p = tmp_path / "mini.csv"; df.to_csv(p, index=False)
    h = dict(zip(*D.hygiene_table(str(p)).values.T))
    assert h["n_raw_rows"] == 4 and h["n_allmissing_rows"] == 1 and h["n_allmissing_thieves"] == 1
    assert h["n_allzero"] == 2 and h["n_rows_in_dup_groups"] == 2 and h["n_dup_groups_conflict"] == 1
    assert h["n_dup_groups_positive"] == 0 and bool(h["file_sorted_thieves_first"]) is False
