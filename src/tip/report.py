"""Hasilkan gambar (PDF), tabel LaTeX, dan makro angka (numbers.tex) dari CSV hasil.

Tidak ada angka yang ditulis manual di paper: setiap angka di teks memakai makro dari
`generated/numbers.tex`, setiap tabel/gambar di-\\input dari `generated/`.
Label keluaran berbahasa Inggris (paper berbahasa Inggris); komentar kode tetap Indonesia.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})
COL = {"gbm_level": "#1f77b4", "gbm_shape": "#d62728", "gbm_level_adv": "#2ca02c",
       "A1": "#1f77b4", "A2": "#ff7f0e", "A3": "#d62728"}
LAB = {"gbm_level": "GBM (shape+level)", "gbm_shape": "GBM (shape only)",
       "gbm_level_adv": "GBM + adv.\\ training", "gbm_level_miss": "GBM + missingness"}
LAB_PLAIN = {k: v.replace("\\ ", " ") for k, v in LAB.items()}


# --------------------------------------------------------------------------- utilitas
def _tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _pm(m, s, scale=1.0, d=2):
    return f"{m * scale:.{d}f}$\\pm${s * scale:.{d}f}"


def _agg(df, by, val):
    g = df.groupby(by)[val].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def _mid(xs):
    xs = sorted(set(xs))
    return xs[len(xs) // 2]


def _pct(x):
    """0.02 -> '2', 0.025 -> '2.5'"""
    return f"{100 * x:g}"


def _pool_lab(spec: str) -> str:
    if "@" in spec:
        k, r = spec.split("@")
        return f"{k} static ($r{{=}}{float(r):g}$)"
    return f"{spec} adaptive"


def _save_tab(tex: str, name: str, dirs: dict):
    for d in (dirs["results"], dirs["paper"]):
        (d / name).write_text(tex + "\n")


def _save_fig(fig, name: str, dirs: dict):
    for d in (dirs["figures"], dirs["paper"]):
        fig.savefig(d / name)
    plt.close(fig)


def _read(path: Path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 5 else pd.DataFrame()


class Macros:
    """Kumpulkan \\newcommand untuk angka di teks. Nama makro hanya huruf (aturan TeX)."""

    def __init__(self):
        self.items: dict[str, str] = {}

    def add(self, name: str, value):
        assert name.isalpha(), name
        self.items[name] = str(value)

    def safe(self, name: str, fn):
        try:
            self.add(name, fn())
        except Exception:   # angka tidak tersedia pada config ini (mis. uji asap) -> penanda jelas
            self.add(name, r"\textbf{??}")

    def tex(self) -> str:
        lines = ["% Dihasilkan otomatis oleh tip.report -- JANGAN diedit manual."]
        lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(self.items.items())]
        return "\n".join(lines) + "\n"


def _escape_share(df):
    """Fraksi energi curian yang lolos = ntl_undetected / stolen_total (per baris)."""
    return df["ntl_undetected"] / df["stolen_total"].where(df["stolen_total"] > 0)


# --------------------------------------------------------------------------- laporan
def stage_report(cfg: dict, dirs: dict):
    R = dirs["results"]
    e0, e1, e2 = _read(R / "e0_data.csv"), _read(R / "e1_detector.csv"), _read(R / "e2_game.csv")
    e4, e5, e6 = _read(R / "e4_mimicry.csv"), _read(R / "e5_certificate.csv"), _read(R / "e6_deterrence.csv")
    rep = cfg.get("report") or {}
    fs_dir = Path(rep["finite_sample"]) if rep.get("finite_sample") else None
    if fs_dir is not None and len(_read(fs_dir / "e4_mimicry.csv")):   # estimator sampel-hingga (configs/diag_e4e5.yaml)
        e4, e5 = _read(fs_dir / "e4_mimicry.csv"), _read(fs_dir / "e5_certificate.csv")
    if len(e4) and "ba_cf" not in e4.columns:     # CSV lama tanpa estimator sampel-hingga
        e4 = e4.assign(tv_kde=e4.tv_bound, tv_kde_lo=e4.tv_bound, tv_kde_hi=e4.tv_bound, ba_cf=e4.balacc_emp, ba_cf_se=0.0, ks_null=0.0)
    sens_dir = Path(rep["br_sensitivity"]) if rep.get("br_sensitivity") else None
    s2_dir = Path(rep["level_matched"]) if rep.get("level_matched") else None
    e2_sens = _read(sens_dir / "e2_game.csv") if sens_dir else pd.DataFrame()
    e2_s2 = _read(s2_dir / "e2_game.csv") if s2_dir else pd.DataFrame()
    e0_s2 = _read(s2_dir / "e0_data.csv") if s2_dir else pd.DataFrame()

    ex = cfg["experiment"]
    b_mid, phi_mid, tau0 = _mid(ex["budgets"]), _mid(ex["penalties"]), min(ex["tau_grid"])
    kinds = list(ex["adversaries"])
    r_stat = sorted(ex["r_static"])
    eps_grid = sorted(ex["eps_grid"])
    e_lo, e_hi = eps_grid[0], eps_grid[-1]
    A = e2[e2["mode"] == "adaptive"]
    Sx = e2[e2["mode"] == "static"]
    ad = A[(A.budget == b_mid) & (A.penalty == phi_mid) & (A.tau == tau0)]
    st = Sx[(Sx.budget == b_mid) & (Sx.tau == tau0)]
    main_v = "gbm_level" if "gbm_level" in set(ad["variant"]) else ad["variant"].iloc[0]
    game_vars = [v for v in ["gbm_level", "gbm_shape", "gbm_level_adv"] if v in set(e2["variant"])]
    M = Macros()

    # ---------------- konfigurasi & data
    M.add("rBmid", _pct(b_mid))
    M.add("rPhiMid", f"{phi_mid:g}")
    M.add("rBudgets", ", ".join(_pct(b) for b in sorted(ex["budgets"])))
    M.add("rPenalties", ", ".join(f"{p:g}" for p in sorted(ex["penalties"])))
    M.add("rNSeeds", str(len(cfg["seeds"])))
    M.add("rBrIters", str(ex["br_iters"]))
    if len(e0):
        r0 = e0.iloc[0]
        M.add("rNConsumers", f"{int(r0.n_consumers):,}".replace(",", "{,}"))
        M.add("rNDays", f"{int(r0.n_days):,}".replace(",", "{,}"))
        M.add("rNLabelled", f"{int(r0.n_labelled_thieves):,}".replace(",", "{,}"))
        M.add("rTheftRate", f"{100 * r0.n_labelled_thieves / r0.n_consumers:.2f}")
        M.add("rNanRate", f"{100 * r0.nan_rate:.1f}")
        M.add("rZeroLoadHonest", f"{100 * r0.zero_load_frac_honest:.1f}")
        M.add("rMedLoadHonest", f"{r0.median_load_honest:.2f}")
        M.add("rMedLoadThief", f"{r0.median_load_labelled_thief:.2f}")
        M.add("rLoadRatio", f"{r0.median_load_labelled_thief / r0.median_load_honest:.2f}")
        M.add("rNSOneThieves", f"{e0.n_s1_thieves.mean():.0f}")
        M.add("rNTestHonest", f"{e0.n_test_honest.mean():,.0f}".replace(",", "{,}"))
        M.add("rMedLoadSOne", f"{e0.median_load_s1_thief.mean():.2f}")
    if len(e0_s2):
        M.add("rMedLoadSTwo", f"{e0_s2.median_load_s1_thief.mean():.2f}")

    # ---------------- T1: detektor (S0)
    bs = sorted(e1["budget"].unique())
    g1 = e1.groupby(["variant", "budget"]).agg(auc=("auc", "mean"), auc_sd=("auc", "std"),
                                               pr=("pr_auc", "mean"), rec=("recall_at_b", "mean"),
                                               prec=("precision_at_b", "mean")).reset_index()
    head = " & ".join(f"R@{_pct(b)}\\%" for b in bs)
    lines = [r"\begin{tabular}{l" + "c" * (3 + len(bs)) + "}", r"\toprule",
             rf"Detector & AUC & PR-AUC & {head} & P@{_pct(b_mid)}\% \\", r"\midrule"]
    order = [v for v in ["gbm_level", "gbm_shape", "gbm_level_adv", "gbm_level_miss"] if v in set(g1.variant)]
    for v in order:
        gv = g1[g1.variant == v].set_index("budget")
        r = gv.loc[b_mid]
        recs = " & ".join(f"{gv.loc[b, 'rec']:.3f}" for b in bs)
        lines.append(f"{LAB.get(v, _tex(v))} & {_pm(r.auc, r.auc_sd, 1, 3)} & {r.pr:.3f} & {recs} & {r.prec:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _save_tab("\n".join(lines), "tab_detector.tex", dirs)
    gm = g1[g1.budget == b_mid].set_index("variant")
    name_map = {"gbm_level": "Level", "gbm_shape": "Shape", "gbm_level_adv": "Adv", "gbm_level_miss": "Miss"}
    for v, nm in name_map.items():
        if v in gm.index:
            M.add(f"rAuc{nm}", f"{gm.loc[v, 'auc']:.3f}")
            M.add(f"rAucSd{nm}", f"{gm.loc[v, 'auc_sd']:.3f}")
            M.add(f"rPrAuc{nm}", f"{gm.loc[v, 'pr']:.3f}")
            M.add(f"rRec{nm}", f"{gm.loc[v, 'rec']:.3f}")
            M.add(f"rPrec{nm}", f"{gm.loc[v, 'prec']:.3f}")
    M.safe("rMissGain", lambda: f"{gm.loc['gbm_level_miss', 'auc'] - gm.loc['gbm_level', 'auc']:.3f}")
    M.safe("rBaseRate", lambda: f"{100 * e0.iloc[0].n_labelled_thieves / e0.iloc[0].n_consumers:.1f}")

    # ---------------- T2: tabel utama (NTL lolos, % beban sistem)
    ncol = len(kinds) * (1 + len(r_stat))
    lines = [r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{1 + len(r_stat)}}}{{c}}{{{k}}}" for k in kinds) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + i * (1 + len(r_stat))}-{1 + (i + 1) * (1 + len(r_stat))}}}"
                     for i in range(len(kinds))),
             "Policy & " + " & ".join((["Adaptive"] + [f"Static $r{{=}}{r:g}$" for r in r_stat]) * len(kinds)) + r" \\",
             r"\midrule"]

    def _cells(sub_a, sub_s, col):
        cells = []
        for k in kinds:
            a = sub_a[sub_a.adversary == k][col]
            cells.append(_pm(a.mean(), a.std(ddof=0), 100))
            for rs in r_stat:
                s = sub_s[(sub_s.adversary == k) & np.isclose(sub_s.r_static, rs)][col]
                cells.append(_pm(s.mean(), s.std(ddof=0), 100))
        return cells

    for eps in eps_grid:
        lab = ("Pure random ($\\varepsilon{=}1$)" if eps == 1 else
               ("Pure top-$k$ ($\\varepsilon{=}0$)" if eps == 0 else f"Mixed $\\varepsilon{{=}}{eps:g}$"))
        cells = _cells(ad[(ad.variant == main_v) & (ad.eps == eps)], st[(st.variant == main_v) & (st.eps == eps)],
                       "ntl_undetected")
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    cells = _cells(ad[(ad.variant == main_v) & (ad.eps == e_lo)], st[(st.variant == main_v) & (st.eps == e_lo)],
                   "stolen_total")
    lines += [r"\midrule", "Total stolen & " + " & ".join(cells) + r" \\", r"\bottomrule", r"\end{tabular}"]
    _save_tab("\n".join(lines), "tab_main.tex", dirs)

    # makro tabel utama
    adm = ad[ad.variant == main_v]
    for k in kinds:
        kk = {"A1": "One", "A2": "Two", "A3": "Three"}.get(k, k)
        M.safe(f"rNtl{kk}Lo", lambda k=k: f"{100 * adm[(adm.adversary == k) & (adm.eps == e_lo)].ntl_undetected.mean():.2f}")
        M.safe(f"rNtl{kk}Hi", lambda k=k: f"{100 * adm[(adm.adversary == k) & (adm.eps == e_hi)].ntl_undetected.mean():.2f}")
        M.safe(f"rStolen{kk}", lambda k=k: f"{100 * adm[(adm.adversary == k) & (adm.eps == e_lo)].stolen_total.mean():.2f}")
        M.safe(f"rMeanR{kk}", lambda k=k: f"{adm[(adm.adversary == k) & (adm.eps == e_lo)].mean_r.mean():.2f}")
    M.safe("rStolenOneHi", lambda: f"{100 * adm[(adm.adversary == 'A1') & (adm.eps == e_hi)].stolen_total.mean():.2f}")
    esc = adm.assign(esc=_escape_share(adm))
    M.safe("rEscapeMin", lambda: f"{100 * esc.groupby(['adversary', 'eps']).esc.mean().min():.1f}")
    M.safe("rRecallAdaptMax", lambda: f"{100 * adm.groupby(['adversary', 'eps']).recall.mean().max():.2f}")
    M.safe("rDeterredAdaptMax", lambda: f"{100 * adm.groupby(['adversary', 'eps']).deterred.mean().max():.1f}")
    # efek eps maksimum di seluruh grid adaptif (gbm_level, tau=0): NTL(eps_lo) - NTL(eps_hi), poin persen
    Am = A[(A.variant == main_v) & (A.tau == tau0)]
    pv = Am.groupby(["budget", "penalty", "adversary", "eps"]).ntl_undetected.mean().unstack("eps")
    M.safe("rEpsGainMax", lambda: f"{100 * (pv[e_lo] - pv[e_hi]).max():.2f}")
    M.safe("rEpsGainMid", lambda: f"{100 * (pv[e_lo] - pv[e_hi]).loc[(b_mid, phi_mid)].max():.2f}")
    # efek phi maksimum
    pp = Am.groupby(["budget", "eps", "adversary", "penalty"]).ntl_undetected.mean().unstack("penalty")
    M.safe("rPhiEffectMax", lambda: f"{100 * (pp.max(1) - pp.min(1)).max():.2f}")
    M.safe("rMaxBOnePlusPhi", lambda: f"{max(ex['budgets']) * (1 + max(ex['penalties'])):g}")
    # efek tau maksimum
    pt = A[A.variant == main_v].groupby(["budget", "penalty", "eps", "adversary", "tau"]).ntl_undetected.mean().unstack("tau")
    M.safe("rTauEffectMax", lambda: f"{100 * (pt.max(1) - pt.min(1)).max():.2f}")

    # statis: recall (jumlah) & pemulihan energi vs acak
    stm = st[st.variant == main_v].assign(rec_e=lambda d: d.ntl_recovered / d.stolen_total)
    sg = stm.groupby(["adversary", "r_static", "eps"])[["recall", "rec_e"]].mean()
    M.safe("rStatRecallOneLow", lambda: f"{100 * sg.loc[('A1', r_stat[0], e_lo), 'recall']:.2f}")
    M.safe("rStatRecallThreeLow", lambda: f"{100 * sg.loc[('A3', r_stat[0], e_lo), 'recall']:.2f}")
    M.safe("rStatRecEOneLow", lambda: f"{100 * sg.loc[('A1', r_stat[0], e_lo), 'rec_e']:.2f}")
    M.safe("rStatRecEOneHigh", lambda: f"{100 * sg.loc[('A1', r_stat[-1], e_lo), 'rec_e']:.2f}")
    M.safe("rStatRecallBelowRandom", lambda: str(int((sg.xs(e_lo, level="eps")["recall"] < b_mid).sum())))
    M.safe("rStatCells", lambda: str(len(sg.xs(e_lo, level="eps"))))
    M.add("rRStatLo", f"{r_stat[0]:g}")
    M.add("rRStatHi", f"{r_stat[-1]:g}")
    # Prop. 2: R(eps) >= (1-eps) R(0) utk pencuri statis, semua sel
    S0 = Sx[Sx.eps == e_lo].set_index(["seed", "variant", "adversary", "budget", "tau", "r_static"]).ntl_recovered
    chk = Sx[Sx.eps > e_lo].set_index(["seed", "variant", "adversary", "budget", "tau", "r_static"])
    ok = chk.ntl_recovered >= (1 - chk.eps) * S0.reindex(chk.index) - 1e-12
    M.add("rPropTwoOk", str(int(ok.sum())))
    M.add("rPropTwoN", str(int(len(ok))))
    # biaya konsistensi: statis, eps_hi vs eps_lo, rata2 semua adversary & r
    M.safe("rConsistCost", lambda: f"{100 * (stm[stm.eps == e_hi].ntl_undetected.mean() - stm[stm.eps == e_lo].ntl_undetected.mean()):.2f}")

    # ---------------- T3: per-varian (pencuri statis vs adaptif, eps=0)
    ad0 = ad[ad.eps == e_lo]
    st0 = st[(st.eps == e_lo) & np.isclose(st.r_static, r_stat[-1])]
    lines = [r"\begin{tabular}{l" + "ccc" * len(kinds) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{k}}}" for k in kinds) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(kinds))),
             "Detector & " + " & ".join([f"Stat.\\ rec.", r"$\bar r^\star$", "Adapt.\\ rec."] * len(kinds)) + r" \\",
             r"\midrule"]
    for v in game_vars:
        cells = []
        for k in kinds:
            s = st0[(st0.variant == v) & (st0.adversary == k)]
            a = ad0[(ad0.variant == v) & (ad0.adversary == k)]
            cells += [f"{100 * (s.ntl_recovered / s.stolen_total).mean():.1f}",
                      f"{a.mean_r.mean():.2f}",
                      f"{100 * (a.ntl_recovered / a.stolen_total).mean():.1f}"]
        lines.append(f"{LAB.get(v, _tex(v))} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", "Random ($\\varepsilon{=}1$) & " +
              " & ".join([f"{_pct(b_mid)}", "--", f"{_pct(b_mid)}"] * len(kinds)) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    _save_tab("\n".join(lines), "tab_variants.tex", dirs)
    for v, nm in name_map.items():
        for k in kinds:
            kk = {"A1": "One", "A2": "Two", "A3": "Three"}.get(k, k)
            s = st0[(st0.variant == v) & (st0.adversary == k)]
            a = ad0[(ad0.variant == v) & (ad0.adversary == k)]
            if len(s) and len(a):
                M.add(f"rStatRec{nm}{kk}", f"{100 * (s.ntl_recovered / s.stolen_total).mean():.1f}")
                M.add(f"rAdaptRec{nm}{kk}", f"{100 * (a.ntl_recovered / a.stolen_total).mean():.1f}")
                M.add(f"rAdaptR{nm}{kk}", f"{a.mean_r.mean():.2f}")
                M.add(f"rAdaptNtl{nm}{kk}", f"{100 * a.ntl_undetected.mean():.2f}")

    # ---------------- T4: anggaran x denda (NTL lolos terburuk atas adversary)
    pens = sorted(ex["penalties"])
    wc = Am.groupby(["budget", "penalty", "eps", "seed", "adversary"]).ntl_undetected.mean().reset_index() \
        .groupby(["budget", "penalty", "eps", "seed"]).ntl_undetected.max().reset_index()
    lines = [r"\begin{tabular}{l" + "cc" * len(pens) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$\varphi{{=}}{p:g}$}}" for p in pens) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(pens))),
             "$b$ & " + " & ".join([f"$\\varepsilon{{=}}{e_lo:g}$", f"$\\varepsilon{{=}}{e_hi:g}$"] * len(pens)) + r" \\",
             r"\midrule"]
    for b in sorted(ex["budgets"]):
        cells = []
        for p in pens:
            for e in (e_lo, e_hi):
                v = wc[(wc.budget == b) & (wc.penalty == p) & (wc.eps == e)].ntl_undetected
                cells.append(_pm(v.mean(), v.std(ddof=0), 100))
        lines.append(f"{_pct(b)}\\% & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _save_tab("\n".join(lines), "tab_budget.tex", dirs)

    # ---------------- T5: konvergensi + sensitivitas br_iters
    Aall = A.copy()
    conv = Aall.groupby(["variant", "adversary"]).converged.mean()
    M.add("rConvMain", f"{100 * Aall.converged.mean():.1f}")
    M.add("rConvAll", f"{100 * e2.converged.mean():.1f}")
    worst = conv.idxmin()
    M.add("rConvWorst", f"{100 * conv.min():.1f}")
    M.add("rConvWorstCell", f"{LAB_PLAIN.get(worst[0], worst[0])} vs {worst[1]}")
    have_sens = len(e2_sens) > 0
    key = ["seed", "variant", "adversary", "mode", "r_static", "budget", "penalty", "eps", "tau"]
    if have_sens:
        As = e2_sens[e2_sens["mode"] == "adaptive"]
        conv_s = As.groupby(["variant", "adversary"]).converged.mean()
        mm = Aall.merge(As, on=key, suffixes=("", "_s"))
        dd = (mm.ntl_undetected_s - mm.ntl_undetected).abs()
        dmax = dd.groupby([mm.variant, mm.adversary]).max()
        M.add("rConvSens", f"{100 * As.converged.mean():.1f}")
        M.add("rSensBrIters", str(int(_read_cfg_br(sens_dir, default=24))))
        M.add("rSensMaxDiff", f"{100 * dd.max():.2f}")
        M.add("rSensMeanDiff", f"{100 * dd.mean():.4f}")
    lines = [r"\begin{tabular}{llccc}" if have_sens else r"\begin{tabular}{llc}", r"\toprule",
             (rf"Detector & Adv. & Conv.\ ($K{{=}}{ex['br_iters']}$) & Conv.\ ($K{{=}}{_read_cfg_br(sens_dir, 24)}$) & max $|\Delta\NTL|$ (pp) \\"
              if have_sens else rf"Detector & Adv. & Conv.\ ($K{{=}}{ex['br_iters']}$) \\"), r"\midrule"]
    for (v, k), c in conv.items():
        row = f"{LAB.get(v, _tex(v))} & {k} & {100 * c:.1f}\\%"
        if have_sens:
            row += f" & {100 * conv_s.get((v, k), np.nan):.1f}\\% & {100 * dmax.get((v, k), np.nan):.3f}"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _save_tab("\n".join(lines), "tab_conv.tex", dirs)

    # ---------------- T6: sertifikat
    if len(e5):
        ncs = sorted(e5.n_cal_req.unique())
        n_show = max(ncs)
        has_tv = "tv_pools_bin" in e5.columns
        agg = dict(mc=("mean_miss_cal", "mean"), uh=("ub_hoeffding", "mean"), uc=("ub_conformal", "mean"),
                   md=("mean_miss_dep", "mean"), ch=("cov_hoeffding", "mean"), cc=("cov_conformal", "mean"),
                   n=("n_cal", "mean"))
        if has_tv:
            agg.update(ks=("ks_pools", "mean"), ua=("ub_hoeffding_adj", "mean"),
                       ca=("cov_hoeffding_adj", "mean"), clb=("cov_conformal_lb", "mean"))
        g = e5[e5.n_cal_req == n_show].groupby(["variant", "cal", "dep"], sort=False).agg(**agg).reset_index()
        lines = [r"\begin{tabular}{lllcccccc" + ("ccc" if has_tv else "") + "}", r"\toprule",
                 r"Detector & Calibration & Deployment & $\bar m_{\mathrm{cal}}$ & Hoeff.\ UB & $\hat q$ & "
                 r"$\bar m_{\mathrm{dep}}$ & Cov.\ (H) & Cov.\ (C)" + (r" & $\mathrm{KS}$ & UB$+$KS & Cov.\ (H, adj.)" if has_tv else "") + r" \\", r"\midrule"]
        prev = None
        for _, r in g.iterrows():
            if prev is not None and r.variant != prev:
                lines.append(r"\midrule")
            vlab = LAB.get(r.variant, _tex(r.variant)) if r.variant != prev else ""
            prev = r.variant
            extra = f" & {r.ks:.2f} & {r.ua:.3f} & {r.ca:.2f}" if has_tv else ""
            lines.append(f"{vlab} & {_pool_lab(r.cal)} & {_pool_lab(r.dep)} & {r.mc:.3f} & {r.uh:.3f} & "
                         f"{r.uc:.3f} & {r.md:.3f} & {r.ch:.2f} & {r.cc:.2f}{extra} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        _save_tab("\n".join(lines), "tab_cert.tex", dirs)
        if has_tv:      # tabel ringkas untuk badan paper: pasangan kunci (A2 statis/adaptif) untuk dua detektor
            short = [("gbm_level", "A2@0.6", "A2@0.6"), ("gbm_level", "A2@0.6", "A2"),
                     ("gbm_level_adv", "A2@0.6", "A2@0.6"), ("gbm_level_adv", "A2@0.6", "A2"), ("gbm_level_adv", "A2", "A2")]
            pl = lambda c, d: f"{'static' if '@' in c else 'adaptive'} $\\to$ {'static' if '@' in d else 'adaptive'}"
            gi_ = g.set_index(["variant", "cal", "dep"])
            sl = [r"\begin{tabular}{llcccccc}", r"\toprule",
                  r"Detector & Calibr.\ $\to$ deploy. & $\bar m_{\mathrm{cal}}$ & Hoeff.\ UB & $\bar m_{\mathrm{dep}}$ & Cov.\ (H) & KS & UB$+$KS \\", r"\midrule"]
            pv = None
            for v, c_, d_ in short:
                if (v, c_, d_) not in gi_.index:
                    continue
                r_ = gi_.loc[(v, c_, d_)]
                if pv is not None and v != pv:
                    sl.append(r"\midrule")
                sl.append(f"{LAB.get(v, _tex(v)) if v != pv else ''} & {pl(c_, d_)} & {r_.mc:.3f} & {r_.uh:.3f} & {r_.md:.3f} & {r_.ch:.2f} & {r_.ks:.2f} & {r_.ua:.3f} \\\\")
                pv = v
            sl += [r"\bottomrule", r"\end{tabular}"]
            _save_tab("\n".join(sl), "tab_cert_short.tex", dirs)
        c = cfg["certificate"]
        M.add("rCertB", _pct(c["budget"]))
        M.add("rCertEps", f"{c['eps']:g}")
        M.add("rCertPhi", f"{c['penalty']:g}")
        M.add("rCertAlpha", f"{c['alpha']:g}")
        M.add("rCertTarget", f"{1 - c['alpha']:g}")
        M.add("rCertNcal", str(int(n_show)))
        M.add("rCertNcals", ", ".join(str(int(x)) for x in ncs))
        M.add("rCertFloorMiss", f"{1 - c['eps'] * c['budget']:.3f}")
        gi = g.set_index(["variant", "cal", "dep"])
        for v, nm in name_map.items():
            if v not in set(e5.variant):
                continue

            def _g(cal, dep, col, v=v):
                return gi.loc[(v, cal, dep), col]
            M.safe(f"rCert{nm}StatMiss", lambda _g=_g: f"{_g('A2@0.6', 'A2@0.6', 'mc'):.3f}")
            M.safe(f"rCert{nm}StatUB", lambda _g=_g: f"{_g('A2@0.6', 'A2@0.6', 'uh'):.3f}")
            M.safe(f"rCert{nm}StatCov", lambda _g=_g: f"{_g('A2@0.6', 'A2@0.6', 'ch'):.2f}")
            M.safe(f"rCert{nm}ShiftUB", lambda _g=_g: f"{_g('A2@0.6', 'A2', 'uh'):.3f}")
            M.safe(f"rCert{nm}ShiftMiss", lambda _g=_g: f"{_g('A2@0.6', 'A2', 'md'):.3f}")
            M.safe(f"rCert{nm}ShiftCovH", lambda _g=_g: f"{_g('A2@0.6', 'A2', 'ch'):.2f}")
            M.safe(f"rCert{nm}ShiftCovC", lambda _g=_g: f"{_g('A2@0.6', 'A2', 'cc'):.2f}")
            M.safe(f"rCert{nm}AdaptMiss", lambda _g=_g: f"{_g('A2', 'A2', 'mc'):.3f}")
            M.safe(f"rCert{nm}AdaptUB", lambda _g=_g: f"{_g('A2', 'A2', 'uh'):.3f}")
        # cakupan minimum pada pasangan statis->adaptif, semua detektor & n_cal
        sh = e5[e5.cal.str.contains("@") & ~e5.dep.str.contains("@")]
        M.safe("rCertShiftCovMax", lambda: f"{sh.groupby(['variant', 'cal', 'dep', 'n_cal_req']).cov_hoeffding.mean().max():.2f}")
        if has_tv:
            shd = e5[e5.cal.str.contains("@") & ~e5.dep.str.contains("@") & (e5.n_cal_req == n_show)]
            gm = shd.groupby(["variant", "cal", "dep"]).mean(numeric_only=True)
            for v, nm in name_map.items():
                if v in set(shd.variant):
                    M.safe(f"rCert{nm}ShiftKs", lambda v=v: f"{gm.loc[(v, 'A2@0.6', 'A2'), 'ks_pools']:.2f}")
                    M.safe(f"rCert{nm}ShiftUBAdj", lambda v=v: f"{gm.loc[(v, 'A2@0.6', 'A2'), 'ub_hoeffding_adj']:.3f}")
                    M.safe(f"rCert{nm}ShiftCovAdj", lambda v=v: f"{gm.loc[(v, 'A2@0.6', 'A2'), 'cov_hoeffding_adj']:.2f}")
            alls = e5[e5.cal != e5.dep]
            M.safe("rCertShiftKsMin", lambda: f"{alls.groupby(['variant', 'cal', 'dep', 'n_cal_req']).ks_pools.mean().min():.2f}")
            M.safe("rCertShiftKsMax", lambda: f"{alls.groupby(['variant', 'cal', 'dep', 'n_cal_req']).ks_pools.mean().max():.2f}")
            M.safe("rCertShiftAdjCovMin", lambda: f"{alls.groupby(['variant', 'cal', 'dep', 'n_cal_req']).cov_hoeffding_adj.mean().min():.2f}")
            # celah rata-rata miss <= KS pada setiap baris (dasar penyesuaian Hoeffding); cakupan konformal >= 1-alpha-KS
            gap_ok = ((e5.mean_miss_dep - e5.mean_miss_cal) <= e5.ks_pools + 1e-9)
            M.add("rCertGapOk", str(int(gap_ok.sum())))
            M.add("rCertGapN", str(int(len(e5))))
            conf_ok = (e5.cov_conformal >= e5.cov_conformal_lb - 1e-9)
            M.add("rCertConfLbOk", str(int(conf_ok.sum())))
            M.add("rCertConfLbN", str(int(len(e5))))
        same = e5[e5.cal == e5.dep]
        M.safe("rCertSameCovMin", lambda: f"{same.groupby(['variant', 'cal', 'n_cal_req']).cov_hoeffding.mean().min():.2f}")
        M.safe("rCertSameCovCMin", lambda: f"{same.groupby(['variant', 'cal', 'n_cal_req']).cov_conformal.mean().min():.2f}")

    # ---------------- T7: skenario S2 (level-matched) vs S1
    if len(e2_s2):
        a2 = e2_s2[(e2_s2["mode"] == "adaptive") & (e2_s2.variant == main_v) & (e2_s2.budget == b_mid)
                   & (e2_s2.penalty == phi_mid) & (e2_s2.tau == tau0)]
        s2s = e2_s2[(e2_s2["mode"] == "static") & (e2_s2.variant == main_v) & (e2_s2.budget == b_mid)
                    & (e2_s2.tau == tau0) & (e2_s2.eps == e_lo)]
        s1s = st[(st.variant == main_v) & (st.eps == e_lo)]
        cols = [f"Stolen, $\\varepsilon{{=}}{e_lo:g}$", f"$\\NTL^{{\\mathrm{{und}}}}$, $\\varepsilon{{=}}{e_lo:g}$",
                f"Stolen, $\\varepsilon{{=}}{e_hi:g}$", f"$\\NTL^{{\\mathrm{{und}}}}$, $\\varepsilon{{=}}{e_hi:g}$",
                f"Stat.\\ rec.\\ $r{{=}}{r_stat[0]:g}$"]
        lines = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
                 "Scenario & Adv. & " + " & ".join(cols) + r" \\", r"\midrule"]
        for scen, aa, ss in (("S1 (level-typical)", adm, s1s), ("S2 (level-matched)", a2, s2s)):
            for i, k in enumerate(kinds):
                x0 = aa[(aa.adversary == k) & (aa.eps == e_lo)]
                x1 = aa[(aa.adversary == k) & (aa.eps == e_hi)]
                sr = ss[(ss.adversary == k) & np.isclose(ss.r_static, r_stat[0])]
                lines.append(f"{scen if i == 0 else ''} & {k} & {_pm(x0.stolen_total.mean(), x0.stolen_total.std(ddof=0), 100)} & "
                             f"{_pm(x0.ntl_undetected.mean(), x0.ntl_undetected.std(ddof=0), 100)} & "
                             f"{_pm(x1.stolen_total.mean(), x1.stolen_total.std(ddof=0), 100)} & "
                             f"{_pm(x1.ntl_undetected.mean(), x1.ntl_undetected.std(ddof=0), 100)} & "
                             f"{100 * (sr.ntl_recovered / sr.stolen_total).mean():.1f} \\\\")
            if scen.startswith("S1"):
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}"]
        _save_tab("\n".join(lines), "tab_s2.tex", dirs)
        esc2 = a2.assign(esc=_escape_share(a2))
        M.safe("rSTwoEscapeMin", lambda: f"{100 * esc2.groupby(['adversary', 'eps']).esc.mean().min():.1f}")
        M.safe("rSTwoStolenOne", lambda: f"{100 * a2[(a2.adversary == 'A1') & (a2.eps == e_lo)].stolen_total.mean():.2f}")
        M.safe("rSTwoNtlOneLo", lambda: f"{100 * a2[(a2.adversary == 'A1') & (a2.eps == e_lo)].ntl_undetected.mean():.2f}")
        M.safe("rSTwoNtlOneHi", lambda: f"{100 * a2[(a2.adversary == 'A1') & (a2.eps == e_hi)].ntl_undetected.mean():.2f}")
        M.safe("rSTwoStatRecOne", lambda: f"{100 * (lambda d: (d.ntl_recovered / d.stolen_total).mean())(s2s[(s2s.adversary == 'A1') & np.isclose(s2s.r_static, r_stat[0])]):.1f}")
        M.safe("rSTwoStolenOneHi", lambda: f"{100 * a2[(a2.adversary == 'A1') & (a2.eps == e_hi)].stolen_total.mean():.2f}")
        M.safe("rSTwoStolenCut", lambda: f"{100 * (1 - a2[(a2.adversary == 'A1') & (a2.eps == e_lo)].stolen_total.mean() / a2[(a2.adversary == 'A1') & (a2.eps == e_hi)].stolen_total.mean()):.1f}")
        M.safe("rSTwoRecallMax", lambda: f"{100 * a2[a2.eps == e_lo].groupby('adversary').recall.mean().max():.2f}")
        M.safe("rSTwoMeanR", lambda: f"{a2[a2.eps == e_lo].mean_r.mean():.2f}")
        pv2 = a2.groupby(["adversary", "eps"]).ntl_undetected.mean().unstack("eps")
        M.safe("rSTwoEpsGainMax", lambda: f"{100 * (pv2[e_lo] - pv2[e_hi]).max():.2f}")

    # ---------------- F1: NTL lolos vs eps per adversary
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.0, 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, kinds):
        for v in game_vars:
            d = _agg(ad[(ad.variant == v) & (ad.adversary == k)], "eps", "ntl_undetected")
            ax.errorbar(d["eps"], d["mean"] * 100, yerr=d["std"] * 100, marker="o", ms=3, lw=1.2,
                        color=COL.get(v), label=LAB_PLAIN.get(v, v), capsize=2)
        ax.set_title(f"Adversary {k}")
        ax.set_xlabel(r"random share $\varepsilon$")
    axes[0].set_ylabel("undetected NTL (% of load)")
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    _save_fig(fig, "fig_eps.pdf", dirs)

    # ---------------- F2: kokoh vs konsisten (per r statis)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    sv = st[st.variant == main_v]
    wv = ad[ad.variant == main_v].groupby(["eps", "seed", "adversary"]).ntl_undetected.mean().reset_index() \
        .groupby(["eps", "seed"]).ntl_undetected.max().groupby("eps").mean()
    for rs, mk in zip(r_stat, ["o", "s", "^"]):
        stat = sv[np.isclose(sv.r_static, rs)].groupby("eps").ntl_undetected.mean()
        ax.plot(stat.values * 100, wv.loc[stat.index].values * 100, "-" + mk, ms=4, color="#444" if rs == r_stat[0] else "#999",
                label=f"static thieves at $r={rs:g}$")
        for e in stat.index:
            ax.annotate(f"{e:g}", (stat[e] * 100, wv[e] * 100), textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax.set_xlabel("undetected NTL, static thieves (%)")
    ax.set_ylabel("worst-case undetected NTL,\nadaptive thieves (%)")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    _save_fig(fig, "fig_tradeoff.pdf", dirs)

    # ---------------- F3: batas mimikri (TV-KDE; akurasi seimbang cross-fit; lantai nol-permutasi)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    d3 = e4[(e4.variant == main_v)]
    tv = d3.groupby("r")["tv_kde"].mean()
    ax.plot(tv.index, 0.5 * (1 + tv.values), "k--", label=r"bound $(1+\mathrm{TV})/2$")
    for k in kinds:
        sk = d3[d3.adversary == k].groupby("r")["ba_cf"].mean()
        ax.plot(sk.index, sk.values, "-o", ms=3, color=COL[k], label=f"{LAB_PLAIN[main_v]}, {k}")
    if "gbm_shape" in set(e4.variant):
        sk = e4[(e4.variant == "gbm_shape") & (e4.adversary == "A1")].groupby("r")["ba_cf"].mean()
        ax.plot(sk.index, sk.values, ":", color="#777", label="GBM (shape only), A1")
    null_ba = float((0.5 * (1 + e4.ks_null)).mean()) if float(e4.ks_null.max()) > 0 else 0.5
    ax.axhline(null_ba, color="#bbb", lw=0.8)
    ax.set_xlabel("under-reporting depth $r$")
    ax.set_ylabel("balanced accuracy")
    ax.legend(fontsize=6.0)
    fig.tight_layout()
    _save_fig(fig, "fig_mimicry.pdf", dirs)
    # makro E4: uji per-seed (seed x varian x jenis x r). A2 mengubah shape sehingga asumsi Prop. 1 tidak berlaku.
    e4["bd"] = 0.5 * (1 + e4.tv_kde)
    e4["bd_hist"] = 0.5 * (1 + e4.tv_bound)
    e4["v_cf"] = (e4.ba_cf - 1.645 * e4.ba_cf_se) > e4.bd + 1e-12      # uji satu-sisi 95% pada akurasi seimbang cross-fit
    e4["v_naive"] = e4.balacc_emp > e4.bd + 1e-12                      # estimator lama (maks. atas ambang) vs batas KDE
    e4["v_hist"] = e4.balacc_emp > e4.bd_hist + 1e-12                  # definisi lama (histogram), untuk perbandingan
    e4c = e4[e4.adversary.isin(["A1", "A3"])]
    M.add("rMimicViol", str(int(e4c.v_cf.sum())))
    M.add("rMimicN", str(int(len(e4c))))
    M.add("rMimicViolNaive", str(int(e4c.v_naive.sum())))
    M.add("rMimicViolHist", str(int(e4c.v_hist.sum())))
    e4a2 = e4[e4.adversary == "A2"]
    M.add("rMimicViolTwo", str(int(e4a2.v_cf.sum())))
    M.add("rMimicNTwo", str(int(len(e4a2))))
    M.safe("rMimicTvGap", lambda: f"{(e4c.tv_bound - e4c.tv_kde).mean():.3f}")
    M.safe("rMimicTvGapSmall", lambda: f"{(e4c[e4c.r == e4c.r.min()].tv_bound - e4c[e4c.r == e4c.r.min()].tv_kde).mean():.3f}")
    M.safe("rMimicOptimism", lambda: f"{(e4c.balacc_emp - e4c.ba_cf).mean():.3f}")
    M.safe("rMimicNullBA", lambda: f"{(0.5 * (1 + e4c.ks_null)).mean():.3f}")
    M.safe("rMimicSmallR", lambda: f"{e4c.r.min():g}")
    M.safe("rBoundSmallR", lambda: f"{e4c[e4c.r == e4c.r.min()].bd.mean():.3f}")
    M.safe("rBoundSmallRLo", lambda: f"{(0.5 * (1 + e4c[e4c.r == e4c.r.min()].tv_kde_hi)).mean():.3f}")
    M.safe("rBoundSmallRHi", lambda: f"{(0.5 * (1 + e4c[e4c.r == e4c.r.min()].tv_kde_lo)).mean():.3f}")
    M.safe("rBalAccShapeMax", lambda: f"{e4[e4.variant == 'gbm_shape'].groupby(['adversary', 'r']).ba_cf.mean().max():.3f}")
    M.safe("rBalAccLevelMax", lambda: f"{d3.groupby(['adversary', 'r']).ba_cf.mean().max():.3f}")
    M.safe("rBoundMax", lambda: f"{0.5 * (1 + tv.max()):.3f}")
    M.safe("rBoundAtMid", lambda: f"{0.5 * (1 + tv.loc[0.5]):.3f}")
    M.safe("rBalAccAtMidOne", lambda: f"{d3[(d3.adversary == 'A1') & np.isclose(d3.r, 0.5)].ba_cf.mean():.3f}")
    M.safe("rRgridMax", lambda: f"{max(ex['r_grid']):g}")

    # ---------------- F4: deterensi (E6)
    if len(e6):
        d = e6.groupby(["budget", "penalty", "eps"]).agg(fi=("floor_index", "mean"), det=("deterred", "mean"),
                                                         ntl=("ntl_undetected", "mean"), conv=("converged", "mean")).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
        mk = {0.0: "o", 0.25: "s", 0.5: "^", 1.0: "D"}
        for (phi, eps), sub in d.groupby(["penalty", "eps"]):
            axes[0].plot(sub.budget * 100, sub.det, "-" + mk.get(eps, "o"), ms=3, lw=1,
                         color="#1f77b4" if phi == min(d.penalty) else "#d62728",
                         label=f"$\\varphi={phi:g}$, $\\varepsilon={eps:g}$")
        axes[0].set_xlabel("inspection budget $b$ (%)")
        axes[0].set_ylabel("fraction of thieves deterred")
        axes[0].legend(fontsize=5.5, ncol=2)
        for (phi, eps), sub in d.groupby(["penalty", "eps"]):
            if eps == 0:
                continue
            axes[1].plot(sub.fi, sub.det, mk.get(eps, "o"), ms=3.5,
                         color="#1f77b4" if phi == min(d.penalty) else "#d62728")
        axes[1].axvline(1.0, color="k", ls="--", lw=0.8)
        axes[1].text(1.02, 0.5, r"$\varepsilon b(1+\varphi)=1$", fontsize=6.5, rotation=90, va="center")
        axes[1].set_xlabel(r"floor index $\varepsilon b(1+\varphi)$")
        axes[1].set_ylabel("fraction of thieves deterred")
        fig.tight_layout()
        _save_fig(fig, "fig_deter.pdf", dirs)
        dd6 = e6.groupby(["budget", "penalty", "eps"]).agg(fi=("floor_index", "mean"), det=("deterred", "mean")).reset_index()
        above, below = dd6[dd6.fi >= 1 - 1e-9], dd6[(dd6.fi < 1 - 1e-9) & (dd6.eps > 0)]
        M.add("rDeterAboveMin", f"{100 * above.det.min():.1f}" if len(above) else r"\textbf{??}")
        M.add("rDeterAboveN", str(len(above)))
        M.add("rDeterBelowMax", f"{100 * below.det.max():.1f}" if len(below) else r"\textbf{??}")
        top = dd6[dd6.eps == 0]
        M.safe("rDeterTopkMax", lambda: f"{100 * top.det.max():.1f}")
        M.safe("rDeterTopkMaxB", lambda: _pct(top.loc[top.det.idxmax(), 'budget']))
        M.safe("rDeterTopkMaxPhi", lambda: f"{top.loc[top.det.idxmax(), 'penalty']:g}")
        dc = cfg["deterrence"]
        M.add("rDeterBudgets", ", ".join(_pct(b) for b in dc["budgets"]))
        M.add("rDeterBudgetMax", _pct(max(dc["budgets"])))
        M.add("rDeterPhis", ", ".join(f"{p:g}" for p in dc["penalties"]))
        M.add("rDeterEps", ", ".join(f"{e:g}" for e in dc["eps_grid"]))
        M.add("rDeterK", str(dc.get("br_iters", ex["br_iters"])))
        M.add("rDeterConv", f"{100 * e6.converged.mean():.1f}")
        M.add("rDeterAdv", ", ".join(dc["adversaries"]))
        # NTL lolos pada b terbesar
        bmax = max(dc["budgets"])
        M.safe("rDeterNtlTopkBmax", lambda: f"{100 * d[(d.budget == bmax) & (d.eps == 0)].ntl.max():.2f}")
        M.add("rDeterBmax", _pct(bmax))

    # ---------------- ekstensi TSG (E7, E9, onset, UCI, batas kedalaman, CNN)
    from .report_ext import stage_report_ext
    stage_report_ext(cfg, dirs, M, _save_tab, _save_fig, LAB)

    # ---------------- numbers.tex + ringkasan
    _save_tab(M.tex().rstrip("\n"), "numbers.tex", dirs)
    summ = [f"# Summary ({cfg['name']})", "", f"- seeds: {cfg['seeds']}", f"- data source: {cfg['data']['source']}",
            "- tables: " + ", ".join(sorted(p.name for p in dirs["paper"].glob("tab_*.tex"))),
            "- figures: " + ", ".join(sorted(p.name for p in dirs["paper"].glob("fig_*.pdf"))), "",
            "## Numbers (macro -> value)", ""]
    summ += [f"- `\\{k}` = {v}" for k, v in sorted(M.items.items())]
    (R / "SUMMARY.md").write_text("\n".join(summ) + "\n")
    shutil.copy(R / "SUMMARY.md", dirs["paper"] / "SUMMARY.md")
    print(f"report -> {dirs['figures']} and {dirs['paper']}")


def _read_cfg_br(sens_dir, default):
    """Baca br_iters run sensitivitas dari berkas penanda yang ditulis pipeline (fallback: default)."""
    if sens_dir is None:
        return default
    f = Path(sens_dir) / "config_used.yaml"
    if f.exists():
        import yaml
        return yaml.safe_load(f.read_text())["experiment"]["br_iters"]
    return default
