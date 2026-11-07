"""Hasilkan gambar (PDF) dan tabel LaTeX dari CSV hasil. Tidak ada angka yang ditulis manual."""
from __future__ import annotations

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"gbm_level": "#1f77b4", "gbm_shape": "#d62728", "gbm_level_adv": "#2ca02c"}
LAB = {"gbm_level": "GBM (shape+level)", "gbm_shape": "GBM (shape saja)",
       "gbm_level_adv": "GBM + adv. training", "gbm_level_miss": "GBM + missingness"}


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


def _save_tab(tex: str, name: str, dirs: dict):
    for d in (dirs["results"], dirs["paper"]):
        (d / name).write_text(tex)


def stage_report(cfg: dict, dirs: dict):
    R = dirs["results"]
    e1 = pd.read_csv(R / "e1_detector.csv")
    e2 = pd.read_csv(R / "e2_game.csv")
    e4 = pd.read_csv(R / "e4_mimicry.csv")
    e5 = pd.read_csv(R / "e5_certificate.csv") if (R / "e5_certificate.csv").stat().st_size > 5 else pd.DataFrame()
    ex = cfg["experiment"]
    b_mid = _mid(ex["budgets"])
    phi_mid = _mid(ex["penalties"])
    tau0 = min(ex["tau_grid"])
    ad = e2[(e2["mode"] == "adaptive") & (e2["budget"] == b_mid) & (e2["penalty"] == phi_mid)
            & (e2["tau"] == tau0)]
    st = e2[(e2["mode"] == "static") & (e2["budget"] == b_mid) & (e2["tau"] == tau0)]

    # ---------------- T1: detektor (S0)
    t = e1.groupby(["variant", "budget"]).agg(auc=("auc", "mean"), pr=("pr_auc", "mean"),
                                              rec=("recall_at_b", "mean"), prec=("precision_at_b", "mean")).reset_index()
    t = t[t["budget"] == b_mid]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             rf"Detektor & AUC & PR-AUC & Recall@{b_mid:.0%} & Presisi@{b_mid:.0%} \\".replace("%", r"\%"), r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{_tex(LAB.get(r.variant, r.variant))} & {r.auc:.3f} & {r.pr:.3f} & {r.rec:.3f} & {r.prec:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _save_tab("\n".join(lines), "tab_detector.tex", dirs)

    # ---------------- T2: tabel utama (varian gbm_level; NTL lolos, % beban sistem)
    main_v = "gbm_level" if "gbm_level" in ad["variant"].unique() else ad["variant"].iloc[0]
    kinds = list(ex["adversaries"])
    lines = [r"\begin{tabular}{l" + "c" * (2 * len(kinds)) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{k}}}" for k in kinds) + r" \\",
             " & " + " & ".join(["Adaptif", "Statis"] * len(kinds)) + r" \\", r"\midrule"]
    for eps in ex["eps_grid"]:
        lab = "Acak murni ($\\varepsilon{=}1$)" if eps == 1 else ("Top-$k$ murni ($\\varepsilon{=}0$)" if eps == 0 else f"Campuran $\\varepsilon{{=}}{eps}$")
        cells = []
        for k in kinds:
            a = ad[(ad.variant == main_v) & (ad.adversary == k) & (ad.eps == eps)]["ntl_undetected"]
            s = st[(st.variant == main_v) & (st.adversary == k) & (st.eps == eps)]["ntl_undetected"]
            cells += [_pm(a.mean(), a.std(ddof=0), 100), _pm(s.mean(), s.std(ddof=0), 100)]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _save_tab("\n".join(lines), "tab_main.tex", dirs)

    # ---------------- T3: sertifikat
    if len(e5):
        g = e5.groupby(["cal", "dep"]).agg(cc=("cov_conformal", "mean"), ch=("cov_hoeffding", "mean"),
                                           ub=("ub_conformal", "mean"), md=("mean_miss_dep", "mean"),
                                           tgt=("target", "mean")).reset_index()
        lines = [r"\begin{tabular}{llcccc}", r"\toprule",
                 r"Kalibrasi & Deployment & Cakupan conformal & Cakupan Hoeffding & $\hat q$ & Miss riil \\", r"\midrule"]
        for _, r in g.iterrows():
            lines.append(f"{r.cal} & {r.dep} & {r.cc:.3f} & {r.ch:.3f} & {r.ub:.3f} & {r.md:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        _save_tab("\n".join(lines), "tab_cert.tex", dirs)

    # ---------------- F1: NTL lolos vs eps per adversary
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.2 * len(kinds), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, kinds):
        for v in ad["variant"].unique():
            d = _agg(ad[(ad.variant == v) & (ad.adversary == k)], "eps", "ntl_undetected")
            ax.errorbar(d["eps"], d["mean"] * 100, yerr=d["std"] * 100, marker="o", ms=3, lw=1.2,
                        color=COL.get(v), label=LAB.get(v, v), capsize=2)
        ax.set_title(f"Adversary {k}")
        ax.set_xlabel(r"porsi acak $\varepsilon$")
    axes[0].set_ylabel("NTL lolos (% beban sistem)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for d in (dirs["figures"], dirs["paper"]):
        fig.savefig(d / "fig_eps.pdf")
    plt.close(fig)

    # ---------------- F2: kokoh vs konsisten
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    v = main_v
    stat = st[st.variant == v].groupby("eps")["ntl_undetected"].mean()
    worst = ad[ad.variant == v].groupby(["eps", "seed", "adversary"])["ntl_undetected"].mean().reset_index() \
        .groupby(["eps", "seed"])["ntl_undetected"].max().groupby("eps").mean()
    ax.plot(stat.values * 100, worst.values * 100, "-o", ms=4, color="#444")
    for e in stat.index:
        ax.annotate(f"{e:g}", (stat[e] * 100, worst[e] * 100), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel("NTL lolos, pencuri statis (konsistensi)")
    ax.set_ylabel("NTL lolos terburuk, adaptif (kekokohan)")
    fig.tight_layout()
    for d in (dirs["figures"], dirs["paper"]):
        fig.savefig(d / "fig_tradeoff.pdf")
    plt.close(fig)

    # ---------------- F3: batas mimikri
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    d3 = e4[(e4.variant == main_v)]
    tv = d3.groupby("r")["tv_bound"].mean()
    ax.plot(tv.index, 0.5 * (1 + tv.values), "k--", label=r"batas $(1+\mathrm{TV})/2$")
    for k, c in zip(kinds, ["#1f77b4", "#ff7f0e", "#d62728"]):
        s = d3[d3.adversary == k].groupby("r")["balacc_emp"].mean()
        ax.plot(s.index, s.values, "-o", ms=3, color=c, label=f"akurasi seimbang {k}")
    ax.set_xlabel("potongan $r$")
    ax.set_ylabel("akurasi seimbang detektor")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for d in (dirs["figures"], dirs["paper"]):
        fig.savefig(d / "fig_mimicry.pdf")
    plt.close(fig)

    # ---------------- ringkasan teks untuk README/paper
    (R / "SUMMARY.md").write_text(
        f"# Ringkasan ({cfg['name']})\n\n- seeds: {cfg['seeds']}\n- sumber data: {cfg['data']['source']}\n"
        f"- tabel: tab_detector.tex, tab_main.tex, tab_cert.tex\n- gambar: fig_eps.pdf, fig_tradeoff.pdf, fig_mimicry.pdf\n")
    shutil.copy(R / "SUMMARY.md", dirs["paper"] / "SUMMARY.md")
    print(f"laporan -> {dirs['figures']} dan {dirs['paper']}")
