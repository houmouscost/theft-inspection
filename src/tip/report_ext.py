"""Laporan untuk ekstensi TSG (E7 kebijakan sadar-strategi, E9 rasionalitas terbatas, skenario onset S3,
replikasi UCI, sensitivitas batas kedalaman dan pembanding CNN). Dipanggil dari report.stage_report.

Membaca run lain dari path di cfg["report"]; bagian yang hasilnya belum ada dilewati.
Semua angka di teks lewat makro (objek Macros dari report.py).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POL_ORDER = ["topk", "random", "milp_score", "milp_level", "interval_level", "milp_joint", "oracle_joint"]
POL_LAB = {"topk": "Top-$k$ ($\\varepsilon{=}0$)", "random": "Random ($\\varepsilon{=}1$)",
           "milp_score": "Strategic: score cells", "milp_level": "Strategic: level cells (no ML)",
           "interval_level": "Strategic: interval rule (3 par.)",
           "milp_joint": "Strategic: score$\\times$level", "oracle_joint": "Oracle: score$\\times$level"}
KK = {"A1": "One", "A2": "Two", "A3": "Three"}


def _read(path: Path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 5 else pd.DataFrame()


def _pm(m, s, scale=100.0, d=2):
    return f"{m * scale:.{d}f}$\\pm${s * scale:.{d}f}"


def _pct(x):
    return f"{100 * x:g}"


def stage_report_ext(cfg: dict, dirs: dict, M, save_tab, save_fig, LAB):
    rep = cfg.get("report") or {}
    R = dirs["results"]
    ex = cfg["experiment"]
    b_mid, phi_mid = sorted(set(ex["budgets"]))[len(set(ex["budgets"])) // 2], sorted(set(ex["penalties"]))[len(set(ex["penalties"])) // 2]

    # =================================================================== E7 kebijakan sadar-strategi
    scen = [("S1", R)]
    if rep.get("level_matched"):
        scen.append(("S2", Path(rep["level_matched"])))
    if rep.get("onset"):
        scen.append(("S3", Path(rep["onset"])))
    if rep.get("uci"):
        scen.append(("UCI", Path(rep["uci"])))
    e7 = {k: _read(p / "e7_strategic.csv") for k, p in scen}
    e7 = {k: v for k, v in e7.items() if len(v)}
    for sc_, dpath in (rep.get("interval_runs") or {}).items():   # aturan interval (configs/diag_interval*.yaml)
        di = _read(Path(dpath) / "e7_strategic.csv")
        if sc_ in e7 and len(di):
            e7[sc_] = pd.concat([e7[sc_], di[di.policy == "interval_level"]], ignore_index=True)
    main_var = {"S1": "gbm_level", "S2": "gbm_level", "S3": rep.get("onset_variant", "gbm_change_adv"),
                "UCI": rep.get("uci_variant", "gbm_change_syn")}
    if "S1" in e7:
        s7 = cfg.get("strategic") or {}
        bb, pp = sorted(s7.get("budgets", ex["budgets"])), sorted(s7.get("penalties", ex["penalties"]))
        pp = sorted(rep.get("strategic_table_penalties") or pp)   # tabel ringkas; semua phi tetap di makro/CSV
        cols = [(sc, b, p) for sc in e7 for b in bb for p in pp
                if len(e7[sc][(e7[sc].budget == b) & (e7[sc].penalty == p)])]
        # tabel utama E7: baris kebijakan, kolom (skenario, b, phi)
        groups = {}
        for sc, b, p in cols:
            groups.setdefault(sc, []).append((b, p))
        head1 = " & " + " & ".join(rf"\multicolumn{{{len(v)}}}{{c}}{{{sc}}}" for sc, v in groups.items()) + r" \\"
        cm, i0 = "", 2
        for sc, v in groups.items():
            cm += rf"\cmidrule(lr){{{i0}-{i0 + len(v) - 1}}}"
            i0 += len(v)
        head2 = "Policy & " + " & ".join(f"{_pct(b)}\\%, $\\varphi{{=}}{p:g}$" for sc, v in groups.items() for b, p in v) + r" \\"
        lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule", head1, cm, head2, r"\midrule"]
        for pol in POL_ORDER:
            cells, any_ = [], False
            for sc, b, p in cols:
                d = e7[sc]
                d = d[(d.variant == main_var[sc]) & (d.budget == b) & (d.penalty == p) & (d.policy == pol)]
                if len(d):
                    any_ = True
                    cells.append(_pm(d.ntl_undetected.mean(), d.ntl_undetected.std(ddof=0)))
                else:
                    cells.append("--")
            if any_:
                lines.append(POL_LAB[pol] + " & " + " & ".join(cells) + r" \\")
                if pol == "random":
                    lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}"]
        save_tab("\n".join(lines), "tab_strategic.tex", dirs)

        # makro E7
        for sc, d in e7.items():
            d = d[d.variant == main_var[sc]]
            g = d.groupby(["budget", "penalty", "policy"]).agg(und=("ntl_undetected", "mean"),
                                                               st=("stolen_total", "mean"), rec=("recall", "mean")).reset_index()
            nm = {"S1": "SOne", "S2": "STwo", "S3": "SThree", "UCI": "Uci"}[sc]
            gi = g.set_index(["budget", "penalty", "policy"])
            for b, bn in ((0.02, "Two"), (0.05, "Five")):
                for p, pn in ((1.0, "One"), (3.0, "Three")):
                    for pol, pl in (("topk", "Topk"), ("random", "Rand"), ("milp_joint", "Joint"),
                                    ("milp_level", "Level"), ("milp_score", "Score"), ("oracle_joint", "Oracle"),
                                    ("interval_level", "Interval")):
                        if (b, p, pol) in gi.index:
                            M.add(f"rSe{nm}B{bn}P{pn}{pl}", f"{100 * gi.loc[(b, p, pol), 'und']:.2f}")
                            M.add(f"rSeSt{nm}B{bn}P{pn}{pl}", f"{100 * gi.loc[(b, p, pol), 'st']:.2f}")
            dd = d[d.policy.str.startswith("milp_") | (d.policy == "interval_level")]
            M.add(f"rSeDeterMax{nm}", f"{100 * dd.groupby(['budget', 'penalty', 'policy']).deterred.mean().max():.1f}")
            # perbaikan terbaik kebijakan strategis (out-of-sample) relatif thd min(top-k, acak)
            w = g.pivot_table(index=["budget", "penalty"], columns="policy", values="und")
            if {"topk", "random", "milp_joint"} <= set(w.columns):
                base = w[["topk", "random"]].min(1)
                # hasil utama: kebijakan joint (ditetapkan sebelumnya); "Best" = varian terbaik per setting (post hoc)
                j = w["milp_joint"]
                M.add(f"rSeGainMax{nm}", f"{100 * (base - j).max():.2f}")
                M.add(f"rSeGainRelMax{nm}", f"{100 * ((base - j) / base).max():.1f}")
                M.add(f"rSeGainMin{nm}", f"{100 * (base - j).min():.2f}")
                best = w[[c for c in ("milp_joint", "milp_level", "milp_score") if c in w]].min(1)
                M.add(f"rSeBestGainMax{nm}", f"{100 * (base - best).max():.2f}")
                M.add(f"rSeBestRelMax{nm}", f"{100 * ((base - best) / base).max():.1f}")
        # aturan interval vs kebijakan sel-level penuh
        for sc, d in e7.items():
            if "interval_level" not in set(d.policy):
                continue
            d = d[d.variant == main_var[sc]]
            w = d.pivot_table(index=["budget", "penalty"], columns="policy", values="ntl_undetected")
            base = w[["topk", "random"]].min(1)
            gl, gi_ = base - w["milp_level"], base - w["interval_level"]
            nm = {"S1": "SOne", "S2": "STwo"}.get(sc)
            if nm is None:
                continue
            ok = gl > 0.001                                   # hanya setting di mana kebijakan level memang menolong
            if ok.any():
                sh = (gi_ / gl)[ok]
                M.add(f"rIvShareMin{nm}", f"{100 * sh.min():.0f}")
                M.add(f"rIvShareMax{nm}", f"{100 * sh.max():.0f}")
                M.add(f"rIvShareN{nm}", str(int(ok.sum())))
            M.add(f"rIvGainMax{nm}", f"{100 * gi_.max():.2f}")
            M.add(f"rIvWorseThanBase{nm}", str(int((gi_ < -0.0005).sum())))
            M.add(f"rLvWorseThanBase{nm}", str(int((gl < -0.0005).sum())))
            iv = d[d.policy == "interval_level"].groupby(["budget", "penalty"])[["iv_lo", "iv_hi", "iv_p"]].mean()
            if (0.02, 3.0) in iv.index:
                r_ = iv.loc[(0.02, 3.0)]
                n_l = 12
                M.add(f"rIvPctLo{nm}", f"{100 * r_.iv_lo / n_l:.0f}")
                M.add(f"rIvPctHi{nm}", f"{100 * (r_.iv_hi + 1) / n_l:.0f}")
                M.add(f"rIvP{nm}", f"{r_.iv_p:.2f}")
            M.add(f"rIvPMin{nm}", f"{iv.iv_p.min():.2f}")
            M.add(f"rIvPMax{nm}", f"{iv.iv_p.max():.2f}")
        # solver check (E7v)
        v = _read(R / "e7_solver_check.csv")
        if len(v):
            M.add("rSolverN", str(len(v)))
            M.add("rSolverNsub", str(int(v.n_sub.iloc[0])))
            M.add("rSolverTime", f"{v.milp_s.max():.0f}")
            M.add("rSolverLocalBeatsMilp", str(int((v.obj_local <= v.obj_milp + 1e-9).sum())))
            M.add("rSolverLocalVsUniform", f"{100 * (1 - (v.obj_local / v.obj_uniform)).mean():.1f}")
            M.add("rSolverMilpVsUniform", f"{100 * (1 - (v.obj_milp / v.obj_uniform)).mean():.1f}")
            db = v.dropna(subset=["milp_dual_bound"])
            db = db[db.milp_dual_bound > 0]
            M.add("rSolverNdual", str(len(db)))
            if len(db):
                M.add("rSolverDualGapMax", f"{100 * ((db.obj_local - db.milp_dual_bound) / db.obj_local).max():.1f}")
                M.add("rSolverDualGapMean", f"{100 * ((db.obj_local - db.milp_dual_bound) / db.obj_local).mean():.1f}")
            M.add("rSolverLocalTime", f"{v.local_s.mean():.0f}")

        # gambar kebijakan: rerata p per sel skor x level (S1 & S2), b mid, phi tertinggi
        cells = {}
        for sc, p in scen[:2]:
            c = _read(p / "e7_policy_cells.csv")
            if len(c):
                c = c[(c["mode"] == "joint") & (c.variant == main_var[sc])]
                if not len(c):
                    continue
                phi_hi = c.penalty.max()
                b_sel = 0.02 if (c.budget == 0.02).any() else c.budget.min()
                cells[sc] = (c[(c.budget == b_sel) & (c.penalty == phi_hi)], b_sel, phi_hi)
        if cells:
            fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 2.7), squeeze=False)
            for ia, (ax, (sc, (c, b_sel, phi_hi))) in enumerate(zip(axes[0], cells.items())):
                h = c.groupby(["s_bin", "l_bin"]).p.mean().unstack("l_bin")
                qlo = c.groupby("s_bin").s_q_lo.first() if "s_q_lo" in c else None
                im = ax.imshow(h.values, origin="lower", cmap="viridis", vmin=0, vmax=max(0.3, float(h.values.max())),
                               aspect="auto")
                for (i, j), val in np.ndenumerate(h.values):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5,
                            color="w" if val < 0.6 * max(0.3, float(h.values.max())) else "k")
                ax.set_xlabel("reported-level sextile (low $\\rightarrow$ high)")
                if ia == 0:
                    ax.set_ylabel("score cell (honest quantile from)")
                ax.set_title(f"{sc}: $b={_pct(b_sel)}\\%$, $\\varphi={phi_hi:g}$", fontsize=8)
                ax.set_xticks(range(h.shape[1])); ax.set_yticks(range(h.shape[0]))
                if qlo is not None:
                    ax.set_yticklabels([f"{q:.2f}" for q in qlo.reindex(h.index).values], fontsize=6)
            fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="inspection probability")
            save_fig(fig, "fig_policy.pdf", dirs)
            c1 = cells.get("S1")
            if c1 is not None:
                h = c1[0].groupby(["s_bin", "l_bin"]).p.mean().unstack("l_bin")
                M.add("rPolicyMaxP", f"{h.values.max():.2f}")
                M.add("rPolicyDetLevel", f"{1 / (1 + c1[2]):.2f}")

    # =================================================================== E9 rasionalitas terbatas
    e9 = _read(R / "e9_bounded.csv")
    if len(e9):
        g = e9.groupby(["adversary", "budget", "eps", "thief", "lam"], dropna=False).ntl_undetected.mean().reset_index()
        bd = cfg.get("bounded") or {}
        fig, axes = plt.subplots(1, len(bd.get("budgets", [0.02])), figsize=(7.0, 2.4), squeeze=False, sharey=True)
        for ax, b in zip(axes[0], bd.get("budgets", [0.02])):
            for k, col in (("A1", "#1f77b4"), ("A2", "#ff7f0e"), ("A3", "#d62728")):
                for eps, ls in ((0.0, "-"), (1.0, "--")):
                    q = g[(g.adversary == k) & (g.budget == b) & (g.eps == eps) & (g.thief == "quantal")].sort_values("lam")
                    if len(q):
                        ax.plot(q.lam, q.ntl_undetected * 100, ls, marker="o", ms=2.5, color=col,
                                label=f"{k}, " + ("top-$k$" if eps == 0 else "random"))
                    br = g[(g.adversary == k) & (g.budget == b) & (g.eps == eps) & (g.thief == "best_response")]
                    tr = g[(g.adversary == k) & (g.budget == b) & (g.eps == eps) & (g.thief == "transfer")]
                    if len(br) and eps == 0:
                        ax.axhline(br.ntl_undetected.iloc[0] * 100, color=col, lw=0.6, ls=":")
                    if len(tr) and eps == 0:
                        ax.plot([bd["lambdas"][-1] * 1.8], [tr.ntl_undetected.iloc[0] * 100], "x", color=col, ms=5)
            ax.set_xscale("log")
            ax.set_xlabel(r"rationality $\lambda$ (x: transfer attack)")
            ax.set_title(f"$b={_pct(b)}\\%$", fontsize=8)
        axes[0][0].set_ylabel("undetected NTL (% of load)")
        axes[0][0].legend(fontsize=5.5, ncol=2)
        fig.tight_layout()
        save_fig(fig, "fig_bounded.pdf", dirs)
        gi = g.set_index(["adversary", "budget", "eps", "thief", "lam"])
        for k in ("A1", "A2", "A3"):
            for b, bn in ((0.02, "Two"), (0.05, "Five")):
                try:
                    br0 = gi.loc[(k, b, 0.0, "best_response", np.inf), "ntl_undetected"]
                    rnd = gi.loc[(k, b, 1.0, "best_response", np.inf), "ntl_undetected"]
                    tr0 = gi.loc[(k, b, 0.0, "transfer", np.nan)] if False else \
                        g[(g.adversary == k) & (g.budget == b) & (g.eps == 0) & (g.thief == "transfer")].ntl_undetected.iloc[0]
                    q0 = g[(g.adversary == k) & (g.budget == b) & (g.eps == 0) & (g.thief == "quantal")]
                    q1 = g[(g.adversary == k) & (g.budget == b) & (g.eps == 1) & (g.thief == "quantal")]
                    M.add(f"rBdBr{KK[k]}B{bn}", f"{100 * br0:.2f}")
                    M.add(f"rBdRand{KK[k]}B{bn}", f"{100 * rnd:.2f}")
                    M.add(f"rBdTr{KK[k]}B{bn}", f"{100 * tr0:.2f}")
                    M.add(f"rBdQLow{KK[k]}B{bn}", f"{100 * q0[q0.lam == q0.lam.min()].ntl_undetected.iloc[0]:.2f}")
                    M.add(f"rBdQLowRand{KK[k]}B{bn}", f"{100 * q1[q1.lam == q1.lam.min()].ntl_undetected.iloc[0]:.2f}")
                    # lambda terkecil di mana top-k tidak lebih baik daripada acak
                    mq = q0.merge(q1, on="lam", suffixes=("_t", "_r"))
                    worse = mq[mq.ntl_undetected_t >= mq.ntl_undetected_r - 1e-12]
                    M.add(f"rBdCross{KK[k]}B{bn}", f"{worse.lam.min():g}" if len(worse) else "none")
                except (KeyError, IndexError):
                    pass
        M.add("rBdLambdas", ", ".join(f"{x:g}" for x in bd.get("lambdas", [])))
        q = g[(g.thief == "quantal")].pivot_table(index=["adversary", "budget", "lam"], columns="eps", values="ntl_undetected")
        if {0.0, 1.0} <= set(q.columns):
            d = (q[0.0] - q[1.0]) * 100
            for k in ("A1", "A2", "A3"):
                if k in d.index.get_level_values(0):
                    M.add(f"rBdDiffMax{KK[k]}", f"{d.xs(k).abs().max():.2f}")
                    M.add(f"rBdDiffMin{KK[k]}", f"{d.xs(k).min():.2f}")

    # =================================================================== robustness: batas kedalaman + CNN
    caps = rep.get("depth_caps") or {}
    cnn_dir = Path(rep["cnn"]) if rep.get("cnn") else None
    rows_r = []

    def _adapt(e2, var):
        a = e2[(e2["mode"] == "adaptive") & (e2.variant == var) & (e2.budget == b_mid) & (e2.penalty == phi_mid) & (e2.tau == 0)]
        out = {}
        for k in ("A1", "A2"):
            x0, x1 = a[(a.adversary == k) & (a.eps == 0)], a[(a.adversary == k) & (a.eps == 1)]
            out[k] = (x0.mean_r.mean(), x0.stolen_total.mean(), x0.ntl_undetected.mean(), x1.ntl_undetected.mean(),
                      (x0.ntl_undetected / x0.stolen_total).mean())
        return out

    for cap, p in sorted(((float(c), Path(p)) for c, p in caps.items())):
        e2 = _read(p / "e2_game.csv")
        if len(e2):
            rows_r.append((f"GBM, $r\\le{cap:g}$", None, None, _adapt(e2, "gbm_level")))
    e2m = _read(R / "e2_game.csv")
    e1m = _read(R / "e1_detector.csv")
    if len(e2m):
        rows_r.append((f"GBM, $r\\le{max(ex['r_grid']):g}$ (main)", e1m[e1m.variant == "gbm_level"].auc.mean() if "auc" in e1m else None,
                       e1m[e1m.variant == "gbm_level"].spearman_level.mean() if "spearman_level" in e1m else None,
                       _adapt(e2m, "gbm_level")))
    if cnn_dir is not None:
        e1c, e2c = _read(cnn_dir / "e1_detector.csv"), _read(cnn_dir / "e2_game.csv")
        if len(e2c):
            ec = e1c[e1c.variant == "cnn_wd"]
            rows_r.append((f"Wide\\&Deep CNN, $r\\le{max(ex['r_grid']):g}$", ec.auc.mean(), ec.spearman_level.mean(),
                           _adapt(e2c, "cnn_wd")))
            M.add("rCnnAuc", f"{ec.auc.mean():.3f}"); M.add("rCnnAucSd", f"{ec.auc.std():.3f}")
            M.add("rCnnRho", f"{ec.spearman_level.mean():.2f}")
            M.add("rCnnPrAuc", f"{ec.pr_auc.mean():.3f}")
            st = e2c[(e2c["mode"] == "static") & (e2c.variant == "cnn_wd") & (e2c.budget == b_mid) & (e2c.eps == 0)]
            M.add("rCnnStatRecallMax", f"{100 * st.groupby(['adversary', 'r_static']).recall.mean().max():.2f}")
            a = e2c[(e2c["mode"] == "adaptive") & (e2c.variant == "cnn_wd") & (e2c.budget == b_mid) & (e2c.penalty == phi_mid)]
            M.add("rCnnEscapeMin", f"{100 * (a.ntl_undetected / a.stolen_total).groupby([a.adversary, a.eps]).mean().min():.1f}")
            M.add("rCnnMeanR", f"{a[a.eps == 0].mean_r.mean():.2f}")
            e1g = e1c[e1c.variant == "gbm_level"]
            M.add("rCnnGbmRho", f"{e1g.spearman_level.mean():.2f}")
    if "spearman_level" in e1m:
        for v, nm in (("gbm_level", "Level"), ("gbm_shape", "Shape"), ("gbm_level_adv", "Adv"), ("gbm_level_miss", "Miss")):
            if v in set(e1m.variant):
                M.add(f"rRho{nm}", f"{e1m[e1m.variant == v].spearman_level.mean():.2f}")
    if rows_r:
        lines = [r"\begin{tabular}{lcc" + "cccc" * 2 + "}", r"\toprule",
                 r" & & & \multicolumn{4}{c}{A1} & \multicolumn{4}{c}{A2} \\",
                 r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
                 r"Setting & AUC & $\rho_{\mathrm{level}}$ & $\bar r^\star$ & Stolen & $\NTL^{\mathrm{und}}_{\varepsilon=0}$ & $\NTL^{\mathrm{und}}_{\varepsilon=1}$"
                 r" & $\bar r^\star$ & Stolen & $\NTL^{\mathrm{und}}_{\varepsilon=0}$ & $\NTL^{\mathrm{und}}_{\varepsilon=1}$ \\", r"\midrule"]
        for lab, auc, rho, out in rows_r:
            cells = [f"{auc:.3f}" if auc is not None else "--", f"{rho:.2f}" if rho is not None else "--"]
            for k in ("A1", "A2"):
                r_, s_, u0, u1, _ = out[k]
                cells += [f"{r_:.2f}", f"{100 * s_:.2f}", f"{100 * u0:.2f}", f"{100 * u1:.2f}"]
            lines.append(lab + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        save_tab("\n".join(lines), "tab_robust.tex", dirs)
        esc = [out[k][4] for _, _, _, out in rows_r for k in ("A1", "A2")]
        M.add("rRobustEscapeMin", f"{100 * min(esc):.1f}")
        for cap, p in caps.items():
            e2 = _read(Path(p) / "e2_game.csv")
            if len(e2):
                o = _adapt(e2, "gbm_level")
                nm = {"0.5": "Five", "0.7": "Seven"}.get(str(cap), "X")
                M.add(f"rCap{nm}StolenOne", f"{100 * o['A1'][1]:.2f}")
                M.add(f"rCap{nm}MeanROne", f"{o['A1'][0]:.2f}")

    # =================================================================== skenario onset S3
    if rep.get("onset"):
        po = Path(rep["onset"])
        e1o, e2o = _read(po / "e1_detector.csv"), _read(po / "e2_game.csv")
        if len(e2o):
            kinds = [k for k in ["A4-0", "A4-30", "A4-90", "A4-180", "A4-365"] if k in set(e2o.adversary)]
            vars_ = [v for v in ["gbm_level", "gbm_change", "gbm_change_adv"] if v in set(e2o.variant)]
            OL = {"gbm_level": "GBM (shape+level)", "gbm_change": "GBM + change", "gbm_change_adv": "GBM + change + adv.\\ tr."}
            a = e2o[(e2o["mode"] == "adaptive") & (e2o.budget == b_mid) & (e2o.penalty == phi_mid) & (e2o.tau == 0)]
            s = e2o[(e2o["mode"] == "static") & (e2o.budget == b_mid) & (e2o.tau == 0) & (e2o.eps == 0)]
            r_hi = max(ex["r_static"])
            lines = [r"\begin{tabular}{lcc" + "cc" * len(kinds) + "}", r"\toprule",
                     r" & & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$L{{=}}{k.split('-')[1]}$\,d}}" for k in kinds) + r" \\",
                     "".join(rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(kinds))),
                     r"Detector & AUC & Stat.\ rec. & " + " & ".join([r"$\bar r^\star$", "Und./Stolen"] * len(kinds)) + r" \\",
                     r"\midrule"]
            for v in vars_:
                auc = e1o[e1o.variant == v].auc.mean() if len(e1o) else np.nan
                ss = s[(s.variant == v) & np.isclose(s.r_static, r_hi)]
                srec = (ss.ntl_recovered / ss.stolen_total).mean()
                cells = [f"{auc:.3f}", f"{100 * srec:.1f}"]
                for k in kinds:
                    x = a[(a.variant == v) & (a.adversary == k) & (a.eps == 0)]
                    cells += [f"{x.mean_r.mean():.2f}", f"{100 * x.ntl_undetected.mean():.2f}/{100 * x.stolen_total.mean():.2f}"]
                lines.append(OL[v] + " & " + " & ".join(cells) + r" \\")
            xr = a[(a.variant == vars_[0]) & (a.eps == 1)]
            cells = ["--", "--"]
            for k in kinds:
                x = xr[xr.adversary == k]
                cells += [f"{x.mean_r.mean():.2f}", f"{100 * x.ntl_undetected.mean():.2f}/{100 * x.stolen_total.mean():.2f}"]
            lines += [r"\midrule", "Random ($\\varepsilon{=}1$) & " + " & ".join(cells) + r" \\", r"\bottomrule", r"\end{tabular}"]
            save_tab("\n".join(lines), "tab_onset.tex", dirs)
            for v, nm in (("gbm_level", "Level"), ("gbm_change", "Change"), ("gbm_change_adv", "ChangeAdv")):
                if v in vars_:
                    M.add(f"rOnAuc{nm}", f"{e1o[e1o.variant == v].auc.mean():.3f}")
                    M.add(f"rOnRho{nm}", f"{e1o[e1o.variant == v].spearman_level.mean():.2f}")
                    ss = s[(s.variant == v) & np.isclose(s.r_static, r_hi)]
                    M.add(f"rOnStatRec{nm}", f"{100 * (ss.ntl_recovered / ss.stolen_total).mean():.1f}")
                    x = a[(a.variant == v) & (a.eps == 0)]
                    M.add(f"rOnMeanR{nm}", f"{x.mean_r.mean():.2f}")
                    M.add(f"rOnEscape{nm}", f"{100 * (x.ntl_undetected / x.stolen_total).mean():.1f}")
            M.add("rOnRStat", f"{r_hi:g}")
            M.add("rOnKinds", ", ".join(k.split("-")[1] for k in kinds))
            ao = a[(a.eps == 0)]
            ar_ = a[(a.eps == 1) & (a.variant == vars_[0])]
            for v, nm in (("gbm_change_adv", "ChangeAdv"), ("gbm_change", "Change")):
                if v in vars_:
                    t = ao[ao.variant == v].groupby("adversary").ntl_undetected.mean()
                    rr = ar_.groupby("adversary").ntl_undetected.mean()
                    M.add(f"rOnGainVsRand{nm}", f"{100 * (rr - t).max():.2f}")
                    M.add(f"rOnGainVsRandMin{nm}", f"{100 * (rr - t).min():.2f}")

    # =================================================================== UCI (E2 per adversary)
    if rep.get("uci"):
        pu = Path(rep["uci"])
        e0u, e2u = _read(pu / "e0_data.csv"), _read(pu / "e2_game.csv")
        if len(e2u):
            M.add("rUciN", str(int(e0u.n_consumers.iloc[0])))
            M.add("rUciDays", f"{int(e0u.n_days.iloc[0]):,}".replace(",", "{,}"))
            M.add("rUciSeeds", str(e0u.seed.nunique()))
            M.add("rUciThieves", f"{e0u.n_s1_thieves.mean():.0f}")
            M.add("rUciTestHonest", f"{e0u.n_test_honest.mean():.0f}")
            vu = rep.get("uci_variant", "gbm_change_syn")
            a = e2u[(e2u["mode"] == "adaptive") & (e2u.variant == vu) & (e2u.budget == b_mid) & (e2u.penalty == phi_mid)]
            s = e2u[(e2u["mode"] == "static") & (e2u.variant == vu) & (e2u.budget == b_mid) & (e2u.eps == 0)]
            kinds = [k for k in ["A1", "A2", "A3", "A4-0", "A4-90", "A4-365"] if k in set(a.adversary)]
            lines = [r"\begin{tabular}{lccccc}", r"\toprule",
                     r"Attack & Stat.\ rec. & $\bar r^\star$ & Stolen & $\NTL^{\mathrm{und}}_{\varepsilon=0}$ & $\NTL^{\mathrm{und}}_{\varepsilon=1}$ \\",
                     r"\midrule"]
            for k in kinds:
                ss = s[(s.adversary == k) & np.isclose(s.r_static, max(ex["r_static"]))]
                x0, x1 = a[(a.adversary == k) & (a.eps == 0)], a[(a.adversary == k) & (a.eps == 1)]
                lab = k if not k.startswith("A4") else f"A4, $L{{=}}{k.split('-')[1]}$\\,d"
                lines.append(f"{lab} & {100 * (ss.ntl_recovered / ss.stolen_total).mean():.1f} & {x0.mean_r.mean():.2f} & "
                             f"{100 * x0.stolen_total.mean():.2f} & {_pm(x0.ntl_undetected.mean(), x0.ntl_undetected.std(ddof=0))} & "
                             f"{_pm(x1.ntl_undetected.mean(), x1.ntl_undetected.std(ddof=0))} \\\\")
            lines += [r"\bottomrule", r"\end{tabular}"]
            save_tab("\n".join(lines), "tab_uci.tex", dirs)
            t0 = a[a.eps == 0].groupby("adversary").ntl_undetected.mean()
            t1 = a[a.eps == 1].groupby("adversary").ntl_undetected.mean()
            for k, nm in (("A1", "One"), ("A2", "Two"), ("A3", "Three"), ("A4-365", "FourL")):
                if k in t0:
                    M.add(f"rUciNtl{nm}Lo", f"{100 * t0[k]:.2f}")
                    M.add(f"rUciNtl{nm}Hi", f"{100 * t1[k]:.2f}")
                    ss = s[(s.adversary == k) & np.isclose(s.r_static, max(ex["r_static"]))]
                    M.add(f"rUciStatRec{nm}", f"{100 * (ss.ntl_recovered / ss.stolen_total).mean():.1f}")
                    M.add(f"rUciMeanR{nm}", f"{a[(a.adversary == k) & (a.eps == 0)].mean_r.mean():.2f}")

    # =================================================================== diagnostik data (seleksi label, kebersihan)
    ds, dh, dz = _read(R / "d_selection.csv"), _read(R / "d_hygiene.csv"), _read(R / "d_zero_load.csv")
    if len(ds):
        dec = ds[ds.dimension == "consumption_decile"].sort_values("bin")
        mb = ds[ds.dimension == "missing_bin"].sort_values("bin")
        mlab = ["$<$1", "1--5", "5--10", "10--20", "20--40", "40--60", r"$\ge$60"]
        lines = [r"\begin{tabular}{l" + "c" * 10 + "}", r"\toprule",
                 "Consumption decile & " + " & ".join(str(int(b)) for b in dec.bin) + r" \\",
                 r"Theft rate (\%) & " + " & ".join(f"{100 * r:.1f}" for r in dec.rate) + r" \\", r"\midrule",
                 r"Missing readings (\%) & " + " & ".join(mlab[: len(mb)]) + " & " * (10 - len(mb)) + r" \\",
                 r"Theft rate (\%) & " + " & ".join(f"{100 * r:.1f}" for r in mb.rate) + " & " * (10 - len(mb)) + r" \\",
                 r"\bottomrule", r"\end{tabular}"]
        save_tab("\n".join(lines), "tab_selection.tex", dirs)
        M.add("rDecLo", f"{100 * dec.rate.iloc[0]:.1f}")
        M.add("rDecHi", f"{100 * dec.rate.iloc[-1]:.1f}")
        M.add("rDecMid", f"{100 * dec.rate.iloc[4]:.1f}")
        M.add("rNSelPos", f"{int(dec.n.sum()):,}".replace(",", "{,}"))
        M.add("rMissFirst", f"{100 * mb.rate.iloc[0]:.1f}")
        M.add("rMissPeak", f"{100 * mb.rate.max():.1f}")
        M.add("rMissPeakBin", mlab[int(mb.rate.values.argmax())])
        M.add("rMissTop", f"{100 * mb.rate.iloc[-1]:.1f}")
        M.add("rMissRho", f"{float(ds[ds.dimension == 'spearman_missing_label'].rate.iloc[0]):.2f}")
    if len(dh):
        h = dict(zip(dh["item"], dh["value"]))
        cnt = lambda k: f"{int(h[k]):,}".replace(",", "{,}")
        for nm, key in (("RawRows", "n_raw_rows"), ("AllMissing", "n_allmissing_rows"), ("AllMissingThieves", "n_allmissing_thieves"),
                        ("AllZero", "n_allzero"), ("AllZeroThieves", "n_allzero_thieves"), ("DupRows", "n_rows_in_dup_groups"),
                        ("DupGroups", "n_dup_groups"), ("DupGroupsPos", "n_dup_groups_positive"),
                        ("DupConflict", "n_dup_groups_conflict")):
            M.add(f"rD{nm}", cnt(key))
    if len(dz):
        g = dz.groupby("variant")
        M.add("rAucNonzeroLevel", f"{g.auc_train_test_nonzero.mean()['gbm_level']:.3f}")
        M.add("rRhoNonzeroLevel", f"{g.spearman_level_nonzero.mean()['gbm_level']:.2f}")
        M.add("rAucNonzeroShape", f"{g.auc_train_test_nonzero.mean()['gbm_shape']:.3f}")
    df_ = _read(R / "d_deter_feasible.csv")
    if len(df_):
        M.add("rFeasMedianM", f"{100 * df_.median_m.iloc[0]:.0f}")
        for (b_, bn), (p_, pn) in ((x, y_) for x in ((0.02, "Two"), (0.05, "Five")) for y_ in ((1.0, "One"), (3.0, "Three"))):
            r_ = df_[np.isclose(df_.budget, b_) & np.isclose(df_.penalty, p_)]
            if len(r_):
                M.add(f"rFeas{bn}{pn}", f"{100 * r_.frac_feasible.iloc[0]:.1f}")
