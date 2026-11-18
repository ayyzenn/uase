#!/usr/bin/env python3
"""Figures for the protocol-v2 results (writes code/figures/v2/ and paper/files/images/v2/)."""
from __future__ import annotations
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
FIG = ROOT / "figures" / "v2"
PAP = ROOT.parent / "paper" / "files" / "images" / "v2"
FIG.mkdir(parents=True, exist_ok=True); PAP.mkdir(parents=True, exist_ok=True)
RENAME = {"Residual Z": "Robust residual", "UASE-v1 rule (LOF-first)": "UASE-v1 rule"}
MAIN = ["Isolation Forest", "LOF", "One-Class SVM", "Elliptic Envelope", "Autoencoder", "ECOD", "LODA", "HBOS",
        "PCA+ECOD", "Rolling Z-Score", "Windowed AE", "Windowed RF", "Spectral Residual", "Residual Z",
        "UASE-v2 (proposed)", "UASE-v1 rule (LOF-first)"]


def save(name):
    plt.savefig(FIG / name, dpi=200, bbox_inches="tight"); plt.close()
    shutil.copy(FIG / name, PAP / name)


def color(m):
    return "#0F766E" if m.startswith("UASE-v2") else ("#F59E0B" if m.startswith("UASE-v1") else "#2563EB")


def bar_ci(stats_csv, metric, title, name):
    df = pd.read_csv(stats_csv).set_index("model").sort_values(f"mean_{metric}")
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(df) + 1.2))
    y = np.arange(len(df)); m = df[f"mean_{metric}"].values
    ax.barh(y, m, xerr=[m - df["ci_lo"].values, df["ci_hi"].values - m], color=[color(i) for i in df.index],
            ecolor="#475569", capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([RENAME.get(i, i) for i in df.index])
    ax.set_xlabel(f"mean {metric.upper().replace('_PA', '$_{PA}$')} (95% bootstrap CI over series)")
    ax.set_title(title); ax.set_xlim(0, 1)
    save(name)


def k8s_bars():
    df = pd.read_csv(V2 / "stats" / "k8s_A_bootstrap.csv").sort_values("mcc")
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(df) + 1.2))
    y = np.arange(len(df)); m = df["mcc"].values
    ax.barh(y, m, xerr=[m - df["mcc_ci_lo"].values, df["mcc_ci_hi"].values - m], color=[color(i) for i in df.model],
            ecolor="#475569", capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([RENAME.get(i, i) for i in df.model])
    ax.set_xlabel("test MCC (95% moving-block bootstrap CI)"); ax.set_xlim(0, 1.02)
    ax.set_title("Kubernetes, protocol v2 (train-only Z-score)")
    save("fig_v2_k8s_mcc.png")


def heat(per_series_csv, metric, title, name, order_cols=None):
    df = pd.read_csv(per_series_csv)
    df = df[df.model.isin(MAIN)]
    p = df.groupby(["model", "benchmark"])[metric].mean().unstack()
    p["Mean (series)"] = df.groupby("model")[metric].mean()
    p = p.sort_values("Mean (series)", ascending=False)
    p.index = [RENAME.get(i, i) for i in p.index]
    p.columns = [c.replace("Benchmark", "") for c in p.columns]
    fig, ax = plt.subplots(figsize=(1.0 * p.shape[1] + 4, 0.38 * len(p) + 1.2))
    sns.heatmap(p, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax, cbar=False, annot_kws={"size": 7})
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    save(name)


def v1_v2():
    rows = []
    o = pd.read_csv(ROOT / "data" / "yahoo_s5_overall.csv").set_index("model")
    c = pd.read_csv(ROOT / "data" / "cloud_monitoring_overall.csv").set_index("model")
    k = pd.read_csv(ROOT / "data" / "ml_comparison_results.csv").set_index("model")
    def best(df, col):
        d = df.drop(index=[i for i in df.index if i.startswith("UASE")]); return d[col].max()
    ent = [("Kubernetes\n(F1)", k.loc["UASE (proposed)", "f1"], best(k, "f1"), V2 / "meta" / "k8s_k8s_A_test_overall.csv", "f1"),
           ("Yahoo S5\n(F1$_{PA}$)", o.loc["UASE (proposed)", "f1_pa"], best(o, "f1_pa"), V2 / "meta" / "yahoo_test_overall.csv", "f1_pa"),
           ("Cloud Mon.\n(F1$_{PA}$)", c.loc["UASE (proposed)", "f1_pa"], best(c, "f1_pa"), V2 / "meta" / "cloud_test_overall.csv", "f1_pa")]
    for lab, v1u, v1b, f, col in ent:
        if not f.exists():
            continue
        d = pd.read_csv(f).set_index("model")
        base = d.drop(index=[i for i in d.index if i.startswith("UASE") or i in ("best1_ap", "top3_ap", "top5_ap", "mean_all", "top2_ap") or i.startswith("gtop")])
        rows.append((lab, v1u, v1b, d.loc["UASE-v1 rule (LOF-first)", col], d.loc["UASE-v2 (proposed)", col], base[col].max()))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(rows)); w = 0.15
    labels = ["v1 protocol: UASE (as published)", "v1 protocol: best other model", "v2: UASE-v1 rule",
              "v2: UASE-v2 (proposed)", "v2: best other model"]
    cols = ["#FDBA74", "#CBD5E1", "#F59E0B", "#0F766E", "#2563EB"]
    for j in range(5):
        vals = [r[j + 1] for r in rows]
        ax.bar(x + (j - 2) * w, vals, w, label=labels[j], color=cols[j])
        for xi, v in zip(x + (j - 2) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, ncol=2, loc="upper right"); ax.set_ylabel("score")
    ax.set_title("Legacy (v1) versus leakage-free (v2) protocol")
    save("fig_v1_vs_v2.png")


if __name__ == "__main__":
    st = V2 / "stats"
    k8s_bars()
    for ds, nm in (("cloud", "Cloud Monitoring"), ("yahoo", "Yahoo S5")):
        f = st / f"{ds}_test_f1_pa_stats.csv"
        if f.exists():
            bar_ci(f, "f1_pa", f"{nm}, protocol v2 (test)", f"fig_v2_{ds}_f1pa_ci.png")
            bar_ci(st / f"{ds}_test_f1_stats.csv", "f1", f"{nm}, point-wise F1 (test)", f"fig_v2_{ds}_f1_ci.png")
            heat(V2 / "meta" / f"{ds}_test_per_series.csv", "f1_pa", f"{nm}: mean F1$_{{PA}}$ by subset (protocol v2)",
                 f"fig_v2_{ds}_heatmap.png")
            shutil.copy(st / f"fig_cd_{ds}_test_f1_pa.png", PAP / f"fig_cd_{ds}_test_f1_pa.png")
    v1_v2()
    print("figures ->", PAP)
