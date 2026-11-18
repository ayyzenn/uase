#!/usr/bin/env python3
"""Generate LaTeX tables for paper_final.tex directly from protocol-v2 result files
(no hand transcription). Writes paper/tables/*.tex"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"
OUT = ROOT.parent / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
MAIN = ["Isolation Forest", "LOF", "One-Class SVM", "Elliptic Envelope", "Autoencoder", "ECOD", "LODA", "HBOS",
        "PCA+ECOD", "Rolling Z-Score", "Windowed AE", "Windowed RF", "Spectral Residual", "Residual Z",
        "UASE-v2 (proposed)", "UASE-v1 rule (LOF-first)"]
REN = {"Residual Z": "Robust residual", "UASE-v2 (proposed)": "\\textbf{UASE-v2 (proposed)}",
       "UASE-v1 rule (LOF-first)": "UASE-v1 rule (LOF-first)"}


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def pfmt(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def bold_max(vals):
    m = np.nanmax(vals)
    return [f"\\textbf{{{f(v)}}}" if np.isclose(v, m) else f(v) for v in vals]


def k8s_table(norm="A"):
    b = pd.read_csv(V2 / "stats" / f"k8s_{norm}_bootstrap.csv").set_index("model")
    o = pd.read_csv(V2 / "meta" / f"k8s_k8s_{norm}_test_overall.csv").set_index("model")
    rows = [m for m in MAIN if m in b.index]
    b = b.loc[rows].sort_values("mcc", ascending=False)
    lines = []
    cols = {c: bold_max(b[c].values) for c in ["f1", "mcc"]}
    prec = bold_max(o.loc[b.index, "precision"].values); rec = bold_max(o.loc[b.index, "recall"].values)
    ap = bold_max(o.loc[b.index, "pr_auc"].values)
    for i, m in enumerate(b.index):
        r = b.loc[m]
        p = "--" if m == "UASE-v2 (proposed)" else pfmt(r.mcnemar_p)
        lines.append(f"{REN.get(m, m)} & {prec[i]} & {rec[i]} & {cols['f1'][i]} & {cols['mcc'][i]} & "
                     f"[{f(r.mcc_ci_lo)}, {f(r.mcc_ci_hi)}] & {ap[i]} & {int(r.tp)}/{int(r.fp)}/{int(r.fn)} & {p} \\\\")
    (OUT / f"tab_k8s_{norm}.tex").write_text("\n".join(lines) + "\n")


def uni_table(ds):
    st = {m: pd.read_csv(V2 / "stats" / f"{ds}_test_{m}_stats.csv").set_index("model") for m in ("f1_pa", "mcc_pa", "f1", "mcc")}
    o = pd.read_csv(V2 / "meta" / f"{ds}_test_overall.csv").set_index("model")
    s = st["f1_pa"].sort_values("mean_f1_pa", ascending=False)
    idx = list(s.index)
    cF = bold_max(s["mean_f1_pa"].values); cM = bold_max(st["mcc_pa"].loc[idx, "mean_mcc_pa"].values)
    cf = bold_max(st["f1"].loc[idx, "mean_f1"].values); cm = bold_max(st["mcc"].loc[idx, "mean_mcc"].values)
    ca = bold_max(o.loc[idx, "pr_auc"].values)
    lines = []
    for i, m in enumerate(idx):
        r = s.loc[m]
        p = "--" if m == "UASE-v2 (proposed)" else pfmt(r.wilcoxon_p_holm)
        lines.append(f"{REN.get(m, m)} & {cF[i]} & [{f(r.ci_lo)}, {f(r.ci_hi)}] & {cM[i]} & {cf[i]} & {cm[i]} & {ca[i]} & "
                     f"{r.avg_rank:.2f} & {p} \\\\")
    (OUT / f"tab_{ds}.tex").write_text("\n".join(lines) + "\n")
    meta = json.loads((V2 / "stats" / f"{ds}_test_f1_pa_stats.json").read_text())
    (OUT / f"tab_{ds}_meta.tex").write_text(
        f"% N={meta['n_series']} k={meta['k_models']} Friedman chi2={meta['friedman_stat']:.2f} p={meta['friedman_p']:.2e} CD={meta['nemenyi_cd_0.05']:.3f}\n")
    if ds == "yahoo":
        ps = pd.read_csv(V2 / "meta" / "yahoo_test_per_series.csv")
        ps = ps[ps.model.isin(idx)]
        w = ps.pivot_table(index=["benchmark", "series"], columns="model", values="f1_pa").dropna()
        bb = w.groupby(level=0).mean().T  # complete-case series, same set as the statistics
        bb = bb.loc[idx]
        cols = {c: bold_max(bb[c].values) for c in bb.columns}
        n = w.groupby(level=0).size()
        L = []
        for i, m in enumerate(idx):
            L.append(f"{REN.get(m, m)} & " + " & ".join(cols[c][i] for c in bb.columns) + f" & {f(s.loc[m, 'mean_f1_pa'])} \\\\")
        (OUT / "tab_yahoo_bench.tex").write_text("\n".join(L) + "\n% n per benchmark: " + n.to_dict().__repr__() + "\n")


def meta_table():
    L = []
    for ds, tag, obj in (("Kubernetes (norm A)", "k8s_k8s_A", "f1"), ("Kubernetes (norm D)", "k8s_k8s_D", "f1"),
                         ("Yahoo S5", "yahoo", "f1_pa"), ("Cloud Monitoring", "cloud", "f1_pa")):
        means = pd.read_csv(V2 / "meta" / f"{tag}_dev_config_means.csv").set_index("model").iloc[:, 0]
        sel = json.loads((V2 / "meta" / f"{tag}_selection.json").read_text())
        best = sel["selected_config"]
        def g(c):
            return f(means.get(c, np.nan))
        bs = means[[c for c in means.index if c.startswith("single:")]]
        L.append(f"{ds} & {sel['n_dev_series']} & {bs.idxmax().replace('single:', '').replace('Residual Z', 'Robust residual')} ({f(bs.max())}) & "
                 f"{g('legacy_lof_first')} & {g('best1_ap')} & {g('top3_ap')} & {g('mean_all')} & "
                 f"{max(means[[c for c in means.index if c.startswith('gtop')]]):.3f} & "
                 f"\\texttt{{{best.replace("_", "\\_").replace('single:', '').replace('Residual Z', 'robust residual')}}} \\\\")
    (OUT / "tab_meta_dev.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    k8s_table("A"); k8s_table("D")
    for ds in ("cloud", "yahoo"):
        if (V2 / "stats" / f"{ds}_test_f1_pa_stats.csv").exists():
            uni_table(ds)
    meta_table()
    for p in sorted(OUT.glob("*.tex")):
        print("==", p.name); print(p.read_text())
