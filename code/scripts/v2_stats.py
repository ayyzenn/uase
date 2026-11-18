#!/usr/bin/env python3
"""Statistics for protocol-v2 result files (code/data/v2/*_per_series.csv).

* mean metrics with 95% bootstrap CIs over series
* Wilcoxon signed-rank (paired, per series) of a reference model vs every other
  model, Holm-corrected
* Friedman test + average ranks + Nemenyi critical difference (CD) plot
Writes everything under code/data/v2/stats/ (never touches v1 files).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "v2"

# Nemenyi q_alpha (alpha=0.05), Demsar 2006, k = 2..20 (two-tailed, studentized range / sqrt2)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489,
       19: 3.517, 20: 3.544}


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def cd_plot(avg_ranks: pd.Series, cd: float, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = avg_ranks.sort_values()
    k = len(r)
    fig, ax = plt.subplots(figsize=(9, 0.32 * k + 1.6))
    y = np.arange(k)[::-1]
    colors = ["#0F766E" if n.startswith("UASE") else "#2563EB" for n in r.index]
    ax.barh(y, r.values, color=colors, alpha=0.85)
    best = r.values[0]
    ax.axvline(best + cd, color="#B91C1C", ls="--", lw=1.2, label=f"best + CD ({cd:.2f})")
    ax.set_yticks(y)
    ax.set_yticklabels(r.index)
    for yi, v in zip(y, r.values):
        ax.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyse(csv: Path, ref: str, metric: str, models: list[str] | None, tag: str, fig_title: str):
    df = pd.read_csv(csv)
    if models:
        df = df[df["model"].isin(models)]
    key = ["benchmark", "series"]
    wide = df.pivot_table(index=key, columns="model", values=metric)
    wide = wide.dropna(axis=0, how="any")  # complete blocks only for paired tests
    out_dir = V2 / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in wide.columns:
        lo, hi = boot_ci(wide[m].values)
        rows.append({"model": m, f"mean_{metric}": wide[m].mean(), "ci_lo": lo, "ci_hi": hi,
                     "n_series": len(wide)})
    summ = pd.DataFrame(rows).set_index("model")
    # Wilcoxon ref vs others
    pvals, names, deltas = [], [], []
    for m in wide.columns:
        if m == ref:
            continue
        d = wide[ref] - wide[m]
        if np.allclose(d, 0):
            p = 1.0
        else:
            p = stats.wilcoxon(wide[ref], wide[m], zero_method="wilcox", alternative="two-sided").pvalue
        pvals.append(p); names.append(m); deltas.append(d.mean())
    adj = holm(pvals) if pvals else []
    summ["delta_ref_minus_model"] = np.nan
    summ["wilcoxon_p"] = np.nan
    summ["wilcoxon_p_holm"] = np.nan
    summ["ref_wins"] = np.nan
    summ["ref_losses"] = np.nan
    for n, p, a, dl in zip(names, pvals, adj, deltas):
        summ.loc[n, ["delta_ref_minus_model", "wilcoxon_p", "wilcoxon_p_holm"]] = [dl, p, a]
        summ.loc[n, "ref_wins"] = int((wide[ref] > wide[n]).sum())
        summ.loc[n, "ref_losses"] = int((wide[ref] < wide[n]).sum())
    # Friedman + ranks
    ranks = wide.rank(axis=1, ascending=False, method="average")
    avg = ranks.mean().sort_values()
    summ["avg_rank"] = avg
    k, N = wide.shape[1], wide.shape[0]
    fr = stats.friedmanchisquare(*[wide[c].values for c in wide.columns]) if k >= 3 else None
    cd = Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * N))
    summ = summ.sort_values(f"mean_{metric}", ascending=False)
    summ.to_csv(out_dir / f"{tag}_{metric}_stats.csv")
    meta = {"csv": str(csv.resolve().relative_to(ROOT)), "metric": metric, "reference": ref, "n_series": int(N),
            "k_models": int(k), "friedman_stat": float(fr.statistic) if fr else None,
            "friedman_p": float(fr.pvalue) if fr else None, "nemenyi_cd_0.05": float(cd)}
    (out_dir / f"{tag}_{metric}_stats.json").write_text(json.dumps(meta, indent=2))
    cd_plot(avg, cd, out_dir / f"fig_cd_{tag}_{metric}.png", fig_title)
    print(f"== {tag} / {metric}  (N={N}, k={k}, Friedman p={meta['friedman_p']}, CD={cd:.3f})")
    print(summ.round(4).to_string())
    return summ, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metric", default="f1_pa")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    analyse(Path(a.csv), a.ref, a.metric, a.models, a.tag, a.title or a.tag)
