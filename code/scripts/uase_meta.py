#!/usr/bin/env python3
"""
UASE-v2 meta-layer study on saved channel scores (protocol v2).

Reads data/v2/scores/<dataset>_<mode>/*.npz written by `uase_v2.py --save-scores`.
Each file holds, per channel, out-of-fold (OOF) training scores and test scores,
plus the training labels (and test labels, used ONLY in the final evaluation).

Pre-declared selection protocol (fixed before any test result was inspected):
  1. Candidate configurations
       per-series rules : legacy_lof_first (v1 selector re-implemented on OOF scores),
                          best1_ap, top{2,3,5}_ap, mean_all   (channel quality = OOF AP)
       corpus-level     : single:<channel>  (one channel for the whole corpus)
                          gtop{k}: equal-weight rank fusion of the k channels with the
                          highest mean development score, k = 2..5
  2. Every configuration is scored on the DEVELOPMENT runs (nested split inside each
     training prefix; test segments never enter).  Objective = mean F1_PA for the
     univariate corpora, mean point-wise F1 for Kubernetes.
  3. UASE-v2 = argmax configuration (ties -> fewer channels).  It is then applied once
     to the test runs.  A 5-fold cross-validation over development series estimates how
     well the selection step itself generalises.
Outputs: data/v2/meta/<dataset>_*.csv / .json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import uase_v2 as u

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data" / "v2" / "scores"
OUTD = ROOT / "data" / "v2" / "meta"
SUPERVISED = {"Windowed RF"}
LEGACY_CAND = ("Isolation Forest", "LOF", "ECOD", "LODA", "HBOS", "Spectral Residual",
               "PCA+ECOD", "Autoencoder", "Elliptic Envelope")


def load(dataset: str, mode: str, bench_filter: str | None = None):
    d = SC / f"{dataset}_{mode}"
    items = []
    for f in sorted(d.glob("*.npz")):
        bench, series = f.stem.split("__", 1)
        if bench_filter and bench != bench_filter:
            continue
        z = np.load(f)
        ch = {}
        for k in z.files:
            if k.startswith("oof::"):
                name = k[5:]
                ch[name] = (z[k], z["te::" + name])
        items.append({"benchmark": bench, "series": series, "ytr": z["ytr"].astype(int),
                      "yte": z["yte"].astype(int), "ch": ch})
    return items


def fuse(ch, names):
    names = [n for n in names if n in ch]
    if not names:
        return None
    fo = np.mean([u.rank01(ch[n][0]) for n in names], 0)
    ft = np.mean([u.ecdf(ch[n][0], ch[n][1]) for n in names], 0)
    return fo, ft


def per_series_rules(it):
    ch, ytr = it["ch"], it["ytr"]
    names = sorted(ch)
    out = {}
    if ytr.sum() == 0:
        return out
    ap = {k: float(average_precision_score(ytr, ch[k][0])) for k in names}
    by_ap = sorted(names, key=lambda k: (-ap[k], k))
    out["best1_ap"] = (ch[by_ap[0]][0], ch[by_ap[0]][1], by_ap[0])
    for kk in (2, 3, 5):
        out[f"top{kk}_ap"] = (*fuse(ch, by_ap[:kk]), "+".join(by_ap[:kk]))
    out["mean_all"] = (*fuse(ch, names), "all")
    return out


def legacy_rule(it, pa):
    ch, ytr = it["ch"], it["ytr"]
    cand = [k for k in LEGACY_CAND if k in ch]
    if not cand or ytr.sum() == 0:
        return None
    pf = {k: u.oof_f1(ytr, ch[k][0], pa) for k in cand}
    best = max(cand, key=lambda k: pf[k])
    if "LOF" in pf and pf["LOF"] >= 0.05:
        best = "LOF"
        if "Isolation Forest" in pf and pf["Isolation Forest"] >= pf["LOF"] + 0.15:
            best = "Isolation Forest"
    return ch[best][0], ch[best][1], best


def evaluate_cfg(it, cfg, pa, univariate, global_rank=None):
    """Return metrics dict (on the item's 'test' part) for one configuration."""
    ch, ytr, yte = it["ch"], it["ytr"], it["yte"]
    info = cfg
    if cfg == "legacy_lof_first":
        r = legacy_rule(it, pa)
        if r is None:
            return None
        fo, ft, info = r
    elif cfg.startswith("single:"):
        n = cfg.split(":", 1)[1]
        if n not in ch:
            return None
        fo, ft = ch[n]
    elif cfg.startswith("gtop"):
        k = int(cfg[4:])
        sel = [n for n in global_rank if n in ch][:k]
        r = fuse(ch, sel)
        if r is None:
            return None
        fo, ft = r
        info = "+".join(sel)
    else:
        rules = per_series_rules(it)
        if cfg not in rules:
            return None
        fo, ft, info = rules[cfg]
    thr = u.calib(ytr, fo, pa)
    pred = (ft >= thr).astype(int)
    met = u.metrics_row(yte, pred, ft, univariate)
    try:
        met["pr_auc"] = float(average_precision_score(yte, ft))
    except Exception:
        met["pr_auc"] = np.nan
    met["uase_choice"] = info
    return met


def table(items, cfgs, pa, univariate, global_rank):
    rows = []
    for it in items:
        for c in cfgs:
            m = evaluate_cfg(it, c, pa, univariate, global_rank)
            if m is None:
                continue
            rows.append({"benchmark": it["benchmark"], "series": it["series"], "model": c, **m})
    return pd.DataFrame(rows)


def channel_names(items):
    s = set()
    for it in items:
        s |= set(it["ch"])
    return sorted(s)


def select(dev_items, pa, univariate, obj):
    chans = channel_names(dev_items)
    singles = [f"single:{c}" for c in chans]
    df = table(dev_items, singles, pa, univariate, None)
    # complete-case: a channel missing on a series scores 0 there (it could not be used)
    piv = df.pivot_table(index=["benchmark", "series"], columns="model", values=obj).fillna(0.0)
    g = piv.mean().sort_values(ascending=False)
    global_rank = [c.split(":", 1)[1] for c in g.index]
    cfgs = singles + [f"gtop{k}" for k in range(2, 6)] + ["legacy_lof_first", "best1_ap", "top2_ap",
                                                          "top3_ap", "top5_ap", "mean_all"]
    df2 = table(dev_items, [c for c in cfgs if not c.startswith("single:")], pa, univariate, global_rank)
    full = pd.concat([df, df2], ignore_index=True)
    piv = full.pivot_table(index=["benchmark", "series"], columns="model", values=obj).fillna(0.0)
    means = piv.mean()

    def n_ch(c):
        if c.startswith("single:") or c in ("legacy_lof_first", "best1_ap"):
            return 1
        if c.startswith("gtop"):
            return int(c[4:])
        if c.startswith("top"):
            return int(c[3])
        return 99
    best = sorted(means.index, key=lambda c: (-round(means[c], 10), n_ch(c), c))[0]
    return best, global_rank, means.sort_values(ascending=False), full


def cv_selection(dev_items, pa, univariate, obj, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dev_items))
    parts = np.array_split(idx, folds)
    rows = []
    for f, held in enumerate(parts):
        train = [dev_items[i] for i in idx if i not in set(held)]
        test = [dev_items[i] for i in held]
        best, grank, _, _ = select(train, pa, univariate, obj)
        df = table(test, [best], pa, univariate, grank)
        rows.append({"fold": f, "selected": best, "n_heldout": len(test),
                     f"heldout_mean_{obj}": float(df[obj].mean()) if len(df) else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="yahoo | cloud | k8s")
    ap.add_argument("--bench", default=None, help="k8s only: k8s_A or k8s_D")
    ap.add_argument("--evaluate-test", action="store_true",
                    help="apply the dev-selected configuration to the test runs")
    ap.add_argument("--cv", action="store_true")
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    univariate = a.dataset != "k8s"
    pa = univariate
    obj = "f1_pa" if univariate else "f1"
    tag = a.dataset + (f"_{a.bench}" if a.bench else "")
    dev = load(a.dataset, "dev", a.bench)
    best, grank, means, full = select(dev, pa, univariate, obj)
    full.to_csv(OUTD / f"{tag}_dev_configs_per_series.csv", index=False)
    means.rename(f"dev_mean_{obj}").to_csv(OUTD / f"{tag}_dev_config_means.csv")
    print(f"[{tag}] dev series: {len(dev)}; global channel ranking: {grank}")
    print(means.round(4).to_string())
    print(f"[{tag}] SELECTED on dev: {best}")
    sel = {"dataset": a.dataset, "bench": a.bench, "objective": obj, "n_dev_series": len(dev),
           "selected_config": best, "global_rank_dev": grank,
           "selected_channels": grank[: int(best[4:])] if best.startswith("gtop") else None,
           "dev_mean_of_selected": float(means[best])}
    if a.cv and len(dev) >= 10:
        cvdf = cv_selection(dev, pa, univariate, obj)
        cvdf.to_csv(OUTD / f"{tag}_dev_selection_cv.csv", index=False)
        print(cvdf.to_string())
        sel["cv_heldout_mean"] = float(cvdf[f"heldout_mean_{obj}"].mean())
    if a.evaluate_test:
        test = load(a.dataset, "test", a.bench)
        chans = channel_names(test)
        cfgs = [f"single:{c}" for c in chans] + ["legacy_lof_first", "best1_ap", "top3_ap", "top5_ap",
                                                 "mean_all", best]
        cfgs = list(dict.fromkeys(cfgs))
        df = table(test, cfgs, pa, univariate, grank)
        df["model"] = df["model"].str.replace("single:", "", regex=False)
        name_map = {best.replace("single:", ""): "UASE-v2 (proposed)",
                    "legacy_lof_first": "UASE-v1 rule (LOF-first)"}
        if best.startswith("single:"):
            # keep the plain channel row as well
            extra = df[df["model"] == best.replace("single:", "")].copy()
            extra["model"] = "UASE-v2 (proposed)"
            df = pd.concat([df, extra], ignore_index=True)
            name_map.pop(best.replace("single:", ""))
        df["model"] = df["model"].replace(name_map)
        df.to_csv(OUTD / f"{tag}_test_per_series.csv", index=False)
        cols = [c for c in ["f1_pa", "mcc_pa", "f1", "mcc", "precision", "recall", "roc_auc", "pr_auc"]
                if c in df.columns]
        ov = df.groupby("model")[cols].mean().sort_values(obj, ascending=False)
        ov["n_series"] = df.groupby("model").size()
        ov.to_csv(OUTD / f"{tag}_test_overall.csv")
        if univariate:
            bb = df.groupby(["benchmark", "model"])[cols].mean().reset_index()
            bb.to_csv(OUTD / f"{tag}_test_by_benchmark.csv", index=False)
        print(ov.round(4).to_string())
        sel["test_mean_of_selected"] = float(ov.loc["UASE-v2 (proposed)", obj])
    (OUTD / f"{tag}_selection.json").write_text(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
