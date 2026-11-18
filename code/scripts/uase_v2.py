#!/usr/bin/env python3
"""
UASE-v2: out-of-fold (OOF) stacked Unified Adaptive Stacking Ensemble under a
leakage-free evaluation protocol ("protocol v2").

Protocol v2 (applies to EVERY model, baselines included):
  * same chronological split as v1 (choose_split for Yahoo/Cloud, 70/30 for K8s);
  * scaler fit on the training prefix only (no full-series Z-score);
  * fixed default hyperparameters (the v1 Yahoo grid-search JSON is NOT used,
    because those grids were scored on test splits of 32 series);
  * every detector's decision threshold is calibrated on *out-of-fold* training
    scores (K contiguous blocks of the training prefix; each block is scored by a
    model fit on the other blocks), then applied to test scores of the model refit
    on the whole training prefix. Test labels are never used.

UASE-v2 meta-layer: channels are ECDF-normalised against their OOF scores;
channel quality is estimated on OOF scores; a fusion *rule* turns the channel
matrix into one score; the rule's threshold is calibrated on the fused OOF score.
The rule is chosen ONCE, globally, by a nested development run that re-applies
the whole pipeline inside the training prefixes only (--mode dev).

Outputs go to code/data/v2/ (never overwrites v1 files).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from sklearn.covariance import EllipticEnvelope  # noqa: E402
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score  # noqa: E402
from sklearn.neighbors import LocalOutlierFactor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

from anomaly_models import (  # noqa: E402
    AutoencoderDetector,
    ECODDetector,
    HBOSDetector,
    LODADetector,
    PCAHybridDetector,
    RollingZScoreDetector,
    WindowedAutoencoder,
    evaluate,
    spectral_residual_scores,
)
import evaluate_yahoo_s5 as ey  # noqa: E402

OUT = ROOT / "data" / "v2"
SAVE_DIR = None  # set by --save-scores
K_FOLDS = 4
FUSE_ALPHAS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]

# Residual feature columns of featurize_univariate (see evaluate_yahoo_s5)
RES_COLS = [20, 22, 23, 4, 26, 27, 29, 30]

CFG = {
    "univariate": dict(iso_n=100, lof_k=20, ee_sf=0.8, ae_hidden=(16, 8, 16), ae_iter=40,
                       loda_cuts=50, loda_bins=20, hbos_bins=20, rz_w=24, win=24,
                       rf_n=200, rf_depth=None, rf_leaf=1, wae_hidden=(32, 12, 32), wae_iter=35,
                       pa=True),
    "k8s": dict(iso_n=300, lof_k=25, ee_sf=0.7, ae_hidden=(32, 16, 32), ae_iter=80,
                loda_cuts=100, loda_bins=20, hbos_bins=20, rz_w=60, win=16,
                rf_n=500, rf_depth=16, rf_leaf=2, wae_hidden=(64, 24, 64), wae_iter=60,
                pa=False),
}


# ----------------------------------------------------------------------------- utils
def rank01(a):
    a = np.asarray(a, float)
    return a.argsort().argsort().astype(float) / max(len(a) - 1, 1)


def ecdf(ref, vals):
    r = np.sort(np.asarray(ref, float))
    return np.searchsorted(r, vals, side="right") / max(len(r), 1)


def calib(y, s, pa):
    return ey.calibrate_threshold(y, s, point_adjust_obj=pa)


def f1_obj(y, pred, pa):
    if pa:
        pred = ey.point_adjust(y, pred)
    return float(f1_score(y, pred, zero_division=0))


def oof_f1(y, s, pa):
    if y.sum() == 0:
        return 0.0
    t = calib(y, s, pa)
    return f1_obj(y, (s >= t).astype(int), pa)


def blocks(n, k):
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(k)]


def windowize(Xs, win):
    """One flattened window per time step (front-padded) so windows align with time."""
    pad = np.repeat(Xs[:1], win - 1, axis=0)
    Z = np.vstack([pad, Xs])
    n, d = Xs.shape
    idx = np.arange(n)[:, None] + np.arange(win)[None, :]
    return Z[idx].reshape(n, win * d)


# ----------------------------------------------------------------------------- channels
def _unsup_factories(c, contam, n_normal):
    k = min(c["lof_k"], max(5, n_normal // 10)) if c["pa"] else c["lof_k"]
    cont = float(min(0.5, max(0.01, contam)))

    def iso():
        return IsolationForest(n_estimators=c["iso_n"], contamination=cont, random_state=42, n_jobs=1)

    def lof():
        return LocalOutlierFactor(n_neighbors=k, contamination=cont, novelty=True)

    def oc():
        return OneClassSVM(kernel="rbf", gamma="scale", nu=cont)

    def ee():
        return EllipticEnvelope(contamination=cont, support_fraction=c["ee_sf"], random_state=42)

    def ae():
        m = AutoencoderDetector(hidden=c["ae_hidden"], contamination=cont, random_state=42)
        m.model.max_iter = c["ae_iter"]
        return m

    return {
        "Isolation Forest": (iso, "neg_decision"),
        "LOF": (lof, "neg_decision"),
        "One-Class SVM": (oc, "neg_decision"),
        "Elliptic Envelope": (ee, "neg_decision"),
        "Autoencoder": (ae, "score"),
        "ECOD": (lambda: ECODDetector(contamination=cont), "score"),
        "LODA": (lambda: LODADetector(n_random_cuts=c["loda_cuts"], n_bins=c["loda_bins"],
                                      contamination=cont, random_state=42), "score"),
        "HBOS": (lambda: HBOSDetector(n_bins=c["hbos_bins"], contamination=cont), "score"),
        "PCA+ECOD": (lambda: PCAHybridDetector(n_components=0.95, contamination=cont, base="ecod",
                                               random_state=42), "score"),
    }


def _score(model, kind, X):
    return -model.decision_function(X) if kind == "neg_decision" else model.score_samples(X)


def _fit_unsup(fac, kind, Xfit, Xscore):
    m = fac()
    m.fit(Xfit)
    return _score(m, kind, Xscore)


def _residual(Xs, tr_mask):
    parts = []
    for j in RES_COLS:
        if Xs.shape[1] > j:
            a = np.abs(Xs[:, j])
            mu, sd = a[tr_mask].mean(), a[tr_mask].std() + 1e-8
            parts.append((a - mu) / sd)
    return np.mean(parts, axis=0)


def channel_scores(X_raw, y, split, c, univariate):
    """Return dict name -> (oof_train_scores[split], test_scores[n-split])."""
    n = len(y)
    scaler = StandardScaler().fit(X_raw[:split])
    Xs = np.nan_to_num(scaler.transform(X_raw), nan=0.0, posinf=0.0, neginf=0.0)
    Xtr, Xte, ytr = Xs[:split], Xs[split:], y[:split]
    contam = float(np.clip(max(ytr.mean(), 1.0 / max(split, 1)), 0.01, 0.2))
    normal = ytr == 0
    folds = blocks(split, K_FOLDS)
    out = {}

    facs = _unsup_factories(c, contam, int(normal.sum()))
    for name, (fac, kind) in facs.items():
        try:
            oof = np.zeros(split)
            for a, b in folds:
                fit_mask = normal.copy()
                fit_mask[a:b] = False
                if fit_mask.sum() < 15:
                    fit_mask = normal.copy()
                oof[a:b] = _fit_unsup(fac, kind, Xtr[fit_mask], Xtr[a:b])
            te = _fit_unsup(fac, kind, Xtr[normal], Xte)
            out[name] = (oof, te)
        except Exception as exc:  # e.g. singular covariance
            print(f"    channel {name} failed: {exc}", flush=True)

    # Rolling z-score: moments from normal train rows (fit is cheap -> OOF too)
    try:
        oof = np.zeros(split)
        for a, b in folds:
            fm = normal.copy(); fm[a:b] = False
            rz = RollingZScoreDetector(window=c["rz_w"], contamination=contam).fit(Xtr[fm], None)
            oof[a:b] = rz.score_samples(Xtr)[a:b]
        rz = RollingZScoreDetector(window=c["rz_w"], contamination=contam).fit(Xtr, ytr)
        out["Rolling Z-Score"] = (oof, rz.score_samples(Xte))
    except Exception as exc:
        print(f"    channel RZ failed: {exc}", flush=True)

    if univariate:
        # Spectral residual (label-free; train prefix and full series as in v1)
        raw = X_raw[:, 0]
        sr_tr = spectral_residual_scores(raw[:split])
        sr_all = spectral_residual_scores(raw)
        out["Spectral Residual"] = (sr_tr, sr_all[split:])
        res = _residual(Xs, np.arange(n) < split)
        out["Residual Z"] = (res[:split], res[split:])

    # Windowed models (time-aligned)
    win = int(max(5, min(c["win"], max(5, split // 3))))
    W = windowize(Xs, win)
    Wtr, Wte = W[:split], W[split:]
    try:
        def wae_fit_score(Wfit, Wsc):
            m = WindowedAutoencoder(window=win, hidden=c["wae_hidden"], contamination=contam, random_state=42)
            m.model.max_iter = c["wae_iter"]
            m.fit(Wfit)
            return m.score_samples(Wsc)
        oof = np.zeros(split)
        for a, b in folds:
            fm = normal.copy(); fm[a:b] = False
            oof[a:b] = wae_fit_score(Wtr[fm], Wtr[a:b])
        out["Windowed AE"] = (oof, wae_fit_score(Wtr[normal], Wte))
    except Exception as exc:
        print(f"    channel WAE failed: {exc}", flush=True)

    if ytr.sum() > 0:
        def rf_fit_score(Wfit, yfit, Wsc):
            if len(np.unique(yfit)) < 2:
                return np.zeros(len(Wsc))
            rf = RandomForestClassifier(n_estimators=c["rf_n"], max_depth=c["rf_depth"],
                                        min_samples_leaf=c["rf_leaf"], class_weight="balanced_subsample",
                                        max_features="sqrt", random_state=42, n_jobs=1)
            rf.fit(Wfit, yfit)
            return rf.predict_proba(Wsc)[:, list(rf.classes_).index(1)]
        oof = np.zeros(split)
        for a, b in folds:
            fm = np.ones(split, bool); fm[a:b] = False
            oof[a:b] = rf_fit_score(Wtr[fm], ytr[fm], Wtr[a:b])
        out["Windowed RF"] = (oof, rf_fit_score(Wtr, ytr, Wte))

    # Residual fusion for LOF and Windowed RF (as in v1), alpha picked on OOF scores
    if univariate and "Residual Z" in out:
        r_oof, r_te = out["Residual Z"]
        for name, rank_model in (("LOF", True), ("Windowed RF", True)):
            if name not in out or ytr.sum() == 0:
                continue
            m_oof, m_te = out[name]
            a = ey._pick_fusion_alpha(m_oof, r_oof, ytr, FUSE_ALPHAS, default_alpha=0.65,
                                      rank_model=rank_model)
            f_oof = a * rank01(m_oof) + (1 - a) * rank01(r_oof)
            f_te = a * ecdf(m_oof, m_te) + (1 - a) * ecdf(r_oof, r_te)
            out[name] = (f_oof, f_te)
    return out


# ----------------------------------------------------------------------------- UASE meta
RULES = ["legacy_lof_first", "best1_pa", "best1_ap", "top2_ap", "top3_ap", "top5_ap",
         "top3_pa", "mean_all", "adaptive_pa"]
SUPERVISED = {"Windowed RF"}


def uase_rules(ch, ytr, pa):
    """ch: name -> (oof, te). Returns rule -> (fused_oof, fused_te, info)."""
    names = sorted(ch)
    Zo = {k: rank01(ch[k][0]) for k in names}
    Zt = {k: ecdf(ch[k][0], ch[k][1]) for k in names}
    has_pos = ytr.sum() > 0
    if has_pos:
        ap = {k: float(average_precision_score(ytr, ch[k][0])) for k in names}
        pf = {k: oof_f1(ytr, ch[k][0], pa) for k in names}
    else:
        ap = {k: 0.0 for k in names}
        pf = {k: 0.0 for k in names}

    def fuse(sel):
        return np.mean([Zo[k] for k in sel], 0), np.mean([Zt[k] for k in sel], 0)

    unsup = [k for k in names if k not in SUPERVISED]
    res = {}
    if not has_pos:
        fo, ft = fuse(unsup)
        for r in RULES:
            res[r] = (fo, ft, "fallback:mean_unsup")
        return res, ap, pf
    by_ap = sorted(names, key=lambda k: (-ap[k], k))
    by_pf = sorted(names, key=lambda k: (-pf[k], -ap[k], k))
    # v1 selector, re-implemented on OOF scores
    cand = [k for k in ("Isolation Forest", "LOF", "ECOD", "LODA", "HBOS", "Spectral Residual",
                        "PCA+ECOD", "Autoencoder", "Elliptic Envelope") if k in ch]
    best = max(cand, key=lambda k: pf[k])
    if "LOF" in pf and pf["LOF"] >= 0.05:
        best = "LOF"
        if "Isolation Forest" in pf and pf["Isolation Forest"] >= pf["LOF"] + 0.15:
            best = "Isolation Forest"
    res["legacy_lof_first"] = (ch[best][0], ch[best][1], best)
    res["best1_pa"] = (ch[by_pf[0]][0], ch[by_pf[0]][1], by_pf[0])
    res["best1_ap"] = (ch[by_ap[0]][0], ch[by_ap[0]][1], by_ap[0])
    for kk in (2, 3, 5):
        sel = by_ap[:kk]
        res[f"top{kk}_ap"] = (*fuse(sel), "+".join(sel))
    sel = by_pf[:3]
    res["top3_pa"] = (*fuse(sel), "+".join(sel))
    res["mean_all"] = (*fuse(names), "all")
    # adaptive: pick among singles and AP-ranked fusions by OOF objective
    pool = {f"single:{k}": (ch[k][0], ch[k][1]) for k in names}
    for kk in (2, 3, 5):
        pool[f"top{kk}_ap"] = res[f"top{kk}_ap"][:2]
    pool["mean_all"] = res["mean_all"][:2]
    scored = {m: oof_f1(ytr, v[0], pa) for m, v in pool.items()}
    pick = max(scored, key=lambda m: (scored[m], 0 if m.startswith("single") else 1))
    res["adaptive_pa"] = (*pool[pick], pick)
    return res, ap, pf


# ----------------------------------------------------------------------------- per series
def metrics_row(yt, pred, score, pa_eval=True):
    met = evaluate(yt, pred, score)
    if pa_eval:
        y_pa = ey.point_adjust(yt, pred)
        mp = evaluate(yt, y_pa, score)
        met.update(f1_pa=mp["f1"], mcc_pa=mp["mcc"], precision_pa=mp["precision"], recall_pa=mp["recall"])
    return met


def process(X_raw, y, split, cfg_name, univariate, meta):
    c = CFG[cfg_name]
    pa = c["pa"]
    t0 = time.time()
    ch = channel_scores(X_raw, y, split, c, univariate)
    ytr, yte = y[:split], y[split:]
    if SAVE_DIR is not None:
        # Persist channel scores so the meta-layer can be studied without refitting
        # (scores only; test labels are stored for the final evaluation step).
        d = SAVE_DIR / f"{meta['dataset']}_{meta['mode']}{meta.get('tag', '')}"
        d.mkdir(parents=True, exist_ok=True)
        arrs = {"ytr": ytr.astype(np.int8), "yte": yte.astype(np.int8)}
        for k, (o, t) in ch.items():
            arrs[f"oof::{k}"] = np.asarray(o, np.float64)
            arrs[f"te::{k}"] = np.asarray(t, np.float64)
        np.savez_compressed(d / f"{meta['benchmark']}__{meta['series']}.npz", **arrs)
    rows = []
    for name, (oof, te) in ch.items():
        thr = calib(ytr, oof, pa)
        rows.append({**meta, "model": name, **metrics_row(yte, (te >= thr).astype(int), te, univariate)})
    rules, ap, pf = uase_rules(ch, ytr, pa)
    for r, (fo, ft, info) in rules.items():
        thr = calib(ytr, fo, pa)
        rows.append({**meta, "model": f"UASE-v2[{r}]", "uase_choice": info,
                     **metrics_row(yte, (ft >= thr).astype(int), ft, univariate)})
    for row in rows:
        row["seconds"] = round(time.time() - t0, 2)
    return rows


# ----------------------------------------------------------------------------- datasets
def yahoo_items():
    for bench in ("A1Benchmark", "A2Benchmark", "A3Benchmark", "A4Benchmark"):
        for p in ey.list_series(bench):
            yield "yahoo", bench, p


def cloud_items():
    import evaluate_cloud_monitoring as ecm
    for dom, p in ecm.discover_all_series():
        yield "cloud", dom, p


def load_uni(dataset, path):
    if dataset == "yahoo":
        v, y = ey.load_series(path)
    else:
        import evaluate_cloud_monitoring as ecm
        v, y = ecm.load_series(path)
    X = np.nan_to_num(ey.featurize_univariate(v), nan=0.0, posinf=0.0, neginf=0.0)
    return X, y.astype(int)


def run_uni_item(dataset, bench, path, mode, save=False):
    global SAVE_DIR
    if save:
        SAVE_DIR = OUT / "scores"
    try:
        X, y = load_uni(dataset, path)
        if len(X) < 80 or y.sum() == 0:
            return []
        split = ey.choose_split(y)
        if split is None or len(np.unique(y[split:])) < 2 or (y[:split] == 0).sum() < 20:
            return []
        if mode == "dev":
            # nested development split INSIDE the training prefix only
            X, y = X[:split], y[:split]
            if y.sum() < 2:
                return []
            s2 = ey.choose_split(y)
            if s2 is None or len(np.unique(y[s2:])) < 2 or (y[:s2] == 0).sum() < 20:
                return []
            split = s2
        meta = {"dataset": dataset, "benchmark": bench, "series": path.name, "mode": mode}
        return process(X, y, split, "univariate", True, meta)
    except Exception as exc:
        print(f"  skip {path.name}: {exc}", flush=True)
        return []


def run_k8s(mode, norm):
    from train_compare_models import enrich_k8s_features, load_data
    from timeseries_normalization import apply_normalizer
    df = load_data()
    X, _ = enrich_k8s_features(df)
    y = df["label"].values.astype(int)
    split = int(0.7 * len(df))
    if mode == "dev":
        X, y, split = X[:split], y[:split], int(0.7 * split)
    if norm == "D":
        X = apply_normalizer("D_causal_rolling_z60", X, split).X_scaled
    meta = {"dataset": "k8s", "benchmark": f"k8s_{norm}", "series": "k8s_metrics.csv", "mode": mode}
    return process(X, y, split, "k8s", False, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["yahoo", "cloud", "k8s"], required=True)
    ap.add_argument("--mode", choices=["dev", "test"], default="test")
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--norm", default="A", help="K8s only: A=train-only Z, D=causal rolling robust Z")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-scores", action="store_true", help="dump per-series channel scores to data/v2/scores/")
    a = ap.parse_args()
    global SAVE_DIR
    if a.save_scores:
        SAVE_DIR = OUT / "scores"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.dataset == "k8s":
        rows = run_k8s(a.mode, a.norm)
        name = f"k8s_{a.norm}_{a.mode}{a.tag}"
    else:
        items = list(yahoo_items() if a.dataset == "yahoo" else cloud_items())
        if a.max_series:
            items = items[: a.max_series]
        print(f"{a.dataset} {a.mode}: {len(items)} candidate series", flush=True)
        res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_uni_item)(d, b, p, a.mode, a.save_scores) for d, b, p in items)
        rows = [r for rr in res for r in rr]
        name = f"{a.dataset}_{a.mode}{a.tag}"
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}_per_series.csv", index=False)
    col = "f1_pa" if "f1_pa" in df.columns else "f1"
    cols = [x for x in ["f1_pa", "mcc_pa", "f1", "mcc", "precision", "recall", "roc_auc"] if x in df.columns]
    ov = df.groupby("model")[cols].mean().sort_values(col, ascending=False)
    ov["n_series"] = df.groupby("model").size()
    ov.to_csv(OUT / f"{name}_overall.csv")
    print(ov.round(4).to_string())
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
