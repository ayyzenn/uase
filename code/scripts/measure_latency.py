#!/usr/bin/env python3
"""
Latency / throughput / fit-time measurement for protocol-v2 detectors and UASE-v2.

Streaming simulation (one new observation at a time, single process, 1 thread):
  * feature update  : the CURRENT implementation recomputes features from history
                      (univariate: full history, because expanding seasonal statistics are used;
                       Kubernetes: trailing 120 rows, which reproduces the batch features exactly
                       because all rolling windows are <= 24 -- verified at run time);
  * scaling         : StandardScaler.transform of one row;
  * channel scoring : each fitted detector scores ONE row (or one flattened window);
  * UASE-v2         : end-to-end = feature update + scaling + member channels
                      + ECDF mapping + mean + threshold comparison.
Spectral Residual is measured causally on the full history up to t (the evaluated variant
was computed on the whole series; see paper limitations).
Throughput = batch scoring of the whole test segment (features precomputed) in points/s.
Fit time   = (i) single fit of each channel on the training prefix, and (ii) the full
             UASE-v2 training pipeline as run in the experiments (uase_v2.channel_scores:
             all channels incl. 4-fold OOF refits).
Outputs: code/data/v2/latency/{hardware.json, latency_points.csv, latency_summary.csv,
         throughput.csv, fit_times.csv}
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uase_v2 as u  # noqa: E402
import evaluate_yahoo_s5 as ey  # noqa: E402
from anomaly_models import RollingZScoreDetector, WindowedAutoencoder, spectral_residual_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "v2" / "latency"
N_POINTS = 200
UASE_MEMBERS = {  # from data/v2/meta/*_selection.json (dev-selected, see paper)
    "k8s": ["Windowed RF", "Autoencoder"],
    "yahoo": ["Residual Z"],
    "cloud": ["Spectral Residual", "ECOD", "Isolation Forest", "HBOS"],
}
ns = time.perf_counter_ns


def hardware():
    cpu = subprocess.run("grep -m1 'model name' /proc/cpuinfo", shell=True, capture_output=True, text=True).stdout
    lscpu = subprocess.run("lscpu", shell=True, capture_output=True, text=True).stdout
    get = lambda k: next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith(k)), None)
    mem = int(open("/proc/meminfo").read().split()[1]) / 1024 / 1024
    return {"cpu_model": cpu.split(":", 1)[1].strip() if cpu else None, "cpu_family": get("CPU family"),
            "cpu_model_number": get("Model:"), "hypervisor": get("Hypervisor vendor"),
            "logical_cpus": os.cpu_count(), "nominal_mhz": subprocess.run("grep -m1 'cpu MHz' /proc/cpuinfo", shell=True,
            capture_output=True, text=True).stdout.split(":")[1].strip(), "ram_gib": round(mem, 1),
            "os": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "threads": "OMP_NUM_THREADS=1, n_jobs=1, single process"}


class Channels:
    """Fit every channel once on the training prefix (as in the test-time model of protocol v2)."""

    def __init__(self, X_raw, y, split, cfg, univariate, raw_values=None):
        c = u.CFG[cfg]
        self.c, self.univariate, self.split = c, univariate, split
        t0 = ns()
        self.scaler = StandardScaler().fit(X_raw[:split])
        Xs = np.nan_to_num(self.scaler.transform(X_raw), nan=0.0, posinf=0.0, neginf=0.0)
        self.Xs = Xs
        ytr = y[:split]
        normal = ytr == 0
        contam = float(np.clip(max(ytr.mean(), 1.0 / max(split, 1)), 0.01, 0.2))
        self.models, self.fit_ms = {}, {}
        for name, (fac, kind) in u._unsup_factories(c, contam, int(normal.sum())).items():
            t = ns()
            try:
                m = fac(); m.fit(Xs[:split][normal]); self.models[name] = (m, kind)
            except Exception:
                continue
            self.fit_ms[name] = (ns() - t) / 1e6
        t = ns()
        self.rz = RollingZScoreDetector(window=c["rz_w"], contamination=contam).fit(Xs[:split], ytr)
        self.fit_ms["Rolling Z-Score"] = (ns() - t) / 1e6
        self.win = int(max(5, min(c["win"], max(5, split // 3))))
        W = u.windowize(Xs, self.win)
        t = ns()
        self.wae = WindowedAutoencoder(window=self.win, hidden=c["wae_hidden"], contamination=contam, random_state=42)
        self.wae.model.max_iter = c["wae_iter"]; self.wae.fit(W[:split][normal])
        self.fit_ms["Windowed AE"] = (ns() - t) / 1e6
        t = ns()
        self.rf = RandomForestClassifier(n_estimators=c["rf_n"], max_depth=c["rf_depth"], min_samples_leaf=c["rf_leaf"],
                                         class_weight="balanced_subsample", max_features="sqrt", random_state=42, n_jobs=1)
        self.rf.fit(W[:split], ytr)
        self.fit_ms["Windowed RF"] = (ns() - t) / 1e6
        if univariate:
            tr = np.arange(len(Xs)) < split
            parts = [np.abs(Xs[:, j]) for j in u.RES_COLS if Xs.shape[1] > j]
            self.res_mu = np.array([p[tr].mean() for p in parts]); self.res_sd = np.array([p[tr].std() + 1e-8 for p in parts])
        self.fit_ms["Scaler"] = 0.0
        # ECDF references (OOF scores are what the paper uses; refit-score proxy is enough for timing)
        self.ref = {}

    def residual_row(self, xs_row):
        a = np.abs(xs_row[[j for j in u.RES_COLS if len(xs_row) > j]])
        return float(np.mean((a - self.res_mu) / self.res_sd))

    def score_row(self, name, xs_row, xs_window_flat, raw_hist=None, xs_tail=None):
        if name in self.models:
            m, kind = self.models[name]
            return float(u._score(m, kind, xs_row[None, :])[0])
        if name == "Windowed RF":
            return float(self.rf.predict_proba(xs_window_flat[None, :])[0, list(self.rf.classes_).index(1)])
        if name == "Windowed AE":
            return float(self.wae.score_samples(xs_window_flat[None, :])[0])
        if name == "Rolling Z-Score":
            return float(self.rz.score_samples(xs_tail)[-1])
        if name == "Spectral Residual":
            return float(spectral_residual_scores(raw_hist)[-1])
        if name == "Residual Z":
            return self.residual_row(xs_row)
        raise KeyError(name)


def k8s_features_tail(df, t, tail=120):
    from train_compare_models import enrich_k8s_features
    lo = max(0, t - tail + 1)
    X, _ = enrich_k8s_features(df.iloc[lo:t + 1].reset_index(drop=True))
    return X[-1]


def measure_series(dataset, label, X_raw, y, split, raw_values=None, df=None):
    univariate = dataset != "k8s"
    cfg = "univariate" if univariate else "k8s"
    rows, thr_rows = [], []
    t = ns(); ch = Channels(X_raw, y, split, cfg, univariate); t_single_total = (ns() - t) / 1e6
    # full UASE-v2 / pool training pipeline as used in the experiments
    t = ns(); u.channel_scores(X_raw, y, split, u.CFG[cfg], univariate); t_pool = (ns() - t) / 1e6
    fit = [{"dataset": dataset, "series": label, "component": k, "fit_ms": v} for k, v in ch.fit_ms.items() if k != "Scaler"]
    fit.append({"dataset": dataset, "series": label, "component": "ALL channels, single fit each", "fit_ms": t_single_total})
    fit.append({"dataset": dataset, "series": label, "component": "UASE-v2 training pipeline (all channels + 4-fold OOF)", "fit_ms": t_pool})
    names = sorted(set(list(ch.models) + ["Rolling Z-Score", "Windowed AE", "Windowed RF"] +
                       (["Spectral Residual", "Residual Z"] if univariate else [])))
    test_idx = np.arange(split, len(y))[:N_POINTS]
    Xs, win = ch.Xs, ch.win
    pad = np.vstack([np.repeat(Xs[:1], win - 1, axis=0), Xs])
    # verify K8s tail features are exact
    if not univariate:
        assert np.allclose(k8s_features_tail(df, int(test_idx[0])), X_raw[test_idx[0]], equal_nan=True)
    # warm-up
    for nm in names:
        ch.score_row(nm, Xs[split], pad[split:split + win].ravel(), raw_values[:split + 1] if univariate else None,
                     Xs[split - ch.c["rz_w"] + 1:split + 1])
    members = UASE_MEMBERS[dataset]
    # ECDF references: sorted training-prefix scores of each member (precomputed once at fit time)
    Wall = u.windowize(Xs, win)
    refs = {}
    for m in members:
        if m in ch.models:
            mm, kind = ch.models[m]; r = u._score(mm, kind, Xs[:split])
        elif m == "Windowed RF":
            r = ch.rf.predict_proba(Wall[:split])[:, list(ch.rf.classes_).index(1)]
        elif m == "Windowed AE":
            r = ch.wae.score_samples(Wall[:split])
        elif m == "Spectral Residual":
            r = spectral_residual_scores(raw_values[:split])
        elif m == "Residual Z":
            r = np.array([ch.residual_row(x) for x in Xs[:split]])
        refs[m] = np.sort(r)
    for t_ in test_idx:
        rec = {"dataset": dataset, "series": label, "t": int(t_)}
        a = ns()
        feat = ey.featurize_univariate(raw_values[:t_ + 1])[-1] if univariate else k8s_features_tail(df, int(t_))
        rec["feature_update"] = (ns() - a) / 1e6
        a = ns(); xs = np.nan_to_num(ch.scaler.transform(feat[None, :])[0]); rec["scaling"] = (ns() - a) / 1e6
        wflat = pad[t_:t_ + win].ravel()
        tail = Xs[max(0, t_ - ch.c["rz_w"] + 1):t_ + 1]
        hist = raw_values[:t_ + 1] if univariate else None
        for nm in names:
            a = ns(); ch.score_row(nm, xs, wflat, hist, tail); rec[nm] = (ns() - a) / 1e6
        # UASE-v2 end to end (fresh feature computation included)
        a = ns()
        feat = ey.featurize_univariate(raw_values[:t_ + 1])[-1] if univariate else k8s_features_tail(df, int(t_))
        xs = np.nan_to_num(ch.scaler.transform(feat[None, :])[0])
        zs = [ch.score_row(m, xs, wflat, hist, tail) for m in members]
        z = np.mean([np.searchsorted(refs[m], v, side="right") / len(refs[m]) for m, v in zip(members, zs)])
        _ = z >= 0.5
        rec["UASE-v2 (end-to-end)"] = (ns() - a) / 1e6
        rows.append(rec)
    # throughput: batch scoring over the whole test segment
    Xte = Xs[split:]; Wte = u.windowize(Xs, win)[split:]
    for nm in names:
        a = ns()
        if nm in ch.models:
            m, kind = ch.models[nm]; u._score(m, kind, Xte)
        elif nm == "Windowed RF":
            ch.rf.predict_proba(Wte)
        elif nm == "Windowed AE":
            ch.wae.score_samples(Wte)
        elif nm == "Rolling Z-Score":
            ch.rz.score_samples(Xte)
        elif nm == "Spectral Residual":
            spectral_residual_scores(raw_values)
        elif nm == "Residual Z":
            [ch.residual_row(r) for r in Xte]
        dt = (ns() - a) / 1e9
        thr_rows.append({"dataset": dataset, "series": label, "component": nm, "n_points": len(Xte),
                         "points_per_s": len(Xte) / max(dt, 1e-12)})
    return rows, thr_rows, fit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hw = hardware(); (OUT / "hardware.json").write_text(json.dumps(hw, indent=2)); print(hw, flush=True)
    allrows, allthr, allfit = [], [], []
    # Kubernetes (train-only z-score, variant A)
    from train_compare_models import enrich_k8s_features, load_data
    df = load_data(); X, _ = enrich_k8s_features(df); y = df["label"].values.astype(int)
    r, tr, f = measure_series("k8s", "k8s_metrics.csv", X, y, int(0.7 * len(df)), df=df)
    allrows += r; allthr += tr; allfit += f; print("k8s done", flush=True)
    # Yahoo: first 2 evaluable series per collection
    for bench in ("A1Benchmark", "A2Benchmark", "A3Benchmark", "A4Benchmark"):
        k = 0
        for p in ey.list_series(bench):
            v, yy = ey.load_series(p); split = ey.choose_split(yy)
            if len(v) < 80 or split is None or len(np.unique(yy[split:])) < 2 or (yy[:split] == 0).sum() < 20:
                continue
            Xu = np.nan_to_num(ey.featurize_univariate(v))
            r, tr, f = measure_series("yahoo", f"{bench}/{p.name}", Xu, yy.astype(int), split, raw_values=v)
            allrows += r; allthr += tr; allfit += f; k += 1
            if k == 2:
                break
    print("yahoo done", flush=True)
    import evaluate_cloud_monitoring as ecm
    # first evaluable series of each Cloud domain (deterministic: sorted order)
    done_dom = set()
    for dom, p in ecm.discover_all_series():
        if dom in done_dom:
            continue
        v, yy = ecm.load_series(p); split = ey.choose_split(yy)
        if len(v) < 80 or split is None or len(np.unique(yy[split:])) < 2 or (yy[:split] == 0).sum() < 20:
            continue
        Xu = np.nan_to_num(ey.featurize_univariate(v))
        r, tr, f = measure_series("cloud", f"{dom}/{p.name}", Xu, yy.astype(int), split, raw_values=v)
        allrows += r; allthr += tr; allfit += f; done_dom.add(dom)
    print("cloud done", flush=True)
    pts = pd.DataFrame(allrows); pts.to_csv(OUT / "latency_points.csv", index=False)
    long = pts.melt(id_vars=["dataset", "series", "t"], var_name="component", value_name="ms")
    summ = long.groupby(["dataset", "component"])["ms"].agg(
        n="count", mean="mean", median="median", p95=lambda s: np.percentile(s, 95), p99=lambda s: np.percentile(s, 99),
        max="max").reset_index()
    summ.to_csv(OUT / "latency_summary.csv", index=False)
    thr = pd.DataFrame(allthr); thr.to_csv(OUT / "throughput.csv", index=False)
    fit = pd.DataFrame(allfit); fit.to_csv(OUT / "fit_times.csv", index=False)
    print(summ.round(3).to_string())
    print(thr.groupby(["dataset", "component"]).points_per_s.median().round(0).to_string())
    print(fit.groupby(["dataset", "component"]).fit_ms.agg(["median", "max"]).round(1).to_string())
    # scaling of the univariate feature update with history length (synthetic random walk, 20 repeats)
    rng = np.random.default_rng(0); sc = []
    for L in (500, 1000, 2000, 5000, 10000, 20000):
        x = np.cumsum(rng.normal(size=L)) + 100
        ey.featurize_univariate(x)
        ts = []
        for _ in range(20):
            a = ns(); ey.featurize_univariate(x); ts.append((ns() - a) / 1e6)
        sc.append({"history_points": L, "median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))})
    pd.DataFrame(sc).to_csv(OUT / "feature_update_scaling.csv", index=False); print(pd.DataFrame(sc))
    # series lengths (for interpreting full-history feature recomputation)
    lens = pts.groupby(["dataset", "series"]).t.max().reset_index(); lens.to_csv(OUT / "series_max_t.csv", index=False)


if __name__ == "__main__":
    main()
