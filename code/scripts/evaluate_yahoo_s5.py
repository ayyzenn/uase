#!/usr/bin/env python3
"""
Evaluate all anomaly detectors on Yahoo Webscope S5 benchmarks A1–A4.

Windowed Random Forest is the proposed supervised methodology; unsupervised,
time-series, hybrid, and ensemble models are baselines for external validation.
UASE (Unified Adaptive Stacking Ensemble) is the proposed cross-dataset
collective model. On Yahoo it keeps LOF when that channel's training
point-adjust F1 is non-trivial, and replaces it with Isolation Forest
only when IF leads by at least 0.15 on the same training score.
Test labels are not used.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anomaly_models import (  # noqa: E402
    AutoencoderDetector,
    ECODDetector,
    HBOSDetector,
    LODADetector,
    PCAHybridDetector,
    RollingZScoreDetector,
    SpectralResidualDetector,
    WindowedAutoencoder,
    evaluate,
    make_windows,
    spectral_residual_scores,
)
from download_yahoo_s5 import DEST as YAHOO_ROOT  # noqa: E402
from download_yahoo_s5 import download as download_yahoo  # noqa: E402

FIG_DIR = ROOT / "figures"
PAPER_IMG = Path(__file__).resolve().parents[2] / "paper" / "files" / "images"
RESULTS_DIR = ROOT / "data"
WINDOW = 24  # Yahoo S5 is hourly; a one-day context window
BEST_PARAMS_PATH = RESULTS_DIR / "yahoo_best_params.json"
BEST_PARAMS: dict = {}
UASE_MODES: list[str] = []
WRITE_PAPER_FIGS = True
if BEST_PARAMS_PATH.exists():
    try:
        BEST_PARAMS = json.loads(BEST_PARAMS_PATH.read_text())
    except Exception:
        BEST_PARAMS = {}


def _p(model: str, key: str, default):
    """Lookup tuned hyperparameter for a model, else default."""
    cfg = BEST_PARAMS.get(model, {})
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    # also allow nested under "params"
    if isinstance(cfg, dict) and isinstance(cfg.get("params"), dict) and key in cfg["params"]:
        return cfg["params"][key]
    return default


def _resolve_contam(raw, y_train: np.ndarray, default: float) -> float:
    """Map tuned contamination ('data'/'auto'/float) to a numeric rate."""
    if raw is None:
        return float(default)
    if isinstance(raw, str) and raw.lower() in {"data", "auto"}:
        return float(np.clip(max(y_train.mean(), 1.0 / max(len(y_train), 1)), 0.01, 0.2))
    try:
        return float(np.clip(float(raw), 0.01, 0.5))
    except Exception:
        return float(default)


def predict_by_train_f1(
    scores_train: np.ndarray,
    y_train: np.ndarray,
    scores_test: np.ndarray,
    strategy: str = "calibrate",
) -> tuple[np.ndarray, float]:
    """
    Choose a decision threshold on TRAIN scores that maximizes F1,
    then apply it to TEST scores. Primary lever for Yahoo F1 gains.
    """
    if strategy == "fixed_0.5":
        thr = 0.5
    else:
        thr = calibrate_threshold(y_train, scores_train)
    return (scores_test >= thr).astype(int), thr


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_IMG.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    folders = (FIG_DIR, PAPER_IMG) if WRITE_PAPER_FIGS else (FIG_DIR,)
    for folder in folders:
        plt.savefig(folder / name, dpi=200, bbox_inches="tight")
    plt.close()


def load_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # value
    value_col = cols.get("value")
    if value_col is None:
        raise ValueError(f"No value column in {path}")
    values = df[value_col].astype(float).values

    if "is_anomaly" in cols:
        y = df[cols["is_anomaly"]].astype(int).values
    elif "anomaly" in cols:
        y = df[cols["anomaly"]].astype(int).values
        if "changepoint" in cols:
            y = np.maximum(y, df[cols["changepoint"]].astype(int).values)
    else:
        raise ValueError(f"No anomaly label in {path}")
    return values, y


def _seasonal_mean_residual(s: pd.Series, period: int) -> pd.Series:
    """Subtract expanding seasonal mean at each phase (causal Yahoo A3/A4 pattern)."""
    n = len(s)
    out = np.zeros(n, dtype=float)
    sums = np.zeros(period, dtype=float)
    counts = np.zeros(period, dtype=float)
    vals = s.to_numpy(dtype=float)
    for i, v in enumerate(vals):
        phase = i % period
        if counts[phase] > 0:
            out[i] = v - sums[phase] / counts[phase]
        else:
            out[i] = 0.0
        sums[phase] += v
        counts[phase] += 1.0
    return pd.Series(out, index=s.index)


def _robust_z(s: pd.Series, window: int) -> pd.Series:
    med = s.rolling(window, min_periods=max(3, window // 4)).median()
    mad = (s - med).abs().rolling(window, min_periods=max(3, window // 4)).median()
    return ((s - med) / (1.4826 * mad + 1e-8)).fillna(0.0)


def featurize_univariate(values: np.ndarray) -> np.ndarray:
    """
    Contextual + seasonal-residual features aligned with Yahoo S5 structure
    (12h / daily / weekly seasonality, trend, short outliers, changepoints).
    """
    s = pd.Series(values.astype(float))
    n = len(s)
    hour = np.arange(n) % 24
    dow = np.arange(n) % 168

    resid12 = _seasonal_mean_residual(s, 12)
    resid24 = _seasonal_mean_residual(s, 24)
    resid168 = _seasonal_mean_residual(s, 168) if n >= 200 else resid24

    # Local trend residual (slow drift / changepoint cue)
    trend = s.rolling(48, min_periods=5).mean().bfill().ffill()
    resid_trend = s - trend
    ewma12 = s.ewm(span=12, adjust=False).mean()
    ewma24 = s.ewm(span=24, adjust=False).mean()
    resid_ewma12 = s - ewma12
    resid_ewma24 = s - ewma24

    # Causal rarity of |resid24| vs expanding MAD (fast alternative to expanding rank)
    abs_r24 = resid24.abs()
    exp_med = abs_r24.expanding(min_periods=5).median()
    exp_mad = (abs_r24 - exp_med).abs().expanding(min_periods=5).median()
    rarity_r24 = (abs_r24 / (1.4826 * exp_mad + 1e-8)).fillna(0.0)

    feats = pd.DataFrame(
        {
            "value": s,
            "diff1": s.diff().fillna(0.0),
            "diff2": s.diff(2).fillna(0.0),
            "diff3": s.diff(3).fillna(0.0),
            "abs_diff1": s.diff().abs().fillna(0.0),
            "roll_mean_12": s.rolling(12, min_periods=1).mean(),
            "roll_std_12": s.rolling(12, min_periods=1).std().fillna(0.0),
            "roll_mean_24": s.rolling(24, min_periods=1).mean(),
            "roll_std_24": s.rolling(24, min_periods=1).std().fillna(0.0),
            "roll_mean_48": s.rolling(48, min_periods=1).mean(),
            "roll_std_48": s.rolling(48, min_periods=1).std().fillna(0.0),
            "lag1": s.shift(1).bfill(),
            "lag2": s.shift(2).bfill(),
            "lag24": s.shift(24).bfill() if n > 24 else s,
            "hour": hour,
            "dow": dow,
            "resid12": resid12,
            "resid24": resid24,
            "resid168": resid168,
            "resid_trend": resid_trend,
            "z_robust_24": _robust_z(s, 24),
            "z_robust_48": _robust_z(s, 48),
            "z_resid24": _robust_z(resid24, 24),
            "z_resid168": _robust_z(resid168, 48),
            "resid_ewma12": resid_ewma12,
            "resid_ewma24": resid_ewma24,
            "z_ewma12": _robust_z(resid_ewma12, 24),
            "z_ewma24": _robust_z(resid_ewma24, 48),
            "roll_max_abs_diff_6": s.diff().abs().rolling(6, min_periods=1).max().fillna(0.0),
            "roll_max_abs_resid24_6": resid24.abs().rolling(6, min_periods=1).max().fillna(0.0),
            "rarity_abs_resid24": rarity_r24,
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 168.0),
            "dow_cos": np.cos(2 * np.pi * dow / 168.0),
        }
    )
    return feats.values.astype(float)


def residual_anomaly_score(X: np.ndarray) -> np.ndarray:
    """
    Unsupervised residual magnitude used for RF score fusion.
    Expects featurize_univariate layout; uses robust z / residual columns.
    """
    # Column indices from featurize_univariate (keep in sync)
    cols = {
        "abs_diff1": 4,
        "resid24": 17,
        "z_robust_24": 20,
        "z_resid24": 22,
        "z_resid168": 23,
        "z_ewma12": 26,
        "z_ewma24": 27,
        "roll_max_abs_resid24_6": 29,
        "rarity_abs_resid24": 30,
    }
    keys = (
        "z_robust_24",
        "z_resid24",
        "z_resid168",
        "abs_diff1",
        "z_ewma12",
        "z_ewma24",
        "roll_max_abs_resid24_6",
        "rarity_abs_resid24",
    )
    parts = []
    for key in keys:
        j = cols[key]
        if X.shape[1] > j:
            parts.append(np.abs(X[:, j]))
    if not parts:
        return np.linalg.norm(X, axis=1)
    stack = np.vstack(parts)
    scores = np.zeros(X.shape[0])
    for row in stack:
        mu, sd = float(np.mean(row)), float(np.std(row) + 1e-8)
        scores += (row - mu) / sd
    return scores / max(len(parts), 1)


def _rank01(a: np.ndarray) -> np.ndarray:
    order = a.argsort().argsort().astype(float)
    return order / max(len(a) - 1, 1)


def _ecdf_map(train_vals: np.ndarray, test_vals: np.ndarray) -> np.ndarray:
    ref = np.sort(train_vals)
    return np.searchsorted(ref, test_vals, side="right") / max(len(ref), 1)


def _pick_fusion_alpha(
    model_tr: np.ndarray,
    res_tr: np.ndarray,
    y_train: np.ndarray,
    alphas: list[float],
    default_alpha: float = 0.65,
    rank_model: bool = True,
    strategy: str = "calibrate",
) -> float:
    """Inner chronological holdout on TRAIN to pick residual fusion weight."""
    from sklearn.metrics import f1_score as _f1

    model_part = _rank01(model_tr) if rank_model else model_tr
    res_n = _rank01(res_tr)
    n_tr = len(y_train)
    inner = max(int(0.7 * n_tr), 20)
    anom_idx = np.flatnonzero(y_train == 1)
    if len(anom_idx) >= 2:
        inner = int(np.clip(inner, anom_idx[0] + 1, anom_idx[-1]))
    if inner < 15 or n_tr - inner < 8:
        inner = max(n_tr // 2, 15)

    best_a, best_f = default_alpha, -1.0
    y_in_tr, y_in_va = y_train[:inner], y_train[inner:]
    if len(np.unique(y_in_va)) > 1 and len(np.unique(y_in_tr)) > 1:
        for a in alphas:
            tr = a * model_part[:inner] + (1.0 - a) * res_n[:inner]
            va = a * model_part[inner:] + (1.0 - a) * res_n[inner:]
            pred_va, _ = predict_by_train_f1(tr, y_in_tr, va, strategy=strategy)
            f = float(_f1(y_in_va, point_adjust(y_in_va, pred_va), zero_division=0))
            if f > best_f:
                best_f, best_a = f, a
    return float(best_a)



def choose_split(y: np.ndarray) -> int | None:
    """
    Chronological split that keeps at least one anomaly in train and one in test
    whenever the series has >=2 anomalies (needed for supervised Windowed RF).
    """
    n = len(y)
    default = int(0.7 * n)
    anom = np.flatnonzero(y == 1)
    if len(anom) == 0:
        return None
    if len(anom) == 1:
        # Keep unsupervised evaluation only if the single anomaly lands in test.
        split = min(max(default, 40), n - 20)
        return split if anom[0] >= split else None

    # Ensure first anomaly is in train and last anomaly remains in test.
    min_split = int(anom[0]) + 1
    max_split = int(anom[-1])
    split = int(np.clip(default, min_split, max_split))
    if split < 40 or n - split < 15:
        return None
    if y[:split].sum() == 0 or y[split:].sum() == 0:
        return None
    return split


def calibrate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    point_adjust_obj: bool = True,
) -> float:
    """
    Threshold on TRAIN scores that maximizes F1 (optionally after point-adjust).
    Point-adjust objective matches the Yahoo/KPI reporting protocol.
    """
    from sklearn.metrics import f1_score as _f1

    if len(np.unique(y_true)) < 2:
        rate = 0.05
        return float(np.quantile(scores, 1.0 - rate))
    best_t, best_f = float(np.median(scores)), -1.0
    # Quantile grid only (fast); covers high-recall through high-precision
    qs = np.unique(
        np.concatenate(
            [
                np.quantile(scores, np.linspace(0.20, 0.999, 55)),
                np.quantile(scores, np.linspace(0.01, 0.20, 20)),
                np.unique(scores),
            ]
        )
    )
    if len(qs) > 80:
        # Cap unique-score explosion while keeping quantile coverage
        qs = np.unique(
            np.concatenate(
                [
                    np.quantile(scores, np.linspace(0.20, 0.999, 55)),
                    np.quantile(scores, np.linspace(0.01, 0.20, 20)),
                ]
            )
        )
    for t in qs:
        pred = (scores >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        if point_adjust_obj:
            pred_eval = point_adjust(y_true, pred)
        else:
            pred_eval = pred
        f = _f1(y_true, pred_eval, zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    if best_f <= 0:
        rate = float(np.clip(max(y_true.mean(), 0.02), 0.02, 0.15))
        return float(np.quantile(scores, 1.0 - rate))
    return float(best_t)


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Standard point-adjust protocol used on Yahoo/KPI benchmarks:
    if any point inside a true anomaly segment is detected, the whole segment
    is counted as detected.
    """
    y_adj = y_pred.copy().astype(int)
    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 1:
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            if y_pred[i:j].any():
                y_adj[i:j] = 1
            i = j
        else:
            i += 1
    return y_adj


def run_models_on_series(X: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """Fit all models on one series; return per-model metrics on chronological test split."""
    if len(X) < 80 or y.sum() == 0:
        return {}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    split = choose_split(y)
    if split is None:
        return {}

    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = y[:split], y[split:]
    if len(np.unique(y_test)) < 2:
        return {}

    normal_mask = y_train == 0
    if normal_mask.sum() < 20:
        return {}
    X_train_normal = X_train[normal_mask]
    contam = float(np.clip(max(y_train.mean(), 1.0 / max(len(y_train), 1)), 0.01, 0.2))

    scores: dict[str, np.ndarray] = {}
    scores_train: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    yrefs: dict[str, np.ndarray] = {}

    def reg(name, sc, pr, yt=None, sc_tr=None):
        scores[name] = sc
        preds[name] = pr
        yrefs[name] = y_test if yt is None else yt
        if sc_tr is not None:
            scores_train[name] = sc_tr

    # Classical unsupervised — hyperparameters from grid search when available;
    # predictions use train-F1-optimal thresholds to maximize Yahoo F1.
    iso_n = int(_p("Isolation Forest", "n_estimators", 100))
    iso_cont = _resolve_contam(_p("Isolation Forest", "contamination", contam), y_train, contam)
    iso_ms = _p("Isolation Forest", "max_samples", "auto")
    iso = IsolationForest(
        n_estimators=iso_n,
        contamination=min(0.5, max(0.01, iso_cont)),
        max_samples=iso_ms,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train_normal)
    iso_tr = -iso.decision_function(X_train)
    iso_te = -iso.decision_function(X_test)
    iso_pred, _ = predict_by_train_f1(iso_tr, y_train, iso_te)
    reg("Isolation Forest", iso_te, iso_pred, sc_tr=iso_tr)

    lof_k = int(_p("LOF", "n_neighbors", min(20, max(5, len(X_train_normal) // 10))))
    lof_k = min(lof_k, max(5, len(X_train_normal) - 1))
    lof_cont = _resolve_contam(_p("LOF", "contamination", contam), y_train, contam)
    lof = LocalOutlierFactor(
        n_neighbors=lof_k,
        contamination=min(0.5, max(0.01, lof_cont)),
        novelty=True,
    )
    lof.fit(X_train_normal)
    lof_tr = -lof.decision_function(X_train)
    lof_te = -lof.decision_function(X_test)
    if bool(_p("LOF", "use_fusion", True)):
        res_tr = residual_anomaly_score(X_train)
        res_te = residual_anomaly_score(X_test)
        lof_tr_n = _rank01(lof_tr)
        lof_te_n = _ecdf_map(lof_tr, lof_te)
        res_tr_n = _rank01(res_tr)
        res_te_n = _ecdf_map(res_tr, res_te)
        alpha = _pick_fusion_alpha(
            lof_tr,
            res_tr,
            y_train,
            [float(a) for a in _p(
                "LOF",
                "fuse_alphas",
                [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0],
            )],
            default_alpha=float(_p("LOF", "fuse_alpha", 0.65)),
        )
        lof_tr = alpha * lof_tr_n + (1.0 - alpha) * res_tr_n
        lof_te = alpha * lof_te_n + (1.0 - alpha) * res_te_n
    lof_pred, _ = predict_by_train_f1(lof_tr, y_train, lof_te)
    reg("LOF", lof_te, lof_pred, sc_tr=lof_tr)

    oc_nu = float(_p("One-Class SVM", "nu", min(0.5, max(0.01, contam))))
    oc_gamma = _p("One-Class SVM", "gamma", "scale")
    ocsvm = OneClassSVM(kernel="rbf", gamma=oc_gamma, nu=min(0.5, max(0.01, oc_nu)))
    ocsvm.fit(X_train_normal)
    oc_tr = -ocsvm.decision_function(X_train)
    oc_te = -ocsvm.decision_function(X_test)
    oc_pred, _ = predict_by_train_f1(oc_tr, y_train, oc_te)
    reg("One-Class SVM", oc_te, oc_pred, sc_tr=oc_tr)

    try:
        ee_cont = _resolve_contam(_p("Elliptic Envelope", "contamination", contam), y_train, contam)
        ee_sf = _p("Elliptic Envelope", "support_fraction", 0.8)
        ee = EllipticEnvelope(
            contamination=min(0.5, max(0.01, ee_cont)),
            random_state=42,
            support_fraction=ee_sf,
        )
        ee.fit(X_train_normal)
        ee_tr = -ee.decision_function(X_train)
        ee_te = -ee.decision_function(X_test)
        ee_pred, _ = predict_by_train_f1(ee_tr, y_train, ee_te)
        reg("Elliptic Envelope", ee_te, ee_pred, sc_tr=ee_tr)
    except Exception:
        pass

    ae_hidden = tuple(_p("Autoencoder", "hidden", (16, 8, 16)))
    ae_cont = _resolve_contam(_p("Autoencoder", "contamination", contam), y_train, contam)
    ae_iter = int(_p("Autoencoder", "max_iter", 40))
    ae = AutoencoderDetector(hidden=ae_hidden, contamination=ae_cont, random_state=42)
    ae.model.max_iter = ae_iter
    ae.fit(X_train_normal)
    ae_tr = ae.score_samples(X_train)
    ae_te = ae.score_samples(X_test)
    ae_pred, _ = predict_by_train_f1(ae_tr, y_train, ae_te)
    reg("Autoencoder", ae_te, ae_pred, sc_tr=ae_tr)

    ec_cont = _resolve_contam(_p("ECOD", "contamination", contam), y_train, contam)
    ecod = ECODDetector(contamination=ec_cont).fit(X_train_normal)
    ec_tr = ecod.score_samples(X_train)
    ec_te = ecod.score_samples(X_test)
    ec_pred, _ = predict_by_train_f1(ec_tr, y_train, ec_te)
    reg("ECOD", ec_te, ec_pred, sc_tr=ec_tr)

    loda_cuts = int(_p("LODA", "n_random_cuts", 50))
    loda_bins = int(_p("LODA", "n_bins", 20))
    loda_cont = _resolve_contam(_p("LODA", "contamination", contam), y_train, contam)
    loda = LODADetector(
        n_random_cuts=loda_cuts, n_bins=loda_bins, contamination=loda_cont, random_state=42
    ).fit(X_train_normal)
    lo_tr = loda.score_samples(X_train)
    lo_te = loda.score_samples(X_test)
    lo_pred, _ = predict_by_train_f1(lo_tr, y_train, lo_te)
    reg("LODA", lo_te, lo_pred, sc_tr=lo_tr)

    hbos_bins = int(_p("HBOS", "n_bins", 20))
    hbos_cont = _resolve_contam(_p("HBOS", "contamination", contam), y_train, contam)
    hbos = HBOSDetector(n_bins=hbos_bins, contamination=hbos_cont).fit(X_train_normal)
    hb_tr = hbos.score_samples(X_train)
    hb_te = hbos.score_samples(X_test)
    hb_pred, _ = predict_by_train_f1(hb_tr, y_train, hb_te)
    reg("HBOS", hb_te, hb_pred, sc_tr=hb_tr)

    rz_w = int(_p("Rolling Z-Score", "window", 24))
    rz_mode = _p("Rolling Z-Score", "contam_mode", _p("Rolling Z-Score", "contamination", contam))
    rz_cont = _resolve_contam(rz_mode, y_train, contam)
    rz = RollingZScoreDetector(window=rz_w, contamination=rz_cont).fit(X_train, y_train)
    rz_tr = rz.score_samples(X_train)
    rz_te = rz.score_samples(X_test)
    rz_pred, _ = predict_by_train_f1(rz_tr, y_train, rz_te)
    reg("Rolling Z-Score", rz_te, rz_pred, sc_tr=rz_tr)

    # Windowed models
    win = int(_p("Windowed RF", "window", WINDOW))
    win = max(5, min(win, max(5, split // 3)))
    Xw, yw = make_windows(Xs, y, window=win)
    w_split = max(win, split - win + 1)
    if w_split > 20 and len(Xw) - w_split > 10:
        Xw_train, Xw_test = Xw[:w_split], Xw[w_split:]
        yw_train, yw_test = yw[:w_split], yw[w_split:]
        if len(np.unique(yw_test)) > 1 and (yw_train == 0).sum() > 10:
            wae_win = int(_p("Windowed AE", "window", win))
            wae_win = max(5, min(wae_win, max(5, split // 3)))
            if wae_win != win:
                Xw_ae, yw_ae = make_windows(Xs, y, window=wae_win)
                w_split_ae = max(wae_win, split - wae_win + 1)
                Xwae_tr, Xwae_te = Xw_ae[:w_split_ae], Xw_ae[w_split_ae:]
                ywae_tr, ywae_te = yw_ae[:w_split_ae], yw_ae[w_split_ae:]
            else:
                Xwae_tr, Xwae_te, ywae_tr, ywae_te = Xw_train, Xw_test, yw_train, yw_test
                wae_win = win

            if len(np.unique(ywae_te)) > 1 and (ywae_tr == 0).sum() > 10:
                wae_hidden = tuple(_p("Windowed AE", "hidden", (32, 12, 32)))
                wae_mode = _p("Windowed AE", "contam_mode", _p("Windowed AE", "contamination", contam))
                wae_cont = _resolve_contam(wae_mode, ywae_tr, contam)
                wae = WindowedAutoencoder(
                    window=wae_win, hidden=wae_hidden, contamination=wae_cont, random_state=42
                )
                wae.model.max_iter = int(_p("Windowed AE", "max_iter", 35))
                wae.fit(Xwae_tr[ywae_tr == 0])
                wae_tr_s = wae.score_samples(Xwae_tr)
                wae_te_s = wae.score_samples(Xwae_te)
                wae_pred, _ = predict_by_train_f1(wae_tr_s, ywae_tr, wae_te_s)
                reg("Windowed AE", wae_te_s, wae_pred, yt=ywae_te, sc_tr=wae_tr_s)

            # Proposed: windowed RF fused with seasonal-residual unsupervised score
            if yw_train.sum() > 0 and len(np.unique(yw_train)) > 1:
                md = _p("Windowed RF", "max_depth", None)
                if md is not None:
                    try:
                        md = int(md)
                    except Exception:
                        md = None
                rf = RandomForestClassifier(
                    n_estimators=int(_p("Windowed RF", "n_estimators", 200)),
                    max_depth=md,
                    random_state=42,
                    n_jobs=-1,
                    class_weight=_p("Windowed RF", "class_weight", "balanced_subsample"),
                    min_samples_leaf=int(_p("Windowed RF", "min_samples_leaf", 1)),
                    max_features=_p("Windowed RF", "max_features", "sqrt"),
                )
                rf.fit(Xw_train, yw_train)

                def rf_scores(Xw_):
                    prob = rf.predict_proba(Xw_)
                    if prob.shape[1] == 1:
                        return prob[:, 0] if rf.classes_[0] == 1 else 1.0 - prob[:, 0]
                    return prob[:, list(rf.classes_).index(1)]

                train_p = rf_scores(Xw_train)
                test_p = rf_scores(Xw_test)

                # Residual fusion on the underlying (non-window) feature rows
                # Window label at index i corresponds to raw row (i) after make_windows offset.
                res_full = residual_anomaly_score(Xs)
                # Align residual scores to window endpoints
                res_aligned = res_full[win - 1 :]
                res_tr = res_aligned[:w_split]
                res_te = res_aligned[w_split:]
                mtr = min(len(train_p), len(res_tr), len(yw_train))
                mte = min(len(test_p), len(res_te), len(yw_test))
                train_p, res_tr, yw_tr_f = train_p[:mtr], res_tr[:mtr], yw_train[:mtr]
                test_p, res_te, yw_te_f = test_p[:mte], res_te[:mte], yw_test[:mte]

                res_tr_n = _rank01(res_tr)
                res_te_n = _ecdf_map(res_tr, res_te)
                use_fuse = bool(_p("Windowed RF", "use_fusion", True))
                thr_strat = str(_p("Windowed RF", "thr_strategy", "calibrate"))

                if use_fuse:
                    best_a = _pick_fusion_alpha(
                        train_p,
                        res_tr,
                        yw_tr_f,
                        [float(a) for a in _p(
                            "Windowed RF",
                            "fuse_alphas",
                            [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0],
                        )],
                        default_alpha=float(_p("Windowed RF", "fuse_alpha", 0.35)),
                        rank_model=False,
                        strategy=thr_strat,
                    )
                    best_tr = best_a * train_p + (1.0 - best_a) * res_tr_n
                    best_te = best_a * test_p + (1.0 - best_a) * res_te_n
                    rf_pred, _ = predict_by_train_f1(
                        best_tr, yw_tr_f, best_te, strategy=thr_strat
                    )
                    reg("Windowed RF", best_te, rf_pred, yt=yw_te_f, sc_tr=best_tr)
                else:
                    rf_pred, _ = predict_by_train_f1(
                        train_p, yw_tr_f, test_p, strategy=thr_strat
                    )
                    reg("Windowed RF", test_p, rf_pred, yt=yw_te_f, sc_tr=train_p)

    pca_n = _p("PCA+ECOD", "n_components", 0.95)
    pca_cont = _resolve_contam(_p("PCA+ECOD", "contamination", contam), y_train, contam)
    pca_base = _p("PCA+ECOD", "base", "ecod")
    hybrid = PCAHybridDetector(
        n_components=pca_n, contamination=pca_cont, base=pca_base, random_state=42
    )
    hybrid.fit(X_train_normal)
    hy_tr = hybrid.score_samples(X_train)
    hy_te = hybrid.score_samples(X_test)
    hy_pred, _ = predict_by_train_f1(hy_tr, y_train, hy_te)
    reg("PCA+ECOD", hy_te, hy_pred, sc_tr=hy_tr)

    # Spectral Residual on raw univariate value (first feature column before scaling)
    try:
        raw_vals = X[:, 0]
        sr_all = spectral_residual_scores(raw_vals)
        sr_tr, sr_te = sr_all[:split], sr_all[split:]
        sr_pred, _ = predict_by_train_f1(sr_tr, y_train, sr_te)
        reg("Spectral Residual", sr_te, sr_pred, sc_tr=sr_tr)
    except Exception:
        pass

    # UASE (proposed): neighborhood-first selection on TRAIN point-adjust F1.
    # LOF is kept when its training F1_pa is non-trivial. Isolation Forest
    # replaces it only when IF leads LOF by at least 0.15 on that same
    # training score. Test labels are not used. The chosen channel's
    # already calibrated test predictions are copied (no second threshold).
    from sklearn.metrics import f1_score as _f1

    _UASE_CANDIDATES = (
        "Isolation Forest",
        "LOF",
        "ECOD",
        "LODA",
        "HBOS",
        "Spectral Residual",
        "PCA+ECOD",
        "Autoencoder",
        "Elliptic Envelope",
    )
    n_tr, n_te = len(y_train), len(y_test)
    f1_tr_map: dict[str, float] = {}
    for _name in _UASE_CANDIDATES:
        if _name not in scores_train or _name not in scores or _name not in preds:
            continue
        _tr_sc, _te_sc = scores_train[_name], scores[_name]
        if len(_tr_sc) != n_tr or len(_te_sc) != n_te or len(preds[_name]) != n_te:
            continue
        _pred_tr, _ = predict_by_train_f1(_tr_sc, y_train, _tr_sc)
        f1_tr_map[_name] = float(_f1(y_train, point_adjust(y_train, _pred_tr), zero_division=0))

    if len(f1_tr_map) >= 2 and y_train.sum() > 0:
        ordered = sorted(f1_tr_map, key=f1_tr_map.get, reverse=True)
        best = ordered[0]
        if "LOF" in f1_tr_map and f1_tr_map["LOF"] >= 0.05:
            best = "LOF"
            if (
                "Isolation Forest" in f1_tr_map
                and f1_tr_map["Isolation Forest"] >= f1_tr_map["LOF"] + 0.15
            ):
                best = "Isolation Forest"
        reg(
            "UASE (proposed)",
            scores[best],
            preds[best],
            sc_tr=scores_train[best],
        )
        UASE_MODES.append(best)

    out = {}
    for name in preds:
        yt, yp, sc = yrefs[name], preds[name], scores[name]
        m = min(len(yt), len(yp), len(sc))
        yt, yp, sc = yt[:m], yp[:m], sc[:m]
        met = evaluate(yt, yp, sc)
        y_pa = point_adjust(yt, yp)
        met_pa = evaluate(yt, y_pa, sc)
        met["f1_pa"] = met_pa["f1"]
        met["mcc_pa"] = met_pa["mcc"]
        met["precision_pa"] = met_pa["precision"]
        met["recall_pa"] = met_pa["recall"]
        out[name] = met
    return out


def list_series(benchmark: str) -> list[Path]:
    folder = YAHOO_ROOT / benchmark
    files = sorted(p for p in folder.glob("*.csv") if "_all" not in p.name)
    return files


def evaluate_benchmark(benchmark: str, max_series: int | None = None) -> pd.DataFrame:
    files = list_series(benchmark)
    if max_series is not None:
        files = files[:max_series]
    print(f"\n=== {benchmark}: {len(files)} series ===")
    rows = []
    for i, path in enumerate(files, 1):
        try:
            values, y = load_series(path)
            X = featurize_univariate(values)
            metrics = run_models_on_series(X, y)
        except Exception as exc:
            print(f"  skip {path.name}: {exc}")
            continue
        if not metrics:
            continue
        for model, met in metrics.items():
            rows.append({"benchmark": benchmark, "series": path.name, "model": model, **met})
        if i % 20 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)}")
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Macro-average metrics across series within each benchmark."""
    metric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "f1_pa",
        "mcc_pa",
        "precision_pa",
        "recall_pa",
    ]
    cols = [c for c in metric_cols if c in df.columns]
    g = df.groupby(["benchmark", "model"], as_index=False)[cols].mean(numeric_only=True)
    return g


def plot_yahoo_heatmap(agg: pd.DataFrame) -> None:
    # Prefer point-adjust MCC when available (Yahoo protocol)
    value_col = "mcc_pa" if "mcc_pa" in agg.columns else "mcc"
    pivot = agg.pivot(index="model", columns="benchmark", values=value_col)
    model_order = [
        "UASE (proposed)",
        "Windowed RF",
        "Spectral Residual",
        "One-Class SVM",
        "Windowed AE",
        "Rolling Z-Score",
        "Elliptic Envelope",
        "Autoencoder",
        "LODA",
        "PCA+ECOD",
        "HBOS",
        "LOF",
        "Isolation Forest",
        "ECOD",
    ]
    cols = [c for c in ["A1Benchmark", "A2Benchmark", "A3Benchmark", "A4Benchmark"] if c in pivot.columns]
    rows = [m for m in model_order if m in pivot.index] + [m for m in pivot.index if m not in model_order]
    pivot = pivot.loc[rows, cols]
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Yahoo S5: mean {value_col.upper()} by model and benchmark")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    savefig("fig_yahoo_s5_mcc_heatmap.png")


def plot_yahoo_bars(overall: pd.DataFrame) -> None:
    value_col = "mcc_pa" if "mcc_pa" in overall.columns else "mcc"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ordered = overall.sort_values(value_col, ascending=True)
    colors = ["#0F766E" if "Proposed" in m else "#94A3B8" if "plain" in m else "#2563EB" for m in ordered["model"]]
    ax.barh(ordered["model"], ordered[value_col], color=colors)
    ax.set_xlabel(f"Mean {value_col} across Yahoo S5 A1–A4")
    ax.set_title("External validation on Yahoo S5 (proposed method highlighted)")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    savefig("fig_yahoo_s5_mcc_bars.png")


def run(max_series: int | None = None, results_dir: Path | None = None, write_figures: bool = True) -> None:
    global BEST_PARAMS, RESULTS_DIR, FIG_DIR, WRITE_PAPER_FIGS, UASE_MODES
    UASE_MODES = []
    if results_dir is not None:
        RESULTS_DIR = Path(results_dir)
        FIG_DIR = RESULTS_DIR / "figures"
        WRITE_PAPER_FIGS = False
    if not write_figures:
        WRITE_PAPER_FIGS = False
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    if BEST_PARAMS_PATH.exists():
        BEST_PARAMS = json.loads(BEST_PARAMS_PATH.read_text())
        print(f"Loaded tuned params for: {list(BEST_PARAMS.keys())}")
    download_yahoo(force=False)

    all_rows = []
    for bench in ("A1Benchmark", "A2Benchmark", "A3Benchmark", "A4Benchmark"):
        part = evaluate_benchmark(bench, max_series=max_series)
        all_rows.append(part)
    detail = pd.concat(all_rows, ignore_index=True)
    detail.to_csv(RESULTS_DIR / "yahoo_s5_per_series.csv", index=False)

    agg = aggregate(detail)
    agg.to_csv(RESULTS_DIR / "yahoo_s5_by_benchmark.csv", index=False)

    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "f1_pa", "mcc_pa"] if c in detail.columns]
    overall = (
        detail.groupby("model", as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values("f1_pa" if "f1_pa" in metric_cols else "f1", ascending=False)
        .reset_index(drop=True)
    )
    overall.to_csv(RESULTS_DIR / "yahoo_s5_overall.csv", index=False)

    if write_figures:
        plot_yahoo_heatmap(agg)
        plot_yahoo_bars(overall)

    rank_col = "mcc_pa" if "mcc_pa" in agg.columns else "mcc"
    wide = agg.pivot(index="model", columns="benchmark", values=rank_col)
    for c in ("A1Benchmark", "A2Benchmark", "A3Benchmark", "A4Benchmark"):
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide[["A1Benchmark", "A2Benchmark", "A3Benchmark", "A4Benchmark"]]
    wide["mean_mcc_pa"] = wide.mean(axis=1)
    wide = wide.sort_values("mean_mcc_pa", ascending=False)
    wide.to_csv(RESULTS_DIR / "yahoo_s5_mcc_wide.csv")

    summary = {
        "best_model_by_mean_f1_pa": str(overall.iloc[0]["model"]),
        "best_mean_f1_pa": float(overall.iloc[0].get("f1_pa", overall.iloc[0]["f1"])),
        "best_model_by_mean_mcc_pa": str(wide.index[0]),
        "best_mean_mcc_pa": float(wide.iloc[0]["mean_mcc_pa"]),
        "n_series_evaluated": int(detail[["benchmark", "series"]].drop_duplicates().shape[0]),
        "uase_mode_counts": {
            mode: UASE_MODES.count(mode) for mode in sorted(set(UASE_MODES))
        },
        "overall": overall.to_dict(orient="records"),
        "by_benchmark_mcc_pa": wide.reset_index().to_dict(orient="records"),
    }
    (RESULTS_DIR / "yahoo_s5_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n===== Yahoo S5 overall (mean across series) =====")
    print(overall.to_string(index=False))
    print("\n===== Point-adjust MCC by benchmark =====")
    print(wide.round(3).to_string())
    print(
        "\nBest F1_pa:",
        summary["best_model_by_mean_f1_pa"],
        summary["best_mean_f1_pa"],
    )
    print(
        "Best MCC_pa:",
        summary["best_model_by_mean_mcc_pa"],
        summary["best_mean_mcc_pa"],
    )
    print("UASE modes:", summary["uase_mode_counts"])


if __name__ == "__main__":
    # python evaluate_yahoo_s5.py --max-series 3 --results-dir data/yahoo_uase_trial --no-figures
    max_series = None
    results_dir = None
    write_figures = "--no-figures" not in sys.argv
    if "--max-series" in sys.argv:
        max_series = int(sys.argv[sys.argv.index("--max-series") + 1])
    if "--results-dir" in sys.argv:
        results_dir = Path(sys.argv[sys.argv.index("--results-dir") + 1])
        if not results_dir.is_absolute():
            results_dir = ROOT / results_dir
    run(max_series=max_series, results_dir=results_dir, write_figures=write_figures)
