#!/usr/bin/env python3
"""
Train and compare anomaly detectors on Kubernetes multivariate metrics.

Includes classical unsupervised detectors, time-series methods, hybrid
(PCA + detector) pipelines, and Windowed Random Forest.
Reports accuracy, precision, recall, F1, ROC-AUC, and MCC.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "k8s_metrics.csv"
FIG_DIR = ROOT / "figures"
PAPER_IMG = Path(__file__).resolve().parents[2] / "paper" / "files" / "images"
RESULTS_DIR = ROOT / "data"

FEATURE_COLS = [
    "cpu_percent",
    "memory_percent",
    "network_rx_mbps",
    "network_tx_mbps",
    "requests_per_sec",
    "latency_ms",
    "error_rate_percent",
    "pod_restarts",
    "ready_pods",
    "disk_io_mbps",
    "locust_users",
    "hour",
    "minute_of_day",
]

RAW_KPI_COLS = [
    "cpu_percent",
    "memory_percent",
    "network_rx_mbps",
    "network_tx_mbps",
    "requests_per_sec",
    "latency_ms",
    "error_rate_percent",
    "pod_restarts",
    "ready_pods",
    "disk_io_mbps",
]


def enrich_k8s_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Add causal rolling / residual features that help sparse K8s anomalies."""
    out = df[FEATURE_COLS].copy()
    for col in RAW_KPI_COLS:
        s = df[col].astype(float)
        out[f"{col}_diff1"] = s.diff().fillna(0.0)
        out[f"{col}_roll_mean_12"] = s.rolling(12, min_periods=1).mean()
        out[f"{col}_roll_std_12"] = s.rolling(12, min_periods=1).std().fillna(0.0)
        med = s.rolling(24, min_periods=3).median()
        mad = (s - med).abs().rolling(24, min_periods=3).median()
        out[f"{col}_z24"] = ((s - med) / (1.4826 * mad + 1e-8)).fillna(0.0)
    # Cross-KPI stress signals
    out["cpu_mem_product"] = df["cpu_percent"] * df["memory_percent"] / 100.0
    out["latency_x_error"] = df["latency_ms"] * (1.0 + df["error_rate_percent"])
    out["load_gap"] = df["requests_per_sec"] - df["locust_users"].astype(float).clip(lower=1)
    out["latency_per_rps"] = df["latency_ms"] / (df["requests_per_sec"].astype(float).clip(lower=1) + 1.0)
    out["cpu_latency_ratio"] = df["cpu_percent"] / (df["latency_ms"].astype(float).clip(lower=1) + 1.0)
    for col in ("cpu_percent", "latency_ms", "error_rate_percent"):
        s = df[col].astype(float)
        out[f"{col}_roll_std_6"] = s.rolling(6, min_periods=1).std().fillna(0.0)
        out[f"{col}_resid12"] = (s - s.rolling(12, min_periods=1).mean()).fillna(0.0)
    cols = list(out.columns)
    return out.values.astype(float), cols


def _rank01(a: np.ndarray) -> np.ndarray:
    order = a.argsort().argsort().astype(float)
    return order / max(len(a) - 1, 1)


def _ecdf_map(train_vals: np.ndarray, test_vals: np.ndarray) -> np.ndarray:
    ref = np.sort(train_vals)
    return np.searchsorted(ref, test_vals, side="right") / max(len(ref), 1)


def _softmax_weights(f1s: dict[str, float], temperature: float = 0.15) -> dict[str, float]:
    names = list(f1s.keys())
    vals = np.asarray([max(float(f1s[n]), 0.0) for n in names], dtype=float)
    if vals.sum() <= 1e-12:
        w = np.ones(len(names)) / len(names)
    else:
        z = vals / max(temperature, 1e-3)
        z = z - z.max()
        e = np.exp(z)
        w = e / (e.sum() + 1e-12)
    return {n: float(wi) for n, wi in zip(names, w)}


def _calibrate_f1_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) maximizing pointwise F1 on train scores."""
    if len(np.unique(y)) < 2:
        thr = float(np.quantile(scores, 0.95))
        return thr, float(f1_score(y, (scores >= thr).astype(int), zero_division=0))
    best_t, best_f = 0.5, -1.0
    qs = np.unique(np.quantile(scores, np.linspace(0.05, 0.99, 49)))
    for t in qs:
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return best_t, best_f


def train_f1_threshold(scores_train: np.ndarray, y_train: np.ndarray, scores_test: np.ndarray) -> np.ndarray:
    """Pick train-F1 threshold then apply to test scores."""
    thr, _ = _calibrate_f1_threshold(scores_train, y_train)
    return (scores_test >= thr).astype(int)


def rf_f1_threshold_inner_holdout(
    Xw_train: np.ndarray,
    yw_train: np.ndarray,
    probs_test: np.ndarray,
    rf_params: dict,
    inner_frac: float = 0.2,
    min_threshold: float = 0.35,
) -> np.ndarray:
    """
    Fit RF on chronological outer train, pick F1 threshold on inner val holdout,
    refit on full train, apply threshold to test (avoids full-train prob overfit).
    """
    n = len(Xw_train)
    inner_start = int((1.0 - inner_frac) * n)
    if inner_start < 50 or len(np.unique(yw_train[inner_start:])) < 2:
        return (probs_test >= 0.5).astype(int)

    X_outer, y_outer = Xw_train[:inner_start], yw_train[:inner_start]
    X_inner, y_inner = Xw_train[inner_start:], yw_train[inner_start:]

    rf_cal = RandomForestClassifier(**rf_params)
    rf_cal.fit(X_outer, y_outer)
    pos_idx = list(rf_cal.classes_).index(1) if 1 in rf_cal.classes_ else 0
    probs_inner = rf_cal.predict_proba(X_inner)[:, pos_idx]

    best_t, best_f = 0.5, -1.0
    qs = np.unique(np.quantile(probs_inner, np.linspace(0.05, 0.99, 49)))
    for t in qs:
        t_floor = max(float(t), min_threshold)
        f = f1_score(y_inner, (probs_inner >= t_floor).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), t_floor
    best_t = max(best_t, min_threshold)
    return (probs_test >= best_t).astype(int)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_IMG.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    for folder in (FIG_DIR, PAPER_IMG):
        plt.savefig(folder / name, dpi=200, bbox_inches="tight")
    plt.close()


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from generate_k8s_metrics import generate_dataset

        df = generate_dataset()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return df
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


def make_windows(X: np.ndarray, y: np.ndarray, window: int = 10) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(window - 1, len(X)):
        xs.append(X[i - window + 1 : i + 1].ravel())
        ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


def predict_from_scores(scores: np.ndarray, contamination: float) -> np.ndarray:
    """Flag top-contamination fraction as anomalies (higher score = more anomalous)."""
    thr = np.quantile(scores, 1.0 - contamination)
    return (scores >= thr).astype(int)


# ---------------------------------------------------------------------------
# Detectors implemented locally (no external pyod dependency required)
# ---------------------------------------------------------------------------


class AutoencoderDetector:
    """Reconstruction-error detector using MLPRegressor as an autoencoder."""

    def __init__(self, hidden=(32, 16, 32), contamination: float = 0.05, random_state: int = 42):
        self.hidden = hidden
        self.contamination = contamination
        self.random_state = random_state
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=80,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.threshold_ = None

    def fit(self, X_normal: np.ndarray):
        self.model.fit(X_normal, X_normal)
        recon = self.model.predict(X_normal)
        err = np.mean((X_normal - recon) ** 2, axis=1)
        self.threshold_ = np.quantile(err, 1.0 - self.contamination)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        recon = self.model.predict(X)
        return np.mean((X - recon) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self.threshold_).astype(int)


class ECODDetector:
    """
    Empirical Cumulative Distribution-based Outlier Detection (ECOD).
    Scores each point by left/right tail probabilities across dimensions.
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.X_train_ = None

    def fit(self, X: np.ndarray):
        self.X_train_ = np.asarray(X, dtype=float)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, d = self.X_train_.shape
        scores = np.zeros(len(X))
        for j in range(d):
            col = self.X_train_[:, j]
            # Empirical CDF via ranking
            # P(X_j <= x) and P(X_j >= x)
            left = np.searchsorted(np.sort(col), X[:, j], side="right") / n
            right = 1.0 - np.searchsorted(np.sort(col), X[:, j], side="left") / n
            left = np.clip(left, 1.0 / (n + 1), 1.0)
            right = np.clip(right, 1.0 / (n + 1), 1.0)
            # Use the more extreme tail
            tail = np.minimum(left, right)
            scores += -np.log(tail)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_from_scores(self.score_samples(X), self.contamination)


class LODADetector:
    """
    Lightweight On-line Detector of Anomalies (LODA).
    Random 1-D projections + histogram negative-log densities.
    """

    def __init__(
        self,
        n_bins: int = 20,
        n_random_cuts: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_bins = n_bins
        self.n_random_cuts = n_random_cuts
        self.contamination = contamination
        self.random_state = random_state
        self.projections_ = None
        self.histograms_ = None
        self.bin_edges_ = None

    def fit(self, X: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        _, d = X.shape
        # Sparse-ish random projections (one non-zero weight per cut often used;
        # here we use dense unit vectors for stability on small d).
        self.projections_ = rng.normal(size=(self.n_random_cuts, d))
        self.projections_ /= np.linalg.norm(self.projections_, axis=1, keepdims=True) + 1e-12
        self.histograms_ = []
        self.bin_edges_ = []
        for w in self.projections_:
            proj = X @ w
            hist, edges = np.histogram(proj, bins=self.n_bins, density=True)
            hist = np.clip(hist, 1e-10, None)
            self.histograms_.append(hist)
            self.bin_edges_.append(edges)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = np.zeros(len(X))
        for w, hist, edges in zip(self.projections_, self.histograms_, self.bin_edges_):
            proj = X @ w
            idx = np.clip(np.searchsorted(edges, proj, side="right") - 1, 0, len(hist) - 1)
            dens = hist[idx]
            scores += -np.log(dens)
        return scores / self.n_random_cuts

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_from_scores(self.score_samples(X), self.contamination)


class HBOSDetector:
    """Histogram-Based Outlier Score (HBOS)."""

    def __init__(self, n_bins: int = 20, contamination: float = 0.05):
        self.n_bins = n_bins
        self.contamination = contamination
        self.histograms_ = None
        self.bin_edges_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        _, d = X.shape
        self.histograms_ = []
        self.bin_edges_ = []
        for j in range(d):
            hist, edges = np.histogram(X[:, j], bins=self.n_bins, density=True)
            hist = np.clip(hist, 1e-10, None)
            # Normalize so max bin = 1 (standard HBOS scaling)
            hist = hist / hist.max()
            self.histograms_.append(hist)
            self.bin_edges_.append(edges)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        scores = np.zeros(len(X))
        for j, (hist, edges) in enumerate(zip(self.histograms_, self.bin_edges_)):
            idx = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1, 0, len(hist) - 1)
            dens = np.clip(hist[idx], 1e-10, None)
            scores += -np.log(dens)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_from_scores(self.score_samples(X), self.contamination)


class RollingZScoreDetector:
    """
    Time-series residual detector: multivariate rolling z-score magnitude.
    Uses only past context (causal) via expanding/rolling statistics on train,
    then applies fitted train moments for test scoring stability.
    """

    def __init__(self, window: int = 60, contamination: float = 0.05):
        self.window = window
        self.contamination = contamination
        self.mu_ = None
        self.sigma_ = None
        self.threshold_ = None

    def fit(self, X_full_train: np.ndarray, y_train: np.ndarray | None = None):
        X = np.asarray(X_full_train, dtype=float)
        # Fit global moments on normal train samples when labels available
        if y_train is not None:
            Xn = X[y_train == 0]
        else:
            Xn = X
        self.mu_ = Xn.mean(axis=0)
        self.sigma_ = Xn.std(axis=0) + 1e-8
        scores = self.score_samples(X)
        if y_train is not None:
            scores_n = scores[y_train == 0]
        else:
            scores_n = scores
        self.threshold_ = np.quantile(scores_n, 1.0 - self.contamination)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = (X - self.mu_) / self.sigma_
        # instantaneous multivariate magnitude
        inst = np.linalg.norm(z, axis=1)
        # rolling mean of magnitude for temporal smoothing
        kernel = np.ones(self.window) / self.window
        smooth = np.convolve(inst, kernel, mode="same")
        return 0.5 * inst + 0.5 * smooth

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self.threshold_).astype(int)


class WindowedAutoencoder:
    """Sequence autoencoder on flattened sliding windows (time-series aware)."""

    def __init__(
        self,
        window: int = 10,
        hidden=(64, 24, 64),
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.window = window
        self.contamination = contamination
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=60,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.threshold_ = None

    def fit(self, X_windows_normal: np.ndarray):
        self.model.fit(X_windows_normal, X_windows_normal)
        recon = self.model.predict(X_windows_normal)
        err = np.mean((X_windows_normal - recon) ** 2, axis=1)
        self.threshold_ = np.quantile(err, 1.0 - self.contamination)
        return self

    def score_samples(self, X_windows: np.ndarray) -> np.ndarray:
        recon = self.model.predict(X_windows)
        return np.mean((X_windows - recon) ** 2, axis=1)

    def predict(self, X_windows: np.ndarray) -> np.ndarray:
        return (self.score_samples(X_windows) > self.threshold_).astype(int)


class PCAHybridDetector:
    """
    Hybrid: PCA feature reduction / reconstruction residual features,
    then a base unsupervised detector (default ECOD) on the reduced space.
    Captures the idea of representation learning + anomaly scoring.
    """

    def __init__(
        self,
        n_components: float | int = 0.95,
        contamination: float = 0.05,
        base: str = "ecod",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.contamination = contamination
        self.base = base
        self.random_state = random_state
        self.pca_ = None
        self.detector_ = None

    def _make_base(self):
        if self.base == "ecod":
            return ECODDetector(contamination=self.contamination)
        if self.base == "iforest":
            return IsolationForest(
                n_estimators=200,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
        if self.base == "ocsvm":
            return OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=min(0.5, max(0.01, self.contamination)),
            )
        raise ValueError(self.base)

    def fit(self, X_normal: np.ndarray):
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state)
        Z = self.pca_.fit_transform(X_normal)
        # Augment reduced features with reconstruction residual magnitude
        X_hat = self.pca_.inverse_transform(Z)
        resid = np.mean((X_normal - X_hat) ** 2, axis=1, keepdims=True)
        feats = np.hstack([Z, resid])
        self.detector_ = self._make_base()
        if self.base == "ecod":
            self.detector_.fit(feats)
        else:
            self.detector_.fit(feats)
        # Store feature transform helper stats via fit on normal only
        self._train_feats_ = feats
        if self.base == "ecod":
            train_scores = self.detector_.score_samples(feats)
            self._thr_ = np.quantile(train_scores, 1.0 - self.contamination)
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.pca_.transform(X)
        X_hat = self.pca_.inverse_transform(Z)
        resid = np.mean((X - X_hat) ** 2, axis=1, keepdims=True)
        return np.hstack([Z, resid])

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        feats = self._transform(X)
        if self.base == "ecod":
            return self.detector_.score_samples(feats)
        # sklearn: higher = more normal → invert
        return -self.detector_.decision_function(feats)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        if self.base == "ecod":
            return (scores >= self._thr_).astype(int)
        return (self.detector_.predict(self._transform(X)) == -1).astype(int)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        out["roc_auc"] = float("nan")
    out["tn"], out["fp"], out["fn"], out["tp"] = map(int, confusion_matrix(y_true, y_pred).ravel())
    return out


def plot_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    sample = df.iloc[:2880]
    axes[0].plot(sample["timestamp"], sample["cpu_percent"], color="#1f77b4", lw=0.8)
    axes[0].set_ylabel("CPU %")
    axes[0].set_title("Kubernetes metrics under Locust-like load (first 2 days)")
    axes[1].plot(sample["timestamp"], sample["latency_ms"], color="#ff7f0e", lw=0.8)
    axes[1].set_ylabel("Latency (ms)")
    axes[2].plot(sample["timestamp"], sample["memory_percent"], color="#2ca02c", lw=0.8)
    axes[2].set_ylabel("Memory %")
    axes[2].set_xlabel("Time")
    for ax, col in zip(axes, ["cpu_percent", "latency_ms", "memory_percent"]):
        anom = sample[sample["label"] == 1]
        seas = sample[sample["is_seasonal_pattern"] == 1]
        ax.scatter(anom["timestamp"], anom[col], c="red", s=8, label="Anomaly", zorder=5)
        ax.scatter(seas["timestamp"], seas[col], c="purple", s=6, alpha=0.5, label="Seasonal (normal)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    savefig("fig_timeseries_overview.png")


def plot_anomaly_types(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    idx = df.index[df["anomaly_type"] == "point"]
    if len(idx):
        c = idx[0]
        w = df.iloc[max(0, c - 40) : c + 40]
        axes[0].plot(w["timestamp"], w["cpu_percent"], color="#1f77b4")
        axes[0].scatter(df.loc[[c], "timestamp"], df.loc[[c], "cpu_percent"], c="red", s=40)
        axes[0].set_title("Point anomaly")
        axes[0].tick_params(axis="x", rotation=30)
    idx = df.index[df["anomaly_type"] == "contextual"]
    if len(idx):
        c = idx[0]
        w = df.iloc[max(0, c - 80) : c + 80]
        axes[1].plot(w["timestamp"], w["cpu_percent"], color="#1f77b4")
        axes[1].scatter(
            w.loc[w["anomaly_type"] == "contextual", "timestamp"],
            w.loc[w["anomaly_type"] == "contextual", "cpu_percent"],
            c="red",
            s=20,
        )
        axes[1].set_title("Contextual anomaly")
        axes[1].tick_params(axis="x", rotation=30)
    idx = df.index[df["anomaly_type"] == "collective"]
    if len(idx):
        c = idx[len(idx) // 2]
        w = df.iloc[max(0, c - 60) : c + 60]
        axes[2].plot(w["timestamp"], w["memory_percent"], color="#2ca02c")
        axes[2].scatter(
            w.loc[w["anomaly_type"] == "collective", "timestamp"],
            w.loc[w["anomaly_type"] == "collective", "memory_percent"],
            c="red",
            s=12,
        )
        axes[2].set_title("Collective anomaly")
        axes[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    savefig("fig_anomaly_types.png")


def plot_seasonal(df: pd.DataFrame) -> None:
    day = df.iloc[:1440]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(day["timestamp"], day["cpu_percent"], color="#1f77b4", label="CPU %")
    seas = day[day["is_seasonal_pattern"] == 1]
    ax.scatter(seas["timestamp"], seas["cpu_percent"], c="purple", s=10, label="Seasonal pattern (normal)")
    ax.set_title("Seasonal nightly batch pattern (labeled normal)")
    ax.legend()
    fig.tight_layout()
    savefig("fig_seasonal_pattern.png")


def plot_confusion(cms: dict[str, np.ndarray]) -> None:
    n = len(cms)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticklabels(["Normal", "Anomaly"])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on held-out test set", y=1.01)
    fig.tight_layout()
    savefig("fig_confusion_matrices.png")


def plot_metric_bars(results: pd.DataFrame) -> None:
    metrics = ["precision", "recall", "f1", "mcc", "roc_auc"]
    plot_df = results.set_index("model")[metrics]
    ax = plot_df.plot(kind="bar", figsize=(12, 5.0), rot=30)
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Algorithm comparison on Kubernetes anomaly detection")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    savefig("fig_algorithm_comparison.png")


def plot_mcc_bars(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ordered = results.sort_values("mcc", ascending=True)
    colors = ["#0F766E" if m >= 0.7 else "#2563EB" if m >= 0.4 else "#B45309" for m in ordered["mcc"]]
    ax.barh(ordered["model"], ordered["mcc"], color=colors)
    ax.set_xlabel("Matthews Correlation Coefficient (MCC)")
    ax.set_title("MCC comparison (balanced metric under class imbalance)")
    ax.set_xlim(-0.05, 1.05)
    fig.tight_layout()
    savefig("fig_mcc_comparison.png")


def plot_roc(roc_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=name, lw=1.4)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig("fig_roc_curves.png")


def plot_pca(X: np.ndarray, y: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(z[y == 0, 0], z[y == 0, 1], s=4, alpha=0.4, label="Normal", c="#1f77b4")
    ax.scatter(z[y == 1, 0], z[y == 1, 1], s=8, alpha=0.7, label="Anomaly", c="red")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection of feature space")
    ax.legend()
    fig.tight_layout()
    savefig("fig_pca_features.png")


def run() -> pd.DataFrame:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    df = load_data()

    plot_overview(df)
    plot_anomaly_types(df)
    plot_seasonal(df)

    X, _feat_names = enrich_k8s_features(df)
    y = df["label"].values.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    plot_pca(X_scaled, y)

    split = int(0.7 * len(df))
    X_train_raw, X_test_raw = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    normal_mask = y_train == 0
    X_train_normal = X_train_raw[normal_mask]
    contam = max(0.01, float(y_train.mean()))

    scores_test: dict[str, np.ndarray] = {}
    scores_train: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    y_refs: dict[str, np.ndarray] = {}

    def register(name: str, score: np.ndarray, pred: np.ndarray, y_ref: np.ndarray | None = None):
        scores_test[name] = score
        preds[name] = pred
        y_refs[name] = y_test if y_ref is None else y_ref

    # 1) Isolation Forest
    iso = IsolationForest(n_estimators=300, contamination=contam, random_state=42, n_jobs=-1)
    iso.fit(X_train_normal)
    iso_tr, iso_te = -iso.decision_function(X_train_raw), -iso.decision_function(X_test_raw)
    scores_train["Isolation Forest"] = iso_tr
    register("Isolation Forest", iso_te, train_f1_threshold(iso_tr, y_train, iso_te))

    # 2) LOF
    lof = LocalOutlierFactor(n_neighbors=25, contamination=contam, novelty=True, n_jobs=-1)
    lof.fit(X_train_normal)
    lof_tr, lof_te = -lof.decision_function(X_train_raw), -lof.decision_function(X_test_raw)
    scores_train["LOF"] = lof_tr
    register("LOF", lof_te, train_f1_threshold(lof_tr, y_train, lof_te))

    # 3) One-Class SVM
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=min(0.5, max(0.01, contam)))
    ocsvm.fit(X_train_normal)
    oc_tr, oc_te = -ocsvm.decision_function(X_train_raw), -ocsvm.decision_function(X_test_raw)
    scores_train["One-Class SVM"] = oc_tr
    register("One-Class SVM", oc_te, train_f1_threshold(oc_tr, y_train, oc_te))

    # 4) Elliptic Envelope
    try:
        ee = EllipticEnvelope(contamination=min(0.5, contam), random_state=42, support_fraction=0.7)
        ee.fit(X_train_normal)
        ee_tr, ee_te = -ee.decision_function(X_train_raw), -ee.decision_function(X_test_raw)
        scores_train["Elliptic Envelope"] = ee_tr
        register("Elliptic Envelope", ee_te, train_f1_threshold(ee_tr, y_train, ee_te))
    except Exception:
        pass

    # 5) Point-wise Autoencoder
    ae = AutoencoderDetector(contamination=contam, random_state=42)
    ae.fit(X_train_normal)
    ae_tr, ae_te = ae.score_samples(X_train_raw), ae.score_samples(X_test_raw)
    scores_train["Autoencoder"] = ae_tr
    register("Autoencoder", ae_te, train_f1_threshold(ae_tr, y_train, ae_te))

    # 6) ECOD
    ecod = ECODDetector(contamination=contam)
    ecod.fit(X_train_normal)
    ec_tr, ec_te = ecod.score_samples(X_train_raw), ecod.score_samples(X_test_raw)
    scores_train["ECOD"] = ec_tr
    register("ECOD", ec_te, train_f1_threshold(ec_tr, y_train, ec_te))

    # 7) LODA
    loda = LODADetector(contamination=contam, random_state=42)
    loda.fit(X_train_normal)
    ld_tr, ld_te = loda.score_samples(X_train_raw), loda.score_samples(X_test_raw)
    scores_train["LODA"] = ld_tr
    register("LODA", ld_te, train_f1_threshold(ld_tr, y_train, ld_te))

    # 8) HBOS
    hbos = HBOSDetector(contamination=contam)
    hbos.fit(X_train_normal)
    hb_tr, hb_te = hbos.score_samples(X_train_raw), hbos.score_samples(X_test_raw)
    scores_train["HBOS"] = hb_tr
    register("HBOS", hb_te, train_f1_threshold(hb_tr, y_train, hb_te))

    # 9) Rolling Z-Score (time-series)
    rz = RollingZScoreDetector(window=60, contamination=contam)
    rz.fit(X_train_raw, y_train)
    rz_tr, rz_te = rz.score_samples(X_train_raw), rz.score_samples(X_test_raw)
    scores_train["Rolling Z-Score"] = rz_tr
    register("Rolling Z-Score", rz_te, train_f1_threshold(rz_tr, y_train, rz_te))

    # 10) Windowed Autoencoder (time-series)
    window = 16
    Xw, yw = make_windows(X_scaled, y, window=window)
    w_split = int(0.7 * len(Xw))
    Xw_train, Xw_test = Xw[:w_split], Xw[w_split:]
    yw_train, yw_test = yw[:w_split], yw[w_split:]
    Xw_train_normal = Xw_train[yw_train == 0]
    wae = WindowedAutoencoder(window=window, contamination=contam, random_state=42)
    wae.fit(Xw_train_normal)
    wae_tr, wae_te = wae.score_samples(Xw_train), wae.score_samples(Xw_test)
    register(
        "Windowed AE",
        wae_te,
        train_f1_threshold(wae_tr, yw_train, wae_te),
        y_ref=yw_test,
    )

    # 11) PCA + ECOD hybrid
    hybrid = PCAHybridDetector(n_components=0.95, contamination=contam, base="ecod", random_state=42)
    hybrid.fit(X_train_normal)
    hy_tr, hy_te = hybrid.score_samples(X_train_raw), hybrid.score_samples(X_test_raw)
    scores_train["PCA+ECOD"] = hy_tr
    register("PCA+ECOD", hy_te, train_f1_threshold(hy_tr, y_train, hy_te))

    # Sliding-window Random Forest (supervised)
    rf_params = {
        "n_estimators": 500,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced_subsample",
        "max_features": "sqrt",
    }
    rf = RandomForestClassifier(**rf_params)
    rf.fit(Xw_train, yw_train)
    pos_idx = list(rf.classes_).index(1) if 1 in rf.classes_ else 0
    rf_prob = rf.predict_proba(Xw_test)[:, pos_idx]
    rf_pred = rf_f1_threshold_inner_holdout(Xw_train, yw_train, rf_prob, rf_params)
    register("Windowed RF", rf_prob, rf_pred, y_ref=yw_test)

    # UASE (proposed): unified adaptive stacking (shared with Yahoo/Cloud path)
    from unified_adaptive_ensemble import run_uase_on_matrix

    uase_selected_mode = ""
    uase_res = run_uase_on_matrix(
        X_scaled,
        y,
        split,
        point_adjust_obj=False,
        residual_fn=None,
        rf_window=window,
    )
    if uase_res is not None:
        uase_selected_mode = uase_res.selected_mode
        register("UASE (proposed)", uase_res.scores_test, uase_res.preds_test)

    rows = []
    cms = {}
    roc_data = {}
    for name in preds:
        yt = y_refs[name]
        yp = preds[name]
        sc = scores_test[name]
        m = min(len(yt), len(yp), len(sc))
        yt, yp, sc = yt[:m], yp[:m], sc[:m]
        met = evaluate(yt, yp, sc)
        met["model"] = name
        rows.append(met)
        cms[name] = confusion_matrix(yt, yp)
        fpr, tpr, _ = roc_curve(yt, sc)
        roc_data[name] = (fpr, tpr)

    results = pd.DataFrame(rows)[
        ["model", "accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "tn", "fp", "fn", "tp"]
    ]
    results = results.sort_values(["mcc", "f1"], ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_DIR / "ml_comparison_results.csv", index=False)
    (RESULTS_DIR / "ml_comparison_results.json").write_text(results.to_json(orient="records", indent=2))

    plot_confusion(cms)
    plot_metric_bars(results)
    plot_mcc_bars(results)
    plot_roc(roc_data)

    proposed = {"Windowed RF", "UASE (proposed)"}
    unsup = results[~results["model"].isin(proposed)]
    best_unsup = unsup.iloc[0]["model"] if len(unsup) else results.iloc[0]["model"]
    best_proposed = (
        results[results["model"].isin(proposed)].iloc[0]["model"]
        if results["model"].isin(proposed).any()
        else None
    )

    test_df = df.iloc[split:].copy().reset_index(drop=True)
    test_df = test_df.iloc[: len(preds[best_unsup])].copy()
    test_df["pred"] = preds[best_unsup][: len(test_df)]
    type_rows = []
    for tname in ["point", "contextual", "collective"]:
        mask = test_df["anomaly_type"] == tname
        if mask.sum() == 0:
            continue
        type_rows.append(
            {
                "anomaly_type": tname,
                "count": int(mask.sum()),
                "detected": int(((test_df["pred"] == 1) & mask).sum()),
                "recall": float(((test_df["pred"] == 1) & mask).sum() / mask.sum()),
            }
        )
    seas = test_df["is_seasonal_pattern"] == 1
    if seas.sum():
        type_rows.append(
            {
                "anomaly_type": "seasonal_as_fp",
                "count": int(seas.sum()),
                "detected": int(((test_df["pred"] == 1) & seas).sum()),
                "recall": float(((test_df["pred"] == 1) & seas).sum() / seas.sum()),
            }
        )
    pd.DataFrame(type_rows).to_csv(RESULTS_DIR / "per_type_detection.csv", index=False)

    summary = {
        "best_model_by_mcc": results.iloc[0]["model"],
        "best_mcc": float(results.iloc[0]["mcc"]),
        "best_model_by_f1": results.sort_values("f1", ascending=False).iloc[0]["model"],
        "best_f1": float(results.sort_values("f1", ascending=False).iloc[0]["f1"]),
        "best_unsupervised_by_mcc": best_unsup,
        "n_models": int(len(results)),
        "results": results.to_dict(orient="records"),
        "per_type": type_rows,
    }
    (RESULTS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2))

    uase_rows = results[results["model"] == "UASE (proposed)"]
    if len(uase_rows) and uase_selected_mode:
        uase_out = uase_rows.copy()
        uase_out["selected_mode"] = uase_selected_mode
        uase_out.to_csv(RESULTS_DIR / "uase_k8s_results.csv", index=False)

    print(results.to_string(index=False))
    print("\nBest by MCC:", summary["best_model_by_mcc"], summary["best_mcc"])
    print("Best unsupervised by MCC:", best_unsup)
    if best_proposed:
        prop = results[results["model"] == best_proposed].iloc[0]
        print(f"Best proposed by MCC: {best_proposed} (MCC={prop['mcc']:.3f}, mode={uase_selected_mode})")
    return results


if __name__ == "__main__":
    run()
