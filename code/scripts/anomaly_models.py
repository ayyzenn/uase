"""Shared anomaly detectors and helpers for K8s and Yahoo S5 experiments."""
from __future__ import annotations

import numpy as np
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
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

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
        w = int(min(self.window, max(1, len(inst))))
        kernel = np.ones(w) / w
        smooth = np.convolve(inst, kernel, mode="same")
        if len(smooth) != len(inst):
            # NumPy may return kernel-length output when kernel is longer than signal
            smooth = np.convolve(inst, kernel, mode="full")
            start = (len(smooth) - len(inst)) // 2
            smooth = smooth[start : start + len(inst)]
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
                n_estimators=100,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=1,
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


class SoftVotingEnsemble:
    """Average (optionally normalized / weighted) anomaly scores from detectors."""

    def __init__(
        self,
        detectors: dict,
        contamination: float = 0.05,
        standardize: bool = True,
        weights: dict | None = None,
    ):
        self.detectors = detectors
        self.contamination = contamination
        self.standardize = standardize
        self.weights = weights  # optional {name: float}; None => equal
        self.score_means_ = {}
        self.score_stds_ = {}
        self.threshold_ = None

    def _raw_score(self, det, X: np.ndarray) -> np.ndarray:
        if hasattr(det, "score_samples"):
            return det.score_samples(X)
        return -det.decision_function(X)

    def _combine(self, stack: list[np.ndarray], names: list[str]) -> np.ndarray:
        mat = np.vstack(stack)
        if self.weights:
            w = np.asarray([float(self.weights.get(n, 1.0)) for n in names], dtype=float)
            w = w / (w.sum() + 1e-12)
            return (w[:, None] * mat).sum(axis=0)
        return mat.mean(axis=0)

    def fit(self, X_normal: np.ndarray):
        train_stack = []
        names = []
        for name, det in self.detectors.items():
            s = self._raw_score(det, X_normal)
            self.score_means_[name] = float(np.mean(s))
            self.score_stds_[name] = float(np.std(s) + 1e-8)
            if self.standardize:
                s = (s - self.score_means_[name]) / self.score_stds_[name]
            train_stack.append(s)
            names.append(name)
        ens = self._combine(train_stack, names)
        self.threshold_ = np.quantile(ens, 1.0 - self.contamination)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        stack = []
        names = []
        for name, det in self.detectors.items():
            s = self._raw_score(det, X)
            if self.standardize:
                s = (s - self.score_means_[name]) / self.score_stds_[name]
            stack.append(s)
            names.append(name)
        return self._combine(stack, names)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score_samples(X) >= self.threshold_).astype(int)


def spectral_residual_scores(
    values: np.ndarray,
    mag_window: int = 3,
    score_window: int = 21,
) -> np.ndarray:
    """
    Spectral Residual (SR) saliency scores for univariate series
    (Ren et al., Time-Series Anomaly Detection Service at Microsoft, KDD'19).
    Higher score = more anomalous.
    """
    x = np.asarray(values, dtype=float).ravel()
    n = len(x)
    if n < 8:
        return np.abs(x - np.median(x))
    # FFT of the series
    eps = 1e-8
    freq = np.fft.fft(x)
    mag = np.abs(freq)
    # log magnitude spectral residual vs local average
    log_mag = np.log(mag + eps)
    # moving average on spectral magnitude (circular-ish via convolve)
    kernel = np.ones(mag_window) / mag_window
    avg_log = np.convolve(log_mag, kernel, mode="same")
    sr = np.exp(log_mag - avg_log)
    # saliency map via inverse FFT of residual spectrum (keep original phase)
    sal = np.abs(np.fft.ifft(sr * freq / (mag + eps)))
    # local average of saliency → innovation
    k2 = np.ones(score_window) / score_window
    sal_avg = np.convolve(sal, k2, mode="same")
    scores = (sal - sal_avg) / (sal_avg + eps)
    return np.abs(scores)


class SpectralResidualDetector:
    """Univariate Spectral Residual detector; expects first feature column = raw value."""

    def __init__(self, contamination: float = 0.05, mag_window: int = 3, score_window: int = 21):
        self.contamination = contamination
        self.mag_window = mag_window
        self.score_window = score_window
        self.threshold_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        vals = X[:, 0] if X.ndim == 2 else X
        scores = spectral_residual_scores(vals, self.mag_window, self.score_window)
        self.threshold_ = np.quantile(scores, 1.0 - self.contamination)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        vals = X[:, 0] if X.ndim == 2 else X
        return spectral_residual_scores(vals, self.mag_window, self.score_window)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_from_scores(self.score_samples(X), self.contamination)



def evaluate(y_true, y_pred, scores=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1 else 0.0,
    }
    # fix MCC when only one class in pred but two in true
    if len(np.unique(y_true)) > 1:
        try:
            out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception:
            out["mcc"] = 0.0
    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"], out["fp"], out["fn"], out["tp"] = map(int, cm)
    return out
