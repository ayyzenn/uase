#!/usr/bin/env python3
"""
Causal / leakage-safe time-series normalization for UASE ablation.

All scalers fit statistics on TRAIN only (or expand causally) and transform
TEST without using future values. This isolates preprocessing from the
UASE detector pool.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


@dataclass
class NormResult:
    name: str
    X_scaled: np.ndarray
    notes: str


def _fit_transform_train_test(
    X: np.ndarray,
    split: int,
    scaler,
) -> np.ndarray:
    X_tr, X_te = X[:split], X[split:]
    scaler.fit(X_tr)
    return np.vstack([scaler.transform(X_tr), scaler.transform(X_te)])


def global_zscore_train(X: np.ndarray, split: int) -> NormResult:
    """Train-only StandardScaler (corrected Z-score; no test leakage)."""
    Xs = _fit_transform_train_test(X, split, StandardScaler())
    return NormResult("A_train_zscore", Xs, "μ,σ from train only; test transformed with train stats")


def leaked_full_zscore(X: np.ndarray, split: int) -> NormResult:
    """Original pipeline behavior: StandardScaler fit on full series (leakage)."""
    Xs = StandardScaler().fit_transform(X)
    return NormResult(
        "A0_leaked_full_zscore",
        Xs,
        "LEGACY: μ,σ from full series including test (temporal leakage)",
    )


def robust_scaler_train(X: np.ndarray, split: int) -> NormResult:
    """Train-only RobustScaler: (x - median) / IQR."""
    Xs = _fit_transform_train_test(X, split, RobustScaler())
    return NormResult("B_robust_iqr", Xs, "median/IQR from train; robust to extreme anomalies in train")


def minmax_train(X: np.ndarray, split: int) -> NormResult:
    """Train-only MinMax to [0, 1]."""
    Xs = _fit_transform_train_test(X, split, MinMaxScaler())
    return NormResult("C_minmax_train", Xs, "min/max from train; sensitive to train extremes")


def causal_rolling_zscore(X: np.ndarray, split: int, window: int = 60) -> NormResult:
    """
    Per-feature causal rolling robust z-score.
    At time t uses only [t-W+1, t] (or expanding until W).
    Split is unused for fitting (fully causal) but kept for API uniformity.
    """
    del split  # fully causal; no train/test fit asymmetry
    import pandas as pd

    X = np.asarray(X, dtype=float)
    df = pd.DataFrame(X)
    med = df.rolling(window, min_periods=1).median()
    mad = (df - med).abs().rolling(window, min_periods=1).median()
    out = ((df - med) / (1.4826 * mad + 1e-8)).fillna(0.0).to_numpy(dtype=float)
    return NormResult(
        "D_causal_rolling_z60",
        out,
        f"causal rolling median/MAD z, window={window}; adapts local level/scale",
    )


def seasonal_residual_then_zscore(
    X: np.ndarray,
    split: int,
    period: int = 1440,
) -> NormResult:
    """
    Subtract causal seasonal mean (per phase of `period`), then train-only Z-score.
    For minute K8s data period=1440 (daily). For hourly Yahoo use 24.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    resid = np.zeros_like(X)
    for j in range(d):
        col = X[:, j]
        sums = np.zeros(period, dtype=float)
        counts = np.zeros(period, dtype=float)
        for t in range(n):
            phase = t % period
            if counts[phase] > 0:
                resid[t, j] = col[t] - sums[phase] / counts[phase]
            else:
                resid[t, j] = 0.0
            sums[phase] += col[t]
            counts[phase] += 1.0
    Xs = _fit_transform_train_test(resid, split, StandardScaler())
    return NormResult(
        "E_seasonal_resid_z",
        Xs,
        f"causal seasonal mean residual (period={period}) then train Z-score",
    )


def diff1_then_zscore(X: np.ndarray, split: int) -> NormResult:
    """First difference (removes slow level/trend), then train-only Z-score."""
    X = np.asarray(X, dtype=float)
    dX = np.zeros_like(X)
    dX[1:] = np.diff(X, axis=0)
    Xs = _fit_transform_train_test(dX, split, StandardScaler())
    return NormResult(
        "F_diff1_zscore",
        Xs,
        "Δx_t = x_t - x_{t-1}; train Z-score on differences (detrends level)",
    )


def none_raw(X: np.ndarray, split: int) -> NormResult:
    """No scaling (control)."""
    del split
    return NormResult("G_raw_unscaled", np.asarray(X, dtype=float), "no normalization")


NORMALIZERS = {
    "A0_leaked_full_zscore": leaked_full_zscore,
    "A_train_zscore": global_zscore_train,
    "B_robust_iqr": robust_scaler_train,
    "C_minmax_train": minmax_train,
    "D_causal_rolling_z60": causal_rolling_zscore,
    "E_seasonal_resid_z": seasonal_residual_then_zscore,
    "F_diff1_zscore": diff1_then_zscore,
    "G_raw_unscaled": none_raw,
}


def apply_normalizer(name: str, X: np.ndarray, split: int, **kwargs) -> NormResult:
    if name not in NORMALIZERS:
        raise KeyError(f"Unknown normalizer {name}; choose from {list(NORMALIZERS)}")
    fn = NORMALIZERS[name]
    # seasonal period override
    if name == "E_seasonal_resid_z" and "period" in kwargs:
        return seasonal_residual_then_zscore(X, split, period=int(kwargs["period"]))
    if name == "D_causal_rolling_z60" and "window" in kwargs:
        return causal_rolling_zscore(X, split, window=int(kwargs["window"]))
    return fn(X, split)
