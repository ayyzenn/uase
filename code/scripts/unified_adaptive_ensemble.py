#!/usr/bin/env python3
"""
Unified Adaptive Stacking Ensemble (UASE)

Cross-dataset collective detector for Kubernetes, Yahoo S5, and Cloud Monitoring.

Design (v2)
-----------
1. Fit a complementary base pool with the same scoring conventions as the
   baseline suite (incl. residual-fused LOF when a residual_fn is provided).
2. Rank-normalize each channel with a train ECDF.
3. Estimate each channel's quality on an *inner chronological holdout* of
   TRAIN (not full-train) to reduce supervised overfit.
4. Soft-weight the top-K channels by softmax(inner F1 / T); fall back to
   equal-weight top-K if scores are weak.
5. Calibrate the fused score threshold on TRAIN (point-adjust F1 on Yahoo/Cloud,
   pointwise F1 on Kubernetes).

Existing 12 detectors remain untouched; UASE is an additive proposed model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor

from anomaly_models import ECODDetector, HBOSDetector, LODADetector, make_windows


def _rank01(a: np.ndarray) -> np.ndarray:
    order = a.argsort().argsort().astype(float)
    return order / max(len(a) - 1, 1)


def _ecdf_map(train_vals: np.ndarray, test_vals: np.ndarray) -> np.ndarray:
    ref = np.sort(train_vals)
    return np.searchsorted(ref, test_vals, side="right") / max(len(ref), 1)


def _point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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


def _calibrate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    point_adjust_obj: bool = True,
) -> float:
    if len(np.unique(y_true)) < 2:
        rate = float(np.clip(max(float(y_true.mean()), 0.02), 0.02, 0.15))
        return float(np.quantile(scores, 1.0 - rate))
    best_t, best_f = float(np.median(scores)), -1.0
    qs = np.unique(
        np.concatenate(
            [
                np.quantile(scores, np.linspace(0.05, 0.999, 70)),
                np.quantile(scores, np.linspace(0.001, 0.05, 20)),
            ]
        )
    )
    for t in qs:
        pred = (scores >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        pred_eval = _point_adjust(y_true, pred) if point_adjust_obj else pred
        f = float(f1_score(y_true, pred_eval, zero_division=0))
        if f > best_f:
            best_f, best_t = f, float(t)
    if best_f <= 0:
        rate = float(np.clip(max(float(y_true.mean()), 0.02), 0.02, 0.15))
        return float(np.quantile(scores, 1.0 - rate))
    return float(best_t)


def _f1_at_calibrated(
    y: np.ndarray,
    scores: np.ndarray,
    point_adjust_obj: bool,
    thr: float | None = None,
) -> tuple[float, float]:
    """Return (f1, threshold). Threshold fit on `y`/`scores` if thr is None."""
    if len(np.unique(y)) < 2:
        return 0.0, float(np.quantile(scores, 0.95))
    if thr is None:
        thr = _calibrate_threshold(y, scores, point_adjust_obj=point_adjust_obj)
    pred = (scores >= thr).astype(int)
    pred_eval = _point_adjust(y, pred) if point_adjust_obj else pred
    return float(f1_score(y, pred_eval, zero_division=0)), float(thr)


def _robust_z_1d(mag: np.ndarray, window: int) -> np.ndarray:
    n = len(mag)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = mag[lo : i + 1]
        med = np.median(chunk)
        mad = np.median(np.abs(chunk - med)) + 1e-8
        out[i] = (mag[i] - med) / (1.4826 * mad)
    return out


def _safe_ee(X_normal: np.ndarray, X_all: np.ndarray, contam: float) -> np.ndarray | None:
    try:
        d = min(6, X_normal.shape[1], max(2, X_normal.shape[0] // 8))
        pca = PCA(n_components=d, random_state=42)
        Zn = pca.fit_transform(X_normal)
        Za = pca.transform(X_all)
        ee = EllipticEnvelope(
            contamination=min(0.5, max(0.01, contam)),
            random_state=42,
            support_fraction=0.9,
        )
        ee.fit(Zn)
        return -ee.decision_function(Za)
    except Exception:
        return None


@dataclass
class UASEResult:
    scores_test: np.ndarray
    preds_test: np.ndarray
    y_test: np.ndarray
    selected_mode: str
    base_weights: dict[str, float]
    threshold: float
    inner_f1: dict[str, float]


class UnifiedAdaptiveStackingEnsemble:
    def __init__(
        self,
        point_adjust_obj: bool = True,
        temperature: float = 0.18,
        top_k: int = 4,
        rf_window: int = 16,
        random_state: int = 42,
    ):
        self.point_adjust_obj = point_adjust_obj
        self.temperature = temperature
        self.top_k = top_k
        self.rf_window = rf_window
        self.random_state = random_state

    def _fit_bases(
        self,
        Xs: np.ndarray,
        y: np.ndarray,
        split: int,
        residual_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        X_train = Xs[:split]
        y_train = y[:split]
        Xn = X_train[y_train == 0]
        if len(Xn) < 15:
            Xn = X_train
        contam = float(np.clip(max(float(y_train.mean()), 1.0 / max(len(y_train), 1)), 0.01, 0.2))
        out: dict[str, np.ndarray] = {}

        iso = IsolationForest(
            n_estimators=200,
            contamination=min(0.5, contam),
            random_state=self.random_state,
            n_jobs=-1,
        )
        iso.fit(Xn)
        out["Isolation Forest"] = -iso.decision_function(Xs)

        k = int(np.clip(len(Xn) // 10, 5, 25))
        k = min(k, max(5, len(Xn) - 1))
        lof = LocalOutlierFactor(n_neighbors=k, contamination=min(0.5, contam), novelty=True)
        lof.fit(Xn)
        lof_raw = -lof.decision_function(Xs)

        # Residual-fused LOF (matches Yahoo baseline strength)
        if residual_fn is not None:
            try:
                res = np.asarray(residual_fn(Xs), dtype=float)
                lof_n = np.concatenate([_rank01(lof_raw[:split]), _ecdf_map(lof_raw[:split], lof_raw[split:])])
                res_n = np.concatenate([_rank01(res[:split]), _ecdf_map(res[:split], res[split:])])
                # Fixed alpha close to tuned Yahoo default; slight preference to LOF
                out["LOF+Residual"] = 0.65 * lof_n + 0.35 * res_n
                out["Residual"] = res
            except Exception:
                out["LOF"] = lof_raw
        else:
            out["LOF"] = lof_raw
            mag = np.linalg.norm(Xs, axis=1)
            out["Residual"] = np.abs(_robust_z_1d(mag, window=min(60, max(12, len(mag) // 8))))

        # Spectral Residual on first coordinate (raw/scaled value)
        try:
            from anomaly_models import spectral_residual_scores

            out["Spectral Residual"] = spectral_residual_scores(Xs[:, 0])
        except Exception:
            pass

        out["ECOD"] = ECODDetector(contamination=contam).fit(Xn).score_samples(Xs)
        out["LODA"] = (
            LODADetector(n_random_cuts=100, n_bins=20, contamination=contam, random_state=self.random_state)
            .fit(Xn)
            .score_samples(Xs)
        )
        out["HBOS"] = HBOSDetector(n_bins=20, contamination=contam).fit(Xn).score_samples(Xs)

        ee = _safe_ee(Xn, Xs, contam)
        if ee is not None:
            out["Elliptic Envelope"] = ee

        win = int(np.clip(self.rf_window, 5, max(5, split // 3)))
        Xw, yw = make_windows(Xs, y, window=win)
        w_split = max(win, split - win + 1)
        if (
            w_split > 25
            and len(Xw) - w_split > 8
            and yw[:w_split].sum() > 0
            and len(np.unique(yw[:w_split])) > 1
        ):
            rf = RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
                max_features="sqrt",
            )
            rf.fit(Xw[:w_split], yw[:w_split])
            prob = rf.predict_proba(Xw)
            p = (
                prob[:, list(rf.classes_).index(1)]
                if 1 in rf.classes_
                else np.zeros(len(Xw))
            )
            rf_all = np.zeros(len(Xs))
            rf_all[win - 1 :] = p
            if len(p):
                rf_all[: win - 1] = p[0]
            out["Windowed RF"] = rf_all

        return out

    def _rank_channels(self, raw: dict[str, np.ndarray], split: int) -> dict[str, np.ndarray]:
        ranked = {}
        for name, s in raw.items():
            s = np.asarray(s, dtype=float)
            # LOF+Residual already rank-normalized
            if name == "LOF+Residual":
                ranked[name] = s
            else:
                ranked[name] = np.concatenate([_rank01(s[:split]), _ecdf_map(s[:split], s[split:])])
        return ranked

    def _inner_split(self, y_train: np.ndarray) -> int | None:
        n = len(y_train)
        inner = max(int(0.7 * n), 20)
        anom = np.flatnonzero(y_train == 1)
        if len(anom) >= 2:
            inner = int(np.clip(inner, anom[0] + 1, anom[-1]))
        if inner < 15 or n - inner < 6:
            return None
        if len(np.unique(y_train[:inner])) < 2 or len(np.unique(y_train[inner:])) < 2:
            return None
        return inner

    def _softmax(self, f1s: dict[str, float]) -> dict[str, float]:
        names = list(f1s.keys())
        vals = np.asarray([max(float(f1s[n]), 0.0) for n in names], dtype=float)
        if vals.sum() <= 1e-12:
            w = np.ones(len(names)) / len(names)
        else:
            z = vals / max(self.temperature, 1e-3)
            z = z - z.max()
            e = np.exp(z)
            w = e / (e.sum() + 1e-12)
        return {n: float(wi) for n, wi in zip(names, w)}

    def _combine(self, ranked: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
        names = [n for n, w in weights.items() if w > 1e-8]
        if not names:
            names = list(ranked.keys())
            weights = {n: 1.0 / len(names) for n in names}
        mat = np.column_stack([ranked[n] for n in names])
        w = np.asarray([weights[n] for n in names], dtype=float)
        w = w / (w.sum() + 1e-12)
        return mat @ w

    def fit_predict(
        self,
        Xs: np.ndarray,
        y: np.ndarray,
        split: int,
        residual_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> UASEResult | None:
        Xs = np.asarray(Xs, dtype=float)
        y = np.asarray(y, dtype=int)
        if split < 30 or len(Xs) - split < 10:
            return None
        y_train, y_test = y[:split], y[split:]
        if len(np.unique(y_test)) < 2:
            return None

        raw = self._fit_bases(Xs, y, split, residual_fn=residual_fn)
        if len(raw) < 2:
            return None
        ranked = self._rank_channels(raw, split)

        inner = self._inner_split(y_train)
        inner_f1: dict[str, float] = {}
        if inner is not None:
            # Fit threshold on inner-train, evaluate on inner-valid
            for name, sc in ranked.items():
                thr = _calibrate_threshold(
                    y_train[:inner], sc[:inner], point_adjust_obj=self.point_adjust_obj
                )
                f, _ = _f1_at_calibrated(
                    y_train[inner:], sc[inner:split], self.point_adjust_obj, thr=thr
                )
                inner_f1[name] = f
        else:
            for name, sc in ranked.items():
                f, _ = _f1_at_calibrated(y_train, sc[:split], self.point_adjust_obj)
                inner_f1[name] = f

        # Keep top-K by inner F1 (always keep LOF+Residual / LOF / ECOD if present and competitive)
        ordered = sorted(inner_f1, key=inner_f1.get, reverse=True)
        keep = ordered[: max(2, self.top_k)]
        # Ensure at least one density / residual channel survives
        for must in ("LOF+Residual", "LOF", "ECOD", "Elliptic Envelope", "Residual"):
            if must in ranked and must not in keep and inner_f1.get(must, 0) >= 0.05:
                keep.append(must)
                break

        top_f = {n: inner_f1[n] for n in keep}
        soft_w = self._softmax(top_f)
        # Zero others
        weights = {n: soft_w.get(n, 0.0) for n in ranked}

        soft_scores = self._combine(ranked, weights)

        # Also consider equal-weight top-3 and single best
        top3 = ordered[:3]
        eq_w = {n: (1.0 / len(top3) if n in top3 else 0.0) for n in ranked}
        eq_scores = self._combine(ranked, eq_w)
        best_name = ordered[0]
        hard_scores = ranked[best_name]
        hard_w = {n: (1.0 if n == best_name else 0.0) for n in ranked}

        candidates = {
            "soft_topk": (soft_scores, weights),
            "equal_top3": (eq_scores, eq_w),
            "hard": (hard_scores, hard_w),
        }

        # Pick fusion mode on inner validation when possible
        best_mode, best_val = "soft_topk", -1.0
        if inner is not None:
            for mode, (sc, _) in candidates.items():
                thr = _calibrate_threshold(
                    y_train[:inner], sc[:inner], point_adjust_obj=self.point_adjust_obj
                )
                f, _ = _f1_at_calibrated(
                    y_train[inner:], sc[inner:split], self.point_adjust_obj, thr=thr
                )
                if f > best_val:
                    best_val, best_mode = f, mode
        else:
            for mode, (sc, _) in candidates.items():
                f, _ = _f1_at_calibrated(y_train, sc[:split], self.point_adjust_obj)
                if f > best_val:
                    best_val, best_mode = f, mode

        chosen, chosen_w = candidates[best_mode]
        thr = _calibrate_threshold(y_train, chosen[:split], point_adjust_obj=self.point_adjust_obj)
        pred = (chosen[split:] >= thr).astype(int)
        return UASEResult(
            scores_test=chosen[split:],
            preds_test=pred,
            y_test=y_test,
            selected_mode=f"{best_mode}:{best_name}",
            base_weights={k: round(v, 4) for k, v in chosen_w.items() if v > 1e-6},
            threshold=thr,
            inner_f1={k: round(v, 4) for k, v in inner_f1.items()},
        )


def run_uase_on_matrix(
    Xs: np.ndarray,
    y: np.ndarray,
    split: int,
    point_adjust_obj: bool = True,
    residual_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    rf_window: int = 16,
) -> UASEResult | None:
    model = UnifiedAdaptiveStackingEnsemble(
        point_adjust_obj=point_adjust_obj,
        rf_window=rf_window,
    )
    return model.fit_predict(Xs, y, split, residual_fn=residual_fn)
