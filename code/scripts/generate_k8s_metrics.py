#!/usr/bin/env python3
"""
Kubernetes-style metrics generator with Locust-like load and anomaly injection.

Hardened to mirror Yahoo Webscope S5 difficulty patterns:
  - overlapping 12h / daily / weekly seasonality (A3/A4 style)
  - additive trend + abrupt change-points (A4)
  - sparse, mostly short anomalies (~1% rate; Yahoo-like)
  - subtle outliers near the noise floor + seasonal lookalikes labeled normal
  - heteroscedastic noise and weak cross-metric coupling

Labels: 0 = normal (includes seasonal peaks), 1 = anomaly.
Anomaly subtypes: point, contextual, collective, changepoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)


def _sin(t: np.ndarray, period: float, phase: float = 0.0) -> np.ndarray:
    return np.sin(2 * np.pi * t / period + phase)


def locust_users(t: np.ndarray, period: int = 1440) -> np.ndarray:
    """
    Concurrent Locust users with Yahoo-like multi-seasonality:
    12h (period/2), daily (period), weekly (7*period), plus trend-ish drift.
    """
    # Primary daily cycle (peak midday)
    daily = 38 + 28 * _sin(t, period, -np.pi / 2)
    # Secondary 12-hour seasonality (A3 seasonality1 analogue)
    half_day = 8 * _sin(t, period / 2, 0.4)
    # Weekly modulation
    weekly = 1.0 + 0.18 * _sin(t, 7 * period, 0.2)
    # Slow non-stationary drift (trend analogue)
    trend = 0.0015 * t
    # Occasional normal traffic bursts (NOT anomalies)
    bursts = np.zeros_like(t, dtype=float)
    n_bursts = max(8, len(t) // 500)
    for _ in range(n_bursts):
        center = int(RNG.integers(0, len(t)))
        width = int(RNG.integers(10, 45))
        amp = float(RNG.uniform(8, 28))
        idx = np.arange(max(0, center - width), min(len(t), center + width))
        bursts[idx] += amp * np.exp(-0.5 * ((idx - center) / (width / 2.5)) ** 2)
    # Heteroscedastic noise (higher variance at peak load)
    base = (daily + half_day) * weekly + trend + bursts
    noise_scale = 2.0 + 0.04 * np.clip(base, 0, None)
    noise = RNG.normal(0, 1.0, size=len(t)) * noise_scale
    users = np.clip(base + noise, 5, None)
    return users


def base_metrics(users: np.ndarray) -> dict[str, np.ndarray]:
    """Map Locust-like load to Kubernetes resource and app metrics."""
    n = len(users)
    # Shared latent factors → correlated noise (harder than independent spikes)
    latent = RNG.normal(0, 1.0, n)
    cpu = 0.35 * users + 1.8 * latent + RNG.normal(8, 3.5, n)
    memory = 0.28 * users + 1.2 * latent + RNG.normal(25, 4.0, n)
    network_rx = 0.9 * users + 0.8 * latent + RNG.normal(5, 3.0, n)
    network_tx = 0.7 * users + 0.6 * latent + RNG.normal(4, 2.5, n)
    rps = 1.8 * users + 1.0 * latent + RNG.normal(0, 6.0, n)
    # Latency with mild nonlinear queueing + weekend-like slowdowns already in users
    latency_ms = 20 + 0.08 * users + 0.0012 * users**2 + 2.0 * latent + RNG.normal(0, 4.0, n)
    error_rate = np.clip(0.002 * users + 0.15 * np.abs(latent) + RNG.normal(0.12, 0.08, n), 0, None)
    pod_restarts = RNG.poisson(0.03, n).astype(float)
    ready_pods = np.clip(np.round(3 + users / 55 + RNG.normal(0, 0.25, n)), 2, 12)
    disk_io = 0.4 * users + 0.5 * latent + RNG.normal(10, 3.0, n)

    return {
        "cpu_percent": np.clip(cpu, 1, 100),
        "memory_percent": np.clip(memory, 5, 100),
        "network_rx_mbps": np.clip(network_rx, 0, None),
        "network_tx_mbps": np.clip(network_tx, 0, None),
        "requests_per_sec": np.clip(rps, 0, None),
        "latency_ms": np.clip(latency_ms, 5, None),
        "error_rate_percent": np.clip(error_rate, 0, 100),
        "pod_restarts": pod_restarts,
        "ready_pods": ready_pods,
        "disk_io_mbps": np.clip(disk_io, 0, None),
        "locust_users": users,
    }


def inject_seasonal_patterns(metrics: dict[str, np.ndarray], period: int = 1440) -> np.ndarray:
    """
    Recurring seasonal peaks (nightly backups / batch jobs) labeled NORMAL.
    Yahoo-hard: these overlap anomaly shapes but must not be flagged.
    """
    n = len(metrics["cpu_percent"])
    seasonal_mask = np.zeros(n, dtype=bool)
    for day_start in range(0, n, period):
        # Nightly batch
        start = day_start + 120
        end = min(day_start + 180, n)
        if start < n:
            idx = np.arange(start, end)
            seasonal_mask[idx] = True
            metrics["cpu_percent"][idx] = np.clip(
                metrics["cpu_percent"][idx] + RNG.uniform(18, 32), 1, 100
            )
            metrics["disk_io_mbps"][idx] = metrics["disk_io_mbps"][idx] + RNG.uniform(30, 55)
            metrics["memory_percent"][idx] = np.clip(
                metrics["memory_percent"][idx] + RNG.uniform(6, 14), 5, 100
            )
            metrics["latency_ms"][idx] = metrics["latency_ms"][idx] + RNG.uniform(4, 10)
        # Mid-day cache-warm / deploy window (second seasonal lookalike)
        start2 = day_start + 700
        end2 = min(day_start + 740, n)
        if start2 < n:
            idx2 = np.arange(start2, end2)
            seasonal_mask[idx2] = True
            metrics["requests_per_sec"][idx2] = metrics["requests_per_sec"][idx2] + RNG.uniform(25, 45)
            metrics["cpu_percent"][idx2] = np.clip(
                metrics["cpu_percent"][idx2] + RNG.uniform(10, 20), 1, 100
            )
    return seasonal_mask


def inject_changepoints(
    metrics: dict[str, np.ndarray], count: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """
    A4-style change-points: lasting level shifts after a short transition.
    Labeled anomalous only at the change onset (1–3 steps), not the whole new regime.
    """
    n = len(metrics["cpu_percent"])
    labels = np.zeros(n, dtype=int)
    subtypes = np.array(["normal"] * n, dtype=object)
    placed = 0
    attempts = 0
    while placed < count and attempts < count * 30:
        attempts += 1
        # Keep away from edges
        t0 = int(RNG.integers(int(0.15 * n), int(0.85 * n)))
        if labels[max(0, t0 - 30) : min(n, t0 + 30)].any():
            continue
        # Level shift magnitudes (subtle → hard)
        d_cpu = float(RNG.uniform(8, 18) * RNG.choice([-1, 1]))
        d_mem = float(RNG.uniform(5, 14) * RNG.choice([-1, 1]))
        d_lat = float(RNG.uniform(8, 25))
        d_err = float(RNG.uniform(0.5, 2.5))
        metrics["cpu_percent"][t0:] = np.clip(metrics["cpu_percent"][t0:] + d_cpu, 1, 100)
        metrics["memory_percent"][t0:] = np.clip(metrics["memory_percent"][t0:] + d_mem, 5, 100)
        metrics["latency_ms"][t0:] = np.clip(metrics["latency_ms"][t0:] + d_lat, 5, None)
        metrics["error_rate_percent"][t0:] = np.clip(
            metrics["error_rate_percent"][t0:] + d_err, 0, 100
        )
        # Mark only the short onset as anomaly (Yahoo A4 changepoint style)
        span = int(RNG.integers(1, 4))
        end = min(n, t0 + span)
        labels[t0:end] = 1
        subtypes[t0:end] = "changepoint"
        placed += 1
    return labels, subtypes


def inject_point_anomalies(metrics: dict[str, np.ndarray], count: int = 35) -> tuple[np.ndarray, np.ndarray]:
    """
    Sparse single-timestep / short outliers (Yahoo A2/A3 style).
    Magnitudes are intentionally milder than before so they sit near noise.
    """
    n = len(metrics["cpu_percent"])
    labels = np.zeros(n, dtype=int)
    subtypes = np.array(["normal"] * n, dtype=object)
    chosen = RNG.choice(np.arange(60, n - 60), size=min(count, n - 120), replace=False)
    for i in chosen:
        # Mix of mild and stronger points (mostly mild)
        mild = RNG.random() < 0.65
        amp = RNG.uniform(12, 22) if mild else RNG.uniform(28, 45)
        metrics["cpu_percent"][i] = float(np.clip(metrics["cpu_percent"][i] + amp, 1, 100))
        metrics["memory_percent"][i] = float(
            np.clip(metrics["memory_percent"][i] + amp * 0.45, 5, 100)
        )
        metrics["latency_ms"][i] = metrics["latency_ms"][i] * (RNG.uniform(1.4, 2.0) if mild else RNG.uniform(2.2, 3.5))
        metrics["error_rate_percent"][i] = metrics["error_rate_percent"][i] + (
            RNG.uniform(1.5, 4.0) if mild else RNG.uniform(6, 14)
        )
        if not mild and RNG.random() < 0.4:
            metrics["pod_restarts"][i] = metrics["pod_restarts"][i] + RNG.integers(1, 3)
        # Occasional 2–3 step run (Yahoo A2 mean run ~2.3)
        run = 1 if RNG.random() < 0.55 else int(RNG.integers(2, 4))
        for k in range(run):
            j = min(n - 1, i + k)
            labels[j] = 1
            subtypes[j] = "point"
            if k > 0:
                metrics["cpu_percent"][j] = float(
                    np.clip(metrics["cpu_percent"][j] + amp * 0.7, 1, 100)
                )
                metrics["latency_ms"][j] = metrics["latency_ms"][j] * RNG.uniform(1.2, 1.8)
    return labels, subtypes


def inject_contextual_anomalies(
    metrics: dict[str, np.ndarray], period: int = 1440, count: int = 18
) -> tuple[np.ndarray, np.ndarray]:
    """High load at night when Locust users should be low (context-dependent)."""
    n = len(metrics["cpu_percent"])
    labels = np.zeros(n, dtype=int)
    subtypes = np.array(["normal"] * n, dtype=object)
    night_idx = []
    for day_start in range(0, n, period):
        night_idx.extend(range(day_start, min(day_start + 240, n)))
        night_idx.extend(
            range(max(day_start + period - 240, day_start), min(day_start + period, n))
        )
    night_idx = np.array(sorted(set(night_idx)), dtype=int)
    if len(night_idx) == 0:
        return labels, subtypes
    # Avoid already-labeled indices later by caller merge; pick candidates here
    chosen = RNG.choice(night_idx, size=min(count, len(night_idx)), replace=False)
    for i in chosen:
        # Milder than v1 so seasonal night batch is confusable
        metrics["locust_users"][i] = metrics["locust_users"][i] + RNG.uniform(35, 70)
        metrics["cpu_percent"][i] = float(np.clip(metrics["cpu_percent"][i] + RNG.uniform(20, 40), 1, 100))
        metrics["requests_per_sec"][i] = metrics["requests_per_sec"][i] + RNG.uniform(40, 90)
        metrics["latency_ms"][i] = metrics["latency_ms"][i] + RNG.uniform(35, 80)
        metrics["error_rate_percent"][i] = metrics["error_rate_percent"][i] + RNG.uniform(2, 7)
        labels[i] = 1
        subtypes[i] = "contextual"
    return labels, subtypes


def inject_collective_anomalies(
    metrics: dict[str, np.ndarray], count: int = 8, length: tuple[int, int] = (8, 22)
) -> tuple[np.ndarray, np.ndarray]:
    """Short collective windows (memory leak / cascading failure) — fewer/shorter than before."""
    n = len(metrics["cpu_percent"])
    labels = np.zeros(n, dtype=int)
    subtypes = np.array(["normal"] * n, dtype=object)
    placed = 0
    attempts = 0
    while placed < count and attempts < count * 40:
        attempts += 1
        win = int(RNG.integers(length[0], length[1] + 1))
        start = int(RNG.integers(80, max(81, n - win - 80)))
        end = start + win
        if labels[start:end].any():
            continue
        # Gradual but subtler ramp
        ramp = np.linspace(1.0, 1.7, win)
        metrics["memory_percent"][start:end] = np.clip(
            metrics["memory_percent"][start:end] * ramp + RNG.uniform(4, 12), 5, 100
        )
        metrics["latency_ms"][start:end] = metrics["latency_ms"][start:end] * ramp
        metrics["error_rate_percent"][start:end] = (
            metrics["error_rate_percent"][start:end] + ramp * RNG.uniform(1.5, 4.0)
        )
        metrics["cpu_percent"][start:end] = np.clip(
            metrics["cpu_percent"][start:end] + ramp * 5, 1, 100
        )
        metrics["pod_restarts"][start:end] = metrics["pod_restarts"][start:end] + RNG.poisson(
            0.25, win
        )
        labels[start:end] = 1
        subtypes[start:end] = "collective"
        placed += 1
    return labels, subtypes


def generate_dataset(
    n_points: int = 10080,
    period: int = 1440,
    seed: int = 42,
) -> pd.DataFrame:
    global RNG
    RNG = np.random.default_rng(seed)

    t = np.arange(n_points)
    users = locust_users(t, period=period)
    metrics = base_metrics(users)

    seasonal_mask = inject_seasonal_patterns(metrics, period=period)

    # Inject anomalies (changepoints first so later injections avoid them where possible)
    lab_cp, sub_cp = inject_changepoints(metrics, count=6)
    lab_c, sub_c = inject_collective_anomalies(metrics, count=5, length=(6, 16))
    lab_x, sub_x = inject_contextual_anomalies(metrics, period=period, count=20)
    lab_p, sub_p = inject_point_anomalies(metrics, count=45)

    labels = np.maximum.reduce([lab_cp, lab_c, lab_x, lab_p])
    subtypes = np.array(["normal"] * n_points, dtype=object)
    subtypes[sub_cp == "changepoint"] = "changepoint"
    subtypes[sub_c == "collective"] = "collective"
    subtypes[sub_x == "contextual"] = "contextual"
    subtypes[sub_p == "point"] = "point"
    # Seasonal windows stay normal unless overwritten by true anomaly injection
    is_seasonal = seasonal_mask & (labels == 0)

    timestamps = pd.date_range("2024-01-01", periods=n_points, freq="min")
    df = pd.DataFrame(metrics)
    df.insert(0, "timestamp", timestamps)
    df["label"] = labels
    df["anomaly_type"] = subtypes
    df["is_seasonal_pattern"] = is_seasonal.astype(int)
    df["hour"] = df["timestamp"].dt.hour
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate K8s anomaly detection dataset")
    parser.add_argument("--n-points", type=int, default=10080, help="Number of minutes (default 7 days)")
    parser.add_argument("--period", type=int, default=1440, help="Seasonal period in minutes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "k8s_metrics.csv",
    )
    args = parser.parse_args()

    df = generate_dataset(n_points=args.n_points, period=args.period, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    summary = {
        "n_points": int(len(df)),
        "n_anomalies": int(df["label"].sum()),
        "anomaly_rate": float(df["label"].mean()),
        "point": int((df["anomaly_type"] == "point").sum()),
        "contextual": int((df["anomaly_type"] == "contextual").sum()),
        "collective": int((df["anomaly_type"] == "collective").sum()),
        "changepoint": int((df["anomaly_type"] == "changepoint").sum()),
        "seasonal_normal": int(df["is_seasonal_pattern"].sum()),
        "path": str(args.out),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
