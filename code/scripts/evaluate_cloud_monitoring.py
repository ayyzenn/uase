#!/usr/bin/env python3
"""
Evaluate all anomaly detectors on the Microsoft Cloud Monitoring Dataset.

Mirrors the Yahoo S5 evaluation pipeline (same featurization, model suite,
point-adjust metrics, and aggregation). Domain folder names act as benchmarks.
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_yahoo_s5 import (  # noqa: E402
    FIG_DIR,
    PAPER_IMG,
    RESULTS_DIR,
    aggregate,
    ensure_dirs,
    featurize_univariate,
    run_models_on_series,
    savefig,
)

CLOUD_ROOT = ROOT / "data" / "cloud-monitoring-dataset" / "data"


def load_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Cloud Monitoring CSV: TimeStamp, Value, Label (headers may be quoted)."""
    df = pd.read_csv(path)
    # Strip quotes/whitespace from column names
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    value_col = cols.get("value")
    if value_col is None:
        raise ValueError(f"No Value column in {path}")
    values = df[value_col].astype(float).values
    # Some series have missing Value entries; interpolate then fill edges.
    s = pd.Series(values)
    if s.isna().any():
        s = s.interpolate(limit_direction="both").fillna(0.0)
        values = s.to_numpy(dtype=float)

    label_col = cols.get("label")
    if label_col is None:
        raise ValueError(f"No Label column in {path}")
    y = df[label_col].astype(int).values
    return values, y


def list_domains() -> list[str]:
    if not CLOUD_ROOT.is_dir():
        raise FileNotFoundError(f"Cloud Monitoring data not found at {CLOUD_ROOT}")
    return sorted(d.name for d in CLOUD_ROOT.iterdir() if d.is_dir())


def list_series(domain: str) -> list[Path]:
    folder = CLOUD_ROOT / domain
    # Ignore *.csv-metadata.json; only real CSVs
    return sorted(p for p in folder.glob("*.csv") if p.is_file())


def discover_all_series() -> list[tuple[str, Path]]:
    """Return (domain, path) pairs sorted by domain then filename."""
    pairs: list[tuple[str, Path]] = []
    for domain in list_domains():
        for path in list_series(domain):
            pairs.append((domain, path))
    return pairs


def evaluate_domain(domain: str, files: list[Path]) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    print(f"\n=== {domain}: {len(files)} series ===")
    rows = []
    skipped = []
    for i, path in enumerate(files, 1):
        try:
            values, y = load_series(path)
            n_anom = int(y.sum())
            if n_anom == 0:
                skipped.append((path.name, "zero anomalies"))
                print(f"  skip {path.name}: zero anomalies (false-positive probe)")
                continue
            X = featurize_univariate(values)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            metrics = run_models_on_series(X, y)
        except Exception as exc:
            skipped.append((path.name, str(exc)))
            print(f"  skip {path.name}: {exc}")
            continue
        if not metrics:
            skipped.append((path.name, "choose_split/run_models returned empty"))
            print(f"  skip {path.name}: no valid train/test split (need positives in both)")
            continue
        for model, met in metrics.items():
            rows.append(
                {
                    "benchmark": domain,
                    "series": path.name,
                    "model": model,
                    "n_points": len(values),
                    "n_anomalies": n_anom,
                    **met,
                }
            )
        print(f"  [{i}/{len(files)}] {path.name}: {n_anom} anomalies, {len(metrics)} models")
    return pd.DataFrame(rows), skipped


def plot_cloud_heatmap(agg: pd.DataFrame) -> None:
    value_col = "mcc_pa" if "mcc_pa" in agg.columns else "mcc"
    pivot = agg.pivot(index="model", columns="benchmark", values=value_col)
    model_order = [
        "Windowed RF",
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
    domains = sorted(pivot.columns.tolist())
    rows = [m for m in model_order if m in pivot.index] + [
        m for m in pivot.index if m not in model_order
    ]
    pivot = pivot.loc[rows, domains]
    fig_w = max(10.0, 1.2 * len(domains) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 7.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Cloud Monitoring: mean {value_col.upper()} by model and domain")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    savefig("fig_cloud_monitoring_mcc_heatmap.png")


def plot_cloud_bars(overall: pd.DataFrame) -> None:
    value_col = "mcc_pa" if "mcc_pa" in overall.columns else "mcc"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ordered = overall.sort_values(value_col, ascending=True)
    colors = [
        "#0F766E" if "Proposed" in m else "#2563EB" for m in ordered["model"]
    ]
    ax.barh(ordered["model"], ordered[value_col], color=colors)
    ax.set_xlabel(f"Mean {value_col} across Cloud Monitoring domains")
    ax.set_title("Cloud Monitoring Dataset (proposed method highlighted)")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    savefig("fig_cloud_monitoring_mcc_bars.png")


def run(max_series: int | None = None) -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    all_pairs = discover_all_series()
    if max_series is not None:
        all_pairs = all_pairs[:max_series]
        print(f"Smoke / limited run: evaluating {len(all_pairs)} series")

    # Group by domain for reporting
    by_domain: dict[str, list[Path]] = {}
    for domain, path in all_pairs:
        by_domain.setdefault(domain, []).append(path)

    all_rows: list[pd.DataFrame] = []
    all_skipped: list[dict] = []
    for domain in sorted(by_domain.keys()):
        part, skipped = evaluate_domain(domain, by_domain[domain])
        if not part.empty:
            all_rows.append(part)
        for name, reason in skipped:
            all_skipped.append({"benchmark": domain, "series": name, "reason": reason})

    if not all_rows:
        print("No series produced evaluation metrics.")
        summary = {
            "n_series_evaluated": 0,
            "skipped": all_skipped,
            "message": "No evaluable series (all zero-anomaly or invalid splits).",
        }
        (RESULTS_DIR / "cloud_monitoring_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        return

    detail = pd.concat(all_rows, ignore_index=True)
    detail.to_csv(RESULTS_DIR / "cloud_monitoring_per_series.csv", index=False)

    agg = aggregate(detail)
    # Rename for clarity in cloud outputs (benchmark == domain)
    agg.to_csv(RESULTS_DIR / "cloud_monitoring_by_domain.csv", index=False)

    metric_cols = [
        c
        for c in [
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
        if c in detail.columns
    ]
    overall = (
        detail.groupby("model", as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values("f1_pa" if "f1_pa" in metric_cols else "f1", ascending=False)
        .reset_index(drop=True)
    )
    overall.to_csv(RESULTS_DIR / "cloud_monitoring_overall.csv", index=False)

    plot_cloud_heatmap(agg)
    plot_cloud_bars(overall)

    rank_col = "mcc_pa" if "mcc_pa" in agg.columns else "mcc"
    wide = agg.pivot(index="model", columns="benchmark", values=rank_col)
    domains = sorted(wide.columns.tolist())
    wide = wide[domains]
    wide["mean_mcc_pa"] = wide.mean(axis=1)
    wide = wide.sort_values("mean_mcc_pa", ascending=False)
    wide.to_csv(RESULTS_DIR / "cloud_monitoring_mcc_wide.csv")

    n_series = int(detail[["benchmark", "series"]].drop_duplicates().shape[0])
    proposed = overall[overall["model"] == "Windowed RF"]
    proposed_rec = proposed.iloc[0].to_dict() if not proposed.empty else {}

    summary = {
        "best_model_by_mean_f1_pa": str(overall.iloc[0]["model"]),
        "best_mean_f1_pa": float(overall.iloc[0].get("f1_pa", overall.iloc[0]["f1"])),
        "best_model_by_mean_mcc_pa": str(wide.index[0]),
        "best_mean_mcc_pa": float(wide.iloc[0]["mean_mcc_pa"]),
        "n_series_evaluated": n_series,
        "n_series_skipped": len(all_skipped),
        "skipped": all_skipped,
        "proposed_windowed_rf": {
            k: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
            for k, v in proposed_rec.items()
        },
        "overall": overall.to_dict(orient="records"),
        "by_domain_mcc_pa": wide.reset_index().to_dict(orient="records"),
    }
    (RESULTS_DIR / "cloud_monitoring_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print("\n===== Cloud Monitoring overall (mean across series) =====")
    print(overall.to_string(index=False))
    print("\n===== Point-adjust MCC by domain =====")
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
    if proposed_rec:
        print(
            "\nWindowed RF:",
            f"F1_pa={proposed_rec.get('f1_pa', float('nan')):.4f}",
            f"MCC_pa={proposed_rec.get('mcc_pa', float('nan')):.4f}",
        )
    if all_skipped:
        print(f"\nSkipped {len(all_skipped)} series:")
        for s in all_skipped:
            print(f"  {s['benchmark']}/{s['series']}: {s['reason']}")


if __name__ == "__main__":
    max_series = None
    if "--max-series" in sys.argv:
        max_series = int(sys.argv[sys.argv.index("--max-series") + 1])
    run(max_series=max_series)
