#!/usr/bin/env python3
"""
Locate the Yahoo Webscope S5 (A1-A4) labelled anomaly benchmark.

The dataset (ydata-labeled-time-series-anomalies-v1_0) is licensed by Yahoo for
non-commercial research by recipients who accepted Yahoo's data terms, and it may
NOT be redistributed. This repository therefore does not ship or download it.
Obtain it yourself from Yahoo (see README.md) and unpack it so that

    code/data/yahoo_s5/A1Benchmark/real_1.csv
    code/data/yahoo_s5/A2Benchmark/synthetic_1.csv
    code/data/yahoo_s5/A3Benchmark/A3Benchmark-TS1.csv
    code/data/yahoo_s5/A4Benchmark/A4Benchmark-TS1.csv
exist.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "yahoo_s5"


def download(force: bool = False) -> Path:
    """Kept for API compatibility: only checks that the data are present."""
    expected = [DEST / b for b in ("A1Benchmark", "A2Benchmark", "A3Benchmark", "A4Benchmark")]
    if all(p.exists() and any(p.glob("*.csv")) for p in expected):
        return DEST
    raise SystemExit(
        f"Yahoo S5 not found under {DEST}. It cannot be redistributed; request it from Yahoo "
        "(https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70 or the gated "
        "https://huggingface.co/datasets/YahooResearch/ydata-labeled-time-series-anomalies-v1_0) "
        "and unpack A1Benchmark..A4Benchmark into that folder (see README.md)."
    )


if __name__ == "__main__":
    print(download())
