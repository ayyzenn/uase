#!/usr/bin/env python3
"""Optional: scrape Prometheus for live K8s metrics into the same CSV schema."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import urllib.parse
import urllib.request
import json


QUERIES = {
    "cpu_percent": 'avg(rate(container_cpu_usage_seconds_total{pod=~".*anomaly-demo.*"}[1m])) * 100',
    "memory_percent": 'avg(container_memory_working_set_bytes{pod=~".*anomaly-demo.*"}) / 1e6',
    "network_rx_mbps": 'sum(rate(container_network_receive_bytes_total{pod=~".*anomaly-demo.*"}[1m])) / 1e5',
    "network_tx_mbps": 'sum(rate(container_network_transmit_bytes_total{pod=~".*anomaly-demo.*"}[1m])) / 1e5',
}


def prom_query(base: str, expr: str) -> float:
    url = f"{base.rstrip('/')}/api/v1/query?{urllib.parse.urlencode({'query': expr})}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    result = data.get("data", {}).get("result", [])
    if not result:
        return float("nan")
    return float(result[0]["value"][1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("../data/live_k8s_metrics.csv"))
    args = parser.parse_args()

    rows = []
    for i in range(args.samples):
        row = {"timestamp": datetime.now(timezone.utc).isoformat()}
        for name, expr in QUERIES.items():
            try:
                row[name] = prom_query(args.prometheus, expr)
            except Exception as exc:  # noqa: BLE001
                print(f"query failed {name}: {exc}")
                row[name] = float("nan")
        # Fill remaining schema columns with NaN for later merge / manual labels
        for col in [
            "requests_per_sec",
            "latency_ms",
            "error_rate_percent",
            "pod_restarts",
            "ready_pods",
            "disk_io_mbps",
            "locust_users",
            "label",
            "anomaly_type",
            "is_seasonal_pattern",
            "hour",
            "minute_of_day",
        ]:
            row.setdefault(col, float("nan") if col not in ("label", "anomaly_type") else 0)
        ts = datetime.now()
        row["hour"] = ts.hour
        row["minute_of_day"] = ts.hour * 60 + ts.minute
        rows.append(row)
        print(f"[{i+1}/{args.samples}] {row['timestamp']} cpu={row['cpu_percent']}")
        if i + 1 < args.samples:
            time.sleep(args.interval)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
