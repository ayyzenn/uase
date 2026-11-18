#!/usr/bin/env python3
"""Kubernetes (single test split) uncertainty for protocol v2:
moving-block bootstrap (block = 60 min, 2000 resamples) of F1/MCC for each model and of the
MCC difference UASE-v2 minus model, plus exact McNemar test on point-wise errors.
Predictions are rebuilt from saved scores exactly as in uase_meta.py.
Output: data/v2/stats/k8s_<norm>_bootstrap.csv"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, matthews_corrcoef
sys.path.insert(0, str(Path(__file__).resolve().parent))
import uase_meta as um
import uase_v2 as u

norm = sys.argv[1] if len(sys.argv) > 1 else "A"
sel = {"A": ["Windowed RF", "Autoencoder"], "D": ["Windowed RF"]}[norm]  # from data/v2/meta/k8s_k8s_<norm>_selection.json
it = um.load("k8s", "test", f"k8s_{norm}")[0]
ch, ytr, yte = it["ch"], it["ytr"], it["yte"]
preds = {}
for n, (o, t) in ch.items():
    preds[n] = (t >= u.calib(ytr, o, False)).astype(int)
fo, ft = um.fuse(ch, sel)
preds["UASE-v2 (proposed)"] = (ft >= u.calib(ytr, fo, False)).astype(int)
lr = um.legacy_rule(it, False)
preds["UASE-v1 rule (LOF-first)"] = (lr[1] >= u.calib(ytr, lr[0], False)).astype(int)
n = len(yte); B = 60; R = 2000
rng = np.random.default_rng(0)
nb = int(np.ceil(n / B))
idxs = []
for _ in range(R):
    st = rng.integers(0, n - B, size=nb)
    idxs.append(np.concatenate([np.arange(s, s + B) for s in st])[:n])
ref = preds["UASE-v2 (proposed)"]
rows = []
for m, p in preds.items():
    f1s, mccs, dm = [], [], []
    for ix in idxs:
        yt = yte[ix]
        if yt.sum() == 0:
            continue
        f1s.append(f1_score(yt, p[ix], zero_division=0)); mc = matthews_corrcoef(yt, p[ix]); mccs.append(mc)
        dm.append(matthews_corrcoef(yt, ref[ix]) - mc)
    e_ref = ref != yte; e_m = p != yte
    b = int((e_ref & ~e_m).sum()); c = int((~e_ref & e_m).sum())
    pm = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    rows.append({"model": m, "f1": f1_score(yte, p), "mcc": matthews_corrcoef(yte, p),
                 "f1_ci_lo": np.percentile(f1s, 2.5), "f1_ci_hi": np.percentile(f1s, 97.5),
                 "mcc_ci_lo": np.percentile(mccs, 2.5), "mcc_ci_hi": np.percentile(mccs, 97.5),
                 "dmcc_uase_minus_model": float(np.mean(dm)), "dmcc_ci_lo": np.percentile(dm, 2.5),
                 "dmcc_ci_hi": np.percentile(dm, 97.5), "mcnemar_uase_only_err": b,
                 "mcnemar_model_only_err": c, "mcnemar_p": pm,
                 "tp": int(((p == 1) & (yte == 1)).sum()), "fp": int(((p == 1) & (yte == 0)).sum()),
                 "fn": int(((p == 0) & (yte == 1)).sum())})
df = pd.DataFrame(rows).sort_values("mcc", ascending=False)
out = Path(__file__).resolve().parents[1] / "data" / "v2" / "stats"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / f"k8s_{norm}_bootstrap.csv", index=False)
print(df.round(4).to_string(index=False))
