#!/usr/bin/env bash
# Friedman/Nemenyi + Holm-Wilcoxon statistics for the univariate corpora.
# usage: bash code/scripts/run_stats.sh <cloud|yahoo> [noresid]
# (the paper uses "noresid" for yahoo, because the selected UASE-v2 channel IS "Residual Z")
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
T=$1; EXTRA=${2:-}
MODELS=("Isolation Forest" "LOF" "One-Class SVM" "Elliptic Envelope" "Autoencoder" "ECOD" "LODA" "HBOS" "PCA+ECOD" "Rolling Z-Score" "Windowed AE" "Windowed RF" "Spectral Residual" "UASE-v2 (proposed)" "UASE-v1 rule (LOF-first)")
[ "$EXTRA" != "noresid" ] && MODELS+=("Residual Z")
for met in f1_pa mcc_pa f1 mcc; do
  $PY v2_stats.py --csv ../data/v2/meta/${T}_test_per_series.csv --ref "UASE-v2 (proposed)" --metric $met --tag ${T}_test --models "${MODELS[@]}" --title "$T (test): average rank by $met"
done
