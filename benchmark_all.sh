#!/usr/bin/env bash
set -euo pipefail

MODELS="${MODELS:-resnet50}"
NS="${NS:-aimodel}"
PROTO="${PROTO:-rest}"
REQS="${REQS:-400}"
CONC="${CONC:-8}"
BATCH="${BATCH:-1}"

for M in $MODELS; do
  APP_LABEL="${M}-triton"
  SVC="${M}-triton"

  OUTDIR="scripts/runs/${M}"
  mkdir -p "$OUTDIR"

  echo "=== Benchmark model: ${M} ==="
  NS="$NS" APP_LABEL="$APP_LABEL" SVC="$SVC" MODEL="$M" \
  PROTO="$PROTO" REQS="$REQS" CONC="$CONC" BATCH="$BATCH" \
  OUTDIR="$OUTDIR" \
  scripts/run_and_collect.sh
done
