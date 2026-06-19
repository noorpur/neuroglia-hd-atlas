#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default.yaml}

echo "[1/5] Dataset registry"
neurogliahd registry

echo "[2/5] Download public data"
neurogliahd download --config "$CONFIG"

echo "[3/5] Build pseudobulk/signature features"
neurogliahd pseudobulk --config "$CONFIG"

echo "[4/5] Train grouped baseline models"
neurogliahd train-baselines --config "$CONFIG"

echo "[5/5] Generate report"
neurogliahd report --config "$CONFIG"
