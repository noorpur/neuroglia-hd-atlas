#!/usr/bin/env bash
set -euo pipefail

# Research-grade big run for NeuroGlia-HD Atlas.
# Run from the repository root after installing with: pip install -e '.[full]'
# If raw GEO files are already downloaded, this will NOT re-download them.

mkdir -p logs reports/tables reports/figures models data/interim data/processed

python scripts/run_research_grade_big_pipeline.py \
  --extract \
  --max-features "${MAX_FEATURES:-8000}" \
  --max-cells-per-h5 "${MAX_CELLS_PER_H5:-250000}" \
  --min-cells-per-pseudobulk "${MIN_CELLS_PER_PSEUDOBULK:-30}" \
  --n-trials "${N_TRIALS:-100}" \
  --n-permutations "${N_PERMUTATIONS:-100}" \
  2>&1 | tee logs/research_grade_big_pipeline.log

echo
echo "Main report: reports/research_grade_big_report.md"
echo "Dataset audit: reports/tables/atlas_dataset_audit.csv"
echo "Supervised audit: reports/tables/supervised_dataset_audit.csv"
