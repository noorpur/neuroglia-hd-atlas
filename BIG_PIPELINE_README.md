# NeuroGlia-HD Atlas: Big Research-Grade Pipeline

This zip fixes the earlier pipeline issues and adds a conservative large-dataset workflow. It does **not** pretend the large run happened unless the downloaded human glial files actually convert into the atlas matrix.

## What was fixed

- Added `tabulate>=0.9` so Markdown report generation does not crash.
- Added robust conversion of downloaded GEO files into pseudobulk matrices.
- Added support for GSE281069 10x `.h5` files, not only loose `matrix.mtx` folders.
- Added a fallback R/Seurat inspector and pseudobulk exporter for the large `GSE281069_...rds.gz` object.
- Added conservative supervised filtering so single-class datasets do not create dataset-label leakage.
- Added nested train/tune/test evaluation, Optuna tuning, permutation-label controls, overfitting diagnostics, calibration curves, leakage screens, PCA/atlas figures, and a final report.

## Install

```bash
cd ~/Downloads/neuroglia-hd-atlas
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e '.[full]'
pytest -q
```

## Download public data

Skip this if you already downloaded data into `data/raw`.

```bash
source .venv/bin/activate
neurogliahd download --config configs/default.yaml
```

## Run the big pipeline

```bash
source .venv/bin/activate
caffeinate -dims bash scripts/run_big_research_grade_pipeline.sh
```

Optional controls:

```bash
MAX_FEATURES=8000 \
MAX_CELLS_PER_H5=250000 \
MIN_CELLS_PER_PSEUDOBULK=30 \
N_TRIALS=100 \
N_PERMUTATIONS=100 \
caffeinate -dims bash scripts/run_big_research_grade_pipeline.sh
```

## Main outputs

```text
reports/research_grade_big_report.md
reports/tables/atlas_dataset_audit.csv
reports/tables/supervised_dataset_audit.csv
reports/tables/nested_cv_summary.csv
reports/tables/nested_cv_metrics.csv
reports/tables/optuna_trials.csv
reports/tables/permutation_label_control.csv
reports/tables/single_feature_leakage_screen.csv
reports/tables/dataset_label_proxy_screen.csv
reports/figures/atlas_dataset_composition.png
reports/figures/atlas_pca_condition.png
reports/figures/atlas_pca_dataset.png
reports/figures/nested_cv_roc_curves.png
reports/figures/nested_cv_precision_recall_curves.png
reports/figures/calibration_curves.png
reports/figures/overfitting_auc_gap_labelled.png
reports/figures/permutation_label_control.png
reports/figures/single_feature_leakage_screen.png
models/*_final.joblib
```

## How to judge whether the run is truly big

Open:

```bash
cat reports/tables/atlas_dataset_audit.csv
cat reports/tables/supervised_dataset_audit.csv
```

A true multi-dataset atlas run should show more than one dataset in `atlas_dataset_audit.csv`. If the supervised audit only contains GSE64810, that means the glial atlas is included only in unsupervised/atlas outputs or was excluded because it had only one supervised condition. That is intentional; it prevents a fake disease classifier caused by dataset-label confounding.

## R/Seurat fallback for GSE281069

The Python big pipeline reads the extracted 10x `.h5` files. If you want to inspect the large Seurat `.rds.gz` object or export annotation-aware pseudobulks, use:

```bash
conda create -n neuroglia-r -c conda-forge r-base r-seurat r-data.table r-matrix -y
conda activate neuroglia-r
Rscript scripts/inspect_gse281069_seurat.R
Rscript scripts/pseudobulk_gse281069_seurat.R
```

If the Seurat converter cannot infer metadata columns, open:

```bash
open data/interim/gse281069_pseudobulk/seurat_object_summary.txt
open data/interim/gse281069_pseudobulk/metadata_columns.csv
```

Then rerun with explicit column names, for example:

```bash
GSE281069_CONDITION_COL="diagnosis" \
GSE281069_DONOR_COL="donor" \
GSE281069_SAMPLE_COL="sample" \
GSE281069_CELLTYPE_COL="cell_type" \
GSE281069_REGION_COL="region" \
Rscript scripts/pseudobulk_gse281069_seurat.R
```

## Scientific guardrail

Near-perfect AUC is treated as an audit trigger, not a victory badge. Review nested CV, permutation controls, calibration, single-feature screens, and metadata proxy screens before interpreting any model as biological signal.
