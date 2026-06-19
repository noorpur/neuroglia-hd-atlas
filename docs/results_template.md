# Results Template

This file is a manuscript-style scaffold for the results that will be generated after the full pipeline runs.

## 1. Dataset audit

| Dataset | Samples/cells retained | Exclusions | Notes |
|---|---:|---:|---|
| GSE281069 | TBD | TBD | TBD |
| GSE180294 | TBD | TBD | TBD |
| GSE159940 | TBD | TBD | TBD |
| GSE64810 | TBD | TBD | TBD |

## 2. Quality control

Insert:

- `reports/figures/qc_overview.png`
- `reports/tables/qc_summary.csv`

Narrative placeholder:

> After QC, I retained [TBD] pseudobulk samples across [TBD] donors, [TBD] brain regions and [TBD] annotated cell-type groups.

## 3. Baseline model performance

Insert:

- `reports/tables/baseline_metrics.csv`
- `reports/figures/baseline_model_comparison.png`
- `reports/figures/roc_pr_curves.png`

Narrative placeholder:

> The strongest model under donor-grouped validation was [TBD], with ROC-AUC [TBD], average precision [TBD] and Brier score [TBD].

## 4. Cell-type ablation

Insert:

- `reports/tables/ablation_by_cell_type.csv`
- `reports/figures/ablation_heatmap.png`

Narrative placeholder:

> The largest decrease in performance occurred when [TBD] features were removed, suggesting [TBD].

## 5. Mechanism-aware feature importance

Insert:

- `reports/tables/top_features.csv`
- `reports/figures/top_features.png`

Narrative placeholder:

> Top-ranked features included [TBD]. These were enriched for [TBD] signatures.

## 6. Latent neuroglial state model

Insert:

- `reports/figures/latent_space.png`
- `reports/tables/latent_reconstruction_error.csv`

Narrative placeholder:

> The latent model separated [TBD] along dimension [TBD], with reconstruction errors indicating [TBD].

## 7. Cross-dataset replication

Insert:

- `reports/tables/cross_region_transfer.csv`
- `reports/tables/cross_species_transfer.csv`

Narrative placeholder:

> Cross-dataset transfer was strongest between [TBD] and weakest between [TBD], consistent with [TBD].

## 8. Interpretation

Keep this section cautious. Expression signatures can support mechanistic hypotheses, but they do not establish causality by themselves.
