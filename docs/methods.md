# Methods

## Overview

NeuroGlia-HD Atlas implements a layered modelling approach:

1. **Raw data acquisition:** download public supplementary files from GEO.
2. **Parsing:** load AnnData/H5AD/MTX/flat matrices where available.
3. **QC:** quantify library size, detected genes, mitochondrial fraction where possible, and sample/cell metadata completeness.
4. **Aggregation:** construct donor × brain-region × cell-type pseudobulk profiles to reduce cell-level pseudoreplication.
5. **Feature construction:** combine variable-gene expression features with curated neuroglial, proteostasis, metabolism and CAG-instability signatures.
6. **Validation:** perform grouped cross-validation by donor, held-out region checks, and independent replication checks.
7. **Model fitting:** compare regularised linear models, random forests, gradient boosting, calibrated ensembles, and neural latent models.
8. **Interpretability:** generate permutation importance, feature-family ablations, signature coverage reports, calibration curves, and latent-state visualisations.

## Pseudobulk design

Single nuclei are aggregated using:

```text
pseudobulk_id = dataset_id | donor_id | brain_region | cell_type | condition
```

A pseudobulk group is excluded when it contains fewer than `min_cells_per_pseudobulk` cells. The default aggregation is mean log-normalised expression.

## Signature scoring

For each curated gene set, the pipeline computes mean within-gene z-scored expression:

```text
signature_score(sample, set) = mean(z_gene_expression for genes present in set)
```

Coverage is reported because not every gene is present in every processed matrix.

## Validation strategy

The main split variable is donor ID. This prevents cells or pseudobulk rows from the same donor appearing in both train and test folds.

Additional checks:

- held-out brain-region transfer;
- cell-type-specific ablations;
- signature-family ablations;
- random-label/null runs;
- calibration and Brier score;
- independent bulk replication.

## Model families

### Classical baselines

- Elastic-net logistic regression
- Random forest
- Histogram gradient boosting

### Neural representation model

- Denoising autoencoder by default
- Config prepared for beta-VAE-style extensions
- MPS-aware device selection on Apple Silicon

## Reporting

The reporting layer writes machine-readable tables plus a markdown report. The report is intentionally conservative and separates metrics from interpretation.
