# Validation Plan

## Leakage controls

- split by donor, not cell or pseudobulk row;
- confirm no donor IDs appear in both train and test folds;
- prevent metadata fields that directly encode diagnosis from entering the feature matrix;
- run null-label controls.

## Statistical controls

- repeated grouped CV;
- bootstrap confidence intervals for metrics;
- permutation importance stability across folds;
- paired ablation comparisons;
- calibration diagnostics.

## Biological controls

- cell-type marker sanity checks;
- signature coverage reporting;
- region-held-out transfer;
- independent bulk replication;
- cross-species comparison as stress test, not direct substitution.

## Readiness ladder

1. **Computational reproducibility:** deterministic pipeline, versioned configs, audit reports.
2. **Internal validity:** grouped CV, null checks, calibration.
3. **External validity:** independent dataset replication.
4. **Mechanistic plausibility:** feature/pathway consistency with known biology.
5. **Experimental follow-up:** targeted validation in cellular or tissue models.
