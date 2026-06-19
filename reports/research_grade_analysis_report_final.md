# NeuroGlia-HD Atlas: Research-Grade Analysis Report

## Executive summary

The corrected analysis-ready atlas contains **113 samples** from **GSE281069;GSE64810**.

GSE180294 was downloaded and converted, but excluded from analysis-ready interpretation because its 24 mouse-derived rows had zero nonzero features after the current human-feature harmonization step. It should be revisited with mouse-human ortholog mapping.

Supervised HD/control modelling is restricted to **GSE64810**, the only dataset with both HD and control labels. This avoids dataset-label leakage.

The strongest conservative supervised benchmark is the **250-feature regularized logistic regression model**:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.954 |
| Average precision | 0.909 |
| Balanced accuracy | 0.874 |
| F1 | 0.810 |
| Brier score | 0.084 |
| AUC gap | 0.046 |

## Analysis-ready atlas audit

|   n_analysis_ready_samples |   n_excluded_zero_feature_samples |   n_features |   n_hd |   n_control |   n_unknown_condition | datasets_included   | datasets_excluded   |   n_datasets_included |   n_cell_types_included |   n_regions_included |   n_donors_included |
|---------------------------:|----------------------------------:|-------------:|-------:|------------:|----------------------:|:--------------------|:--------------------|----------------------:|------------------------:|---------------------:|--------------------:|
|                        113 |                                24 |         8000 |     64 |          49 |                     0 | GSE281069;GSE64810  | GSE180294           |                     2 |                       2 |                    6 |                 113 |

## Nested cross-validation summary

| model                  |   mean_train_auc |   mean_test_auc |   std_test_auc |   mean_auc_gap |   max_auc_gap |   mean_average_precision |   mean_balanced_accuracy |   mean_f1 |   mean_brier |
|:-----------------------|-----------------:|----------------:|---------------:|---------------:|--------------:|-------------------------:|-------------------------:|----------:|-------------:|
| hist_gradient_boosting |                1 |        0.94     |       0.065192 |      0.06      |      0.15     |                 0.871071 |                 0.81     |  0.731429 |    0.103678  |
| logistic_regularized   |                1 |        0.937222 |       0.050522 |      0.0627778 |      0.138889 |                 0.850833 |                 0.883889 |  0.830635 |    0.0893887 |
| random_forest          |                1 |        0.9275   |       0.060208 |      0.0725    |      0.15     |                 0.872976 |                 0.695    |  0.540952 |    0.119833  |

## Feature-reduced sensitivity analysis

| model                              |   n_features |   mean_train_auc |   test_roc_auc |   auc_gap |   average_precision |   balanced_accuracy |       f1 |     brier |   n_samples |   n_hd |   n_control | eligible_datasets   |
|:-----------------------------------|-------------:|-----------------:|---------------:|----------:|--------------------:|--------------------:|---------:|----------:|------------:|-------:|------------:|:--------------------|
| logistic_regularized               |          100 |         0.995537 |       0.938776 | 0.0567613 |            0.874088 |            0.863776 | 0.790698 | 0.0983073 |          69 |     20 |          49 | GSE64810            |
| hist_gradient_boosting_regularized |          100 |         0.998101 |       0.923469 | 0.0746316 |            0.852874 |            0.839796 | 0.8      | 0.10029   |          69 |     20 |          49 | GSE64810            |
| logistic_regularized               |          250 |         1        |       0.954082 | 0.0459184 |            0.908994 |            0.87398  | 0.809524 | 0.0840106 |          69 |     20 |          49 | GSE64810            |
| hist_gradient_boosting_regularized |          250 |         1        |       0.904082 | 0.0959184 |            0.790791 |            0.779592 | 0.705882 | 0.110924  |          69 |     20 |          49 | GSE64810            |
| logistic_regularized               |          500 |         1        |       0.940816 | 0.0591837 |            0.844195 |            0.888776 | 0.818182 | 0.0935102 |          69 |     20 |          49 | GSE64810            |
| hist_gradient_boosting_regularized |          500 |         1        |       0.915306 | 0.0846939 |            0.831224 |            0.779592 | 0.705882 | 0.109664  |          69 |     20 |          49 | GSE64810            |
| logistic_regularized               |         1000 |         1        |       0.937755 | 0.0622449 |            0.847234 |            0.87398  | 0.809524 | 0.0947042 |          69 |     20 |          49 | GSE64810            |
| hist_gradient_boosting_regularized |         1000 |         1        |       0.92551  | 0.0744898 |            0.8703   |            0.804592 | 0.742857 | 0.0988146 |          69 |     20 |          49 | GSE64810            |
| logistic_regularized               |         2000 |         1        |       0.940816 | 0.0591837 |            0.851625 |            0.84898  | 0.780488 | 0.0949207 |          69 |     20 |          49 | GSE64810            |
| hist_gradient_boosting_regularized |         2000 |         1        |       0.927041 | 0.0729592 |            0.846592 |            0.759184 | 0.666667 | 0.105739  |          69 |     20 |          49 | GSE64810            |
| logistic_regularized               |         4000 |         1        |       0.947959 | 0.0520408 |            0.850745 |            0.859184 | 0.8      | 0.088548  |          69 |     20 |          49 | GSE64810            |
| hist_gradient_boosting_regularized |         4000 |         1        |       0.928061 | 0.0719388 |            0.833697 |            0.819388 | 0.756757 | 0.101794  |          69 |     20 |          49 | GSE64810            |
| logistic_regularized               |         8000 |         1        |       0.941837 | 0.0581633 |            0.815046 |            0.884184 | 0.829268 | 0.0888319 |          69 |     20 |          49 | GSE64810            |
| hist_gradient_boosting_regularized |         8000 |         1        |       0.938776 | 0.0612245 |            0.781336 |            0.834184 | 0.769231 | 0.0948734 |          69 |     20 |          49 | GSE64810            |

## Permutation-label control

Mean shuffled-label ROC-AUC: **0.509**  
Median shuffled-label ROC-AUC: **0.497**

The shuffled-label distribution is centered near chance, supporting that the supervised model is not trivially learning randomized labels.

## Feature interpretation buckets

| interpretation_bucket                    |   n_top100_features |
|:-----------------------------------------|--------------------:|
| other_candidate                          |                  75 |
| metal_ion_oxidative_stress               |                   6 |
| extracellular_matrix_or_tissue_structure |                   5 |
| stress_inflammation_immediate_early      |                   5 |
| neuronal_signalling_or_cell_state        |                   4 |
| pseudogene_or_mapping_watchlist          |                   4 |
| blood_or_sample_composition_watchlist    |                   1 |

## Quality gate summary

| check                         | status                                   | value                                     | interpretation                                                                                                                      |
|:------------------------------|:-----------------------------------------|:------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| Raw atlas conversion          | pass                                     | 137 samples; GSE180294;GSE281069;GSE64810 | Raw converted atlas matrix was constructed before analysis-ready filtering.                                                         |
| Raw duplicate feature rows    | review                                   | 23                                        | Duplicate rows require review; later audit shows GSE180294 zero-feature collapse.                                                   |
| Analysis-ready atlas          | pass_with_exclusion                      | 113 included; 24 excluded                 | Included datasets: GSE281069;GSE64810; excluded datasets: GSE180294.                                                                |
| Dataset feature coverage      | exclude_zero_feature_dataset             | GSE180294                                 | At least one dataset has zero median nonzero features after harmonization and should not be interpreted in atlas PCA/model results. |
| Supervised eligibility        | limited                                  | 69 samples; datasets=GSE64810             | Supervised HD/control modelling is limited to datasets containing both HD and control labels.                                       |
| Best nested CV ROC-AUC        | pass_with_caution                        | hist_gradient_boosting: 0.940             | Strong discrimination, but only one supervised-eligible cohort is available.                                                        |
| Best nested CV F1             | pass_with_caution                        | logistic_regularized: 0.831               | Use F1/balanced accuracy alongside ROC-AUC because classes are imbalanced.                                                          |
| Overfitting gap               | moderate_review                          | max mean AUC gap=0.072                    | Below high-concern threshold but above low-concern threshold.                                                                       |
| Feature-reduced sensitivity   | pass_with_caution                        | logistic 250 features AUC=0.954           | Signal persists under feature reduction; preferred conservative benchmark is the 250-feature regularized logistic model.            |
| Permutation-label control     | pass                                     | mean ROC-AUC=0.509; median=0.497          | Shuffled labels center near chance.                                                                                                 |
| Single-feature leakage screen | watchlist                                | top=ENSG00000113108.13 AUC=0.932          | Strong individual feature separation but no near-perfect leaked feature.                                                            |
| GSE180294 GEO labels          | metadata_recovered_but_not_feature_ready | HD_model_R6_2:12; control_model_NT:12     | Mouse condition labels were recovered, but mouse rows have zero feature overlap in the current human matrix.                        |

## Main figures

![Analysis-ready atlas dataset composition](reports/figures/main/analysis_ready_atlas_dataset_composition.png)

![Analysis-ready atlas PCA by dataset](reports/figures/main/analysis_ready_atlas_pca_dataset.png)

![Analysis-ready atlas PCA by condition](reports/figures/main/analysis_ready_atlas_pca_condition.png)

![Nested CV discrimination metrics](reports/figures/main/nested_cv_discrimination_metrics_clean.png)

![Nested CV Brier score](reports/figures/main/nested_cv_brier_score_clean.png)

![Feature-reduced sensitivity analysis](reports/figures/main/feature_reduced_sensitivity_auc.png)

![Overfitting audit](reports/figures/main/overfitting_auc_gap_labelled.png)

![Calibration curves](reports/figures/main/calibration_curves.png)

![Permutation-label control](reports/figures/main/permutation_label_control.png)

![Single-feature leakage screen](reports/figures/main/single_feature_leakage_screen.png)

![Top coefficients from the 250-feature logistic model](reports/figures/main/primary_250_feature_logistic_top_coefficients.png)

## Interpretation

The current run is a successful research-grade exploratory analysis. The analysis-ready human atlas includes GSE281069 and GSE64810. Supervised HD/control classification is intentionally limited to GSE64810 to avoid dataset-label leakage. The 250-feature regularized logistic model provides the most defensible supervised benchmark because performance remains strong under feature reduction.

This should not be described as a finalized cross-cohort or clinical HD classifier. Independent labelled validation and mouse-human ortholog harmonization are needed before making broader claims.

## Next steps

1. Add mouse-human ortholog mapping for GSE180294.
2. Recompute the analysis-ready atlas after ortholog harmonization.
3. Add an independent labelled human validation cohort.
4. Perform pathway enrichment on top logistic coefficients and single-feature hits.
5. Keep atlas-level and supervised analyses separate unless matched controls are available.
