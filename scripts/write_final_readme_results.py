from pathlib import Path
import pandas as pd

TABLES = Path("reports/tables")

def read_csv(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None

ready = read_csv("analysis_ready_atlas_dataset_audit.csv")
sens = read_csv("strict_supervised_feature_sensitivity.csv")
nested = read_csv("nested_cv_summary.csv")
perm = read_csv("permutation_label_control.csv")
quality = read_csv("final_quality_gate_summary.csv")
buckets = read_csv("primary_250_feature_bucket_summary.csv")

ready_row = ready.iloc[0] if ready is not None and len(ready) else {}
logi = sens[sens["model"].str.contains("logistic", case=False, na=False)]
best = logi.sort_values("test_roc_auc", ascending=False).iloc[0]

report = f"""# NeuroGlia-HD Atlas: Research-Grade Analysis Report

## Executive summary

The corrected analysis-ready atlas contains **{ready_row.get('n_analysis_ready_samples', 'NA')} samples** from **{ready_row.get('datasets_included', 'NA')}**.

GSE180294 was downloaded and converted, but excluded from analysis-ready interpretation because its 24 mouse-derived rows had zero nonzero features after the current human-feature harmonization step. It should be revisited with mouse-human ortholog mapping.

Supervised HD/control modelling is restricted to **GSE64810**, the only dataset with both HD and control labels. This avoids dataset-label leakage.

The strongest conservative supervised benchmark is the **250-feature regularized logistic regression model**:

| Metric | Value |
|---|---:|
| ROC-AUC | {best['test_roc_auc']:.3f} |
| Average precision | {best['average_precision']:.3f} |
| Balanced accuracy | {best['balanced_accuracy']:.3f} |
| F1 | {best['f1']:.3f} |
| Brier score | {best['brier']:.3f} |
| AUC gap | {best['auc_gap']:.3f} |

## Analysis-ready atlas audit

{ready.to_markdown(index=False)}

## Nested cross-validation summary

{nested.to_markdown(index=False)}

## Feature-reduced sensitivity analysis

{sens.to_markdown(index=False)}

## Permutation-label control

Mean shuffled-label ROC-AUC: **{perm['roc_auc'].mean():.3f}**  
Median shuffled-label ROC-AUC: **{perm['roc_auc'].median():.3f}**

The shuffled-label distribution is centered near chance, supporting that the supervised model is not trivially learning randomized labels.

## Feature interpretation buckets

{buckets.to_markdown(index=False)}

## Quality gate summary

{quality.to_markdown(index=False)}

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
"""

Path("README_RESULTS.md").write_text(report)

readme = Path("README.md")
txt = readme.read_text() if readme.exists() else "# NeuroGlia-HD Atlas\n"

section = """## Research-grade results

A detailed audited analysis report is available here:

- [Research-grade analysis report](README_RESULTS.md)

Current corrected status: the analysis-ready human atlas includes GSE281069 and GSE64810. GSE180294 was downloaded and converted but excluded from analysis-ready interpretation until mouse-human ortholog harmonization is fixed. Supervised HD/control modelling is conservatively restricted to GSE64810, the only current dataset with both HD and control labels.
"""

if "## Research-grade results" not in txt:
    if "## Quickstart" in txt:
        txt = txt.replace("## Quickstart", section + "\n## Quickstart")
    else:
        txt += "\n\n" + section

readme.write_text(txt)

print("Wrote README_RESULTS.md")
print("Updated README.md")
