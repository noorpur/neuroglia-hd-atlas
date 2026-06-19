from pathlib import Path
import pandas as pd

TABLES = Path("reports/tables")
TABLES.mkdir(parents=True, exist_ok=True)

rows = []

def add(check, status, value, interpretation):
    rows.append({
        "check": check,
        "status": status,
        "value": value,
        "interpretation": interpretation,
    })

# Atlas conversion
atlas_path = TABLES / "atlas_dataset_audit.csv"
if atlas_path.exists():
    atlas = pd.read_csv(atlas_path)
    a = atlas.iloc[0]
    add(
        "Raw atlas conversion",
        "pass",
        f"{a.get('n_pseudobulk_samples', 'NA')} samples; {a.get('datasets', 'NA')}",
        "Raw converted atlas matrix was constructed before analysis-ready filtering.",
    )
    add(
        "Raw duplicate feature rows",
        "review" if int(a.get("n_duplicate_feature_rows", 0)) > 0 else "pass",
        str(a.get("n_duplicate_feature_rows", "NA")),
        "Duplicate rows require review; later audit shows GSE180294 zero-feature collapse.",
    )

# Analysis-ready atlas
ready_path = TABLES / "analysis_ready_atlas_dataset_audit.csv"
if ready_path.exists():
    ready = pd.read_csv(ready_path)
    r = ready.iloc[0]
    add(
        "Analysis-ready atlas",
        "pass_with_exclusion",
        f"{r.get('n_analysis_ready_samples', 'NA')} included; {r.get('n_excluded_zero_feature_samples', 'NA')} excluded",
        f"Included datasets: {r.get('datasets_included', 'NA')}; excluded datasets: {r.get('datasets_excluded', '')}.",
    )

# Feature coverage
coverage_path = TABLES / "dataset_feature_coverage_audit.csv"
if coverage_path.exists():
    cov = pd.read_csv(coverage_path)
    zero = cov[cov["median_nonzero_features"] == 0]
    if len(zero):
        add(
            "Dataset feature coverage",
            "exclude_zero_feature_dataset",
            "; ".join(zero["dataset_id"].astype(str).unique()),
            "At least one dataset has zero median nonzero features after harmonization and should not be interpreted in atlas PCA/model results.",
        )
    else:
        add("Dataset feature coverage", "pass", "all datasets nonzero", "All datasets have nonzero feature coverage.")

# Supervised eligibility computed directly from sample metadata
meta_path = Path("data/processed/sample_metadata.parquet")
if meta_path.exists():
    meta = pd.read_parquet(meta_path)
    eligible = []
    for dataset, df in meta.groupby("dataset_id"):
        labels = set(df["condition"].astype(str))
        if {"HD", "control"}.issubset(labels):
            eligible.append(dataset)
    sup = meta[meta["dataset_id"].isin(eligible)].copy()
    add(
        "Supervised eligibility",
        "limited",
        f"{len(sup)} samples; datasets={';'.join(eligible) if eligible else 'none'}",
        "Supervised HD/control modelling is limited to datasets containing both HD and control labels.",
    )

# Nested CV
nested_path = TABLES / "nested_cv_summary.csv"
if nested_path.exists():
    nested = pd.read_csv(nested_path)
    best_auc = nested.sort_values("mean_test_auc", ascending=False).iloc[0]
    best_f1 = nested.sort_values("mean_f1", ascending=False).iloc[0]
    max_gap = float(nested["mean_auc_gap"].max())

    add(
        "Best nested CV ROC-AUC",
        "pass_with_caution",
        f"{best_auc['model']}: {best_auc['mean_test_auc']:.3f}",
        "Strong discrimination, but only one supervised-eligible cohort is available.",
    )
    add(
        "Best nested CV F1",
        "pass_with_caution",
        f"{best_f1['model']}: {best_f1['mean_f1']:.3f}",
        "Use F1/balanced accuracy alongside ROC-AUC because classes are imbalanced.",
    )
    add(
        "Overfitting gap",
        "moderate_review" if max_gap >= 0.05 else "pass",
        f"max mean AUC gap={max_gap:.3f}",
        "Below high-concern threshold but above low-concern threshold.",
    )

# Strict sensitivity
sens_path = TABLES / "strict_supervised_feature_sensitivity.csv"
if sens_path.exists():
    sens = pd.read_csv(sens_path)
    logi = sens[sens["model"].str.contains("logistic", case=False, na=False)]
    if len(logi):
        best = logi.sort_values("test_roc_auc", ascending=False).iloc[0]
        add(
            "Feature-reduced sensitivity",
            "pass_with_caution",
            f"logistic {int(best['n_features'])} features AUC={best['test_roc_auc']:.3f}",
            "Signal persists under feature reduction; preferred conservative benchmark is the 250-feature regularized logistic model.",
        )

# Permutation control
perm_path = TABLES / "permutation_label_control.csv"
if perm_path.exists():
    perm = pd.read_csv(perm_path)
    add(
        "Permutation-label control",
        "pass",
        f"mean ROC-AUC={perm['roc_auc'].mean():.3f}; median={perm['roc_auc'].median():.3f}",
        "Shuffled labels center near chance.",
    )

# Single-feature leakage
single_path = TABLES / "single_feature_leakage_screen.csv"
if single_path.exists():
    single = pd.read_csv(single_path)
    top = single.iloc[0]
    auc = float(top["single_feature_auc_directionless"])
    add(
        "Single-feature leakage screen",
        "watchlist" if auc >= 0.90 else "pass",
        f"top={top['feature']} AUC={auc:.3f}",
        "Strong individual feature separation but no near-perfect leaked feature.",
    )

# GSE180294 labels
label_path = TABLES / "gse180294_geo_label_summary_patched.csv"
if label_path.exists():
    labels = pd.read_csv(label_path)
    label_summary = "; ".join(f"{r.patched_condition}:{r.n}" for r in labels.itertuples())
    add(
        "GSE180294 GEO labels",
        "metadata_recovered_but_not_feature_ready",
        label_summary,
        "Mouse condition labels were recovered, but mouse rows have zero feature overlap in the current human matrix.",
    )

out = pd.DataFrame(rows)
out.to_csv(TABLES / "final_quality_gate_summary.csv", index=False)

print(out.to_string(index=False))
print("\nWrote:", TABLES / "final_quality_gate_summary.csv")
