from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
TABLES = Path("reports/tables")
FIGS = Path("reports/figures")
MAIN = FIGS / "main"
TABLES.mkdir(parents=True, exist_ok=True)
MAIN.mkdir(parents=True, exist_ok=True)

x = pd.read_parquet(PROCESSED / "feature_matrix.parquet")
meta = pd.read_parquet(PROCESSED / "sample_metadata.parquet").copy()

if "pseudobulk_id" in meta.columns and set(meta["pseudobulk_id"].astype(str)).issubset(set(x.index.astype(str))):
    x = x.loc[meta["pseudobulk_id"].astype(str)]
else:
    x = x.iloc[: len(meta)]

x = x.reset_index(drop=True)
meta = meta.reset_index(drop=True)

meta["n_nonzero_features"] = (x != 0).sum(axis=1).astype(int)
meta["fraction_nonzero_features"] = (x != 0).mean(axis=1)
meta["row_sum"] = x.sum(axis=1)
meta["row_variance"] = x.var(axis=1)
meta["analysis_ready"] = meta["n_nonzero_features"] > 0

excluded = meta[~meta["analysis_ready"]].copy()
included = meta[meta["analysis_ready"]].copy()
x_ready = x.loc[meta["analysis_ready"].values].copy()

excluded.to_csv(TABLES / "analysis_excluded_zero_feature_samples.csv", index=False)

audit = pd.DataFrame([{
    "n_analysis_ready_samples": len(included),
    "n_excluded_zero_feature_samples": len(excluded),
    "n_features": x_ready.shape[1],
    "n_hd": int((included["condition"].astype(str) == "HD").sum()),
    "n_control": int((included["condition"].astype(str) == "control").sum()),
    "n_unknown_condition": int((included["condition"].astype(str) == "unknown").sum()),
    "datasets_included": ";".join(sorted(included["dataset_id"].astype(str).unique())),
    "datasets_excluded": ";".join(sorted(excluded["dataset_id"].astype(str).unique())) if len(excluded) else "",
    "n_datasets_included": included["dataset_id"].astype(str).nunique(),
    "n_cell_types_included": included["cell_type"].astype(str).nunique() if "cell_type" in included else "",
    "n_regions_included": included["brain_region"].astype(str).nunique() if "brain_region" in included else "",
    "n_donors_included": included["donor_id"].astype(str).nunique() if "donor_id" in included else "",
}])

audit.to_csv(TABLES / "analysis_ready_atlas_dataset_audit.csv", index=False)

composition = (
    included.groupby(["dataset_id", "condition"], dropna=False)
    .size()
    .reset_index(name="n")
)
composition.to_csv(TABLES / "analysis_ready_atlas_dataset_condition_composition.csv", index=False)

# Dataset composition figure
pivot = composition.pivot(index="dataset_id", columns="condition", values="n").fillna(0)
ax = pivot.plot(kind="bar", figsize=(10, 6))
ax.set_title("Analysis-ready atlas dataset composition")
ax.set_ylabel("pseudobulk samples")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(MAIN / "analysis_ready_atlas_dataset_composition.png", dpi=200)
plt.close()

# PCA figures
xs = x_ready.copy()
# Drop zero-variance features before PCA.
var = xs.var(axis=0)
xs = xs.loc[:, var > 0]

scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(xs)
coords = PCA(n_components=2, random_state=42).fit_transform(scaled)

pca_df = included.copy()
pca_df["PC1"] = coords[:, 0]
pca_df["PC2"] = coords[:, 1]
pca_df.to_csv(TABLES / "analysis_ready_atlas_pca_coordinates.csv", index=False)

def scatter_by(field, out_name, title):
    plt.figure(figsize=(9, 7))
    for level, df in pca_df.groupby(field, dropna=False):
        plt.scatter(df["PC1"], df["PC2"], label=str(level), alpha=0.75)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MAIN / out_name, dpi=200)
    plt.close()

scatter_by("dataset_id", "analysis_ready_atlas_pca_dataset.png", "Analysis-ready atlas PCA by dataset")
scatter_by("condition", "analysis_ready_atlas_pca_condition.png", "Analysis-ready atlas PCA by condition")

print("=== Analysis-ready atlas audit ===")
print(audit.to_string(index=False))

print("\n=== Excluded zero-feature samples ===")
if len(excluded):
    cols = [c for c in ["pseudobulk_id", "dataset_id", "condition", "brain_region", "cell_type", "n_nonzero_features"] if c in excluded.columns]
    print(excluded[cols].to_string(index=False))
else:
    print("None")

print("\nWrote analysis-ready atlas figures to:")
print(MAIN)
