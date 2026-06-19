from pathlib import Path
import pandas as pd
import numpy as np

PROCESSED = Path("data/processed")
TABLES = Path("reports/tables")
TABLES.mkdir(parents=True, exist_ok=True)

x = pd.read_parquet(PROCESSED / "feature_matrix.parquet")
meta = pd.read_parquet(PROCESSED / "sample_metadata.parquet").copy()

if "pseudobulk_id" in meta.columns and set(meta["pseudobulk_id"].astype(str)).issubset(set(x.index.astype(str))):
    x = x.loc[meta["pseudobulk_id"].astype(str)]
else:
    x = x.iloc[: len(meta)]

x = x.copy()
meta = meta.reset_index(drop=True)
x = x.reset_index(drop=True)

nonzero = (x != 0)
meta["n_nonzero_features"] = nonzero.sum(axis=1).astype(int)
meta["fraction_nonzero_features"] = nonzero.mean(axis=1)
meta["row_sum"] = x.sum(axis=1)
meta["row_variance"] = x.var(axis=1)
meta["is_duplicate_feature_row"] = x.duplicated(keep=False)

coverage = (
    meta.groupby(["dataset_id", "condition"], dropna=False)
    .agg(
        n=("dataset_id", "size"),
        n_duplicate_rows=("is_duplicate_feature_row", "sum"),
        median_nonzero_features=("n_nonzero_features", "median"),
        min_nonzero_features=("n_nonzero_features", "min"),
        max_nonzero_features=("n_nonzero_features", "max"),
        median_fraction_nonzero=("fraction_nonzero_features", "median"),
        median_row_sum=("row_sum", "median"),
        median_row_variance=("row_variance", "median"),
    )
    .reset_index()
)

dups = meta[meta["is_duplicate_feature_row"]].copy()
dups = dups.sort_values(["dataset_id", "condition", "pseudobulk_id" if "pseudobulk_id" in dups.columns else "sample_id"])

coverage.to_csv(TABLES / "dataset_feature_coverage_audit.csv", index=False)
dups.to_csv(TABLES / "duplicate_feature_rows_detail.csv", index=False)

print("\n=== Dataset feature coverage audit ===")
print(coverage.to_string(index=False))

print("\n=== Duplicate rows by dataset ===")
if len(dups):
    print(dups.groupby(["dataset_id", "condition"], dropna=False).size().reset_index(name="n_duplicate_rows").to_string(index=False))
else:
    print("No duplicate rows detected.")

print("\nWrote:")
print(TABLES / "dataset_feature_coverage_audit.csv")
print(TABLES / "duplicate_feature_rows_detail.csv")
