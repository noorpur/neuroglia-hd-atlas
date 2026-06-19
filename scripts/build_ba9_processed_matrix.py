from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

matches = list(RAW.rglob("GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz"))
if not matches:
    raise FileNotFoundError("Could not find GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz under data/raw")

path = matches[0]
print(f"Reading {path}")

df = pd.read_csv(path, sep="\t", compression="gzip")

# Detect gene column.
first_col = df.columns[0]
if first_col.lower() in {"gene", "genes", "symbol", "gene_symbol", "id", "x"} or not re.match(r"^[CH]_", str(first_col)):
    df = df.set_index(first_col)
else:
    df.index = df.iloc[:, 0]
    df = df.drop(columns=[first_col])

df.index = df.index.astype(str)

# Keep sample columns only. In this dataset C_* = control and H_* = HD.
sample_cols = [c for c in df.columns if re.match(r"^[CH]_\d+", str(c))]
if len(sample_cols) < 10:
    raise ValueError(f"Too few sample columns detected: {sample_cols[:10]}")

expr = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).T

# Log transform if values look count-like.
if float(expr.max().max()) > 50:
    expr = np.log1p(expr)

expr.index.name = "pseudobulk_id"

meta = pd.DataFrame({
    "pseudobulk_id": expr.index,
    "sample_id": expr.index,
    "dataset_id": "human_bulk_ba9",
    "donor_id": expr.index,
    "brain_region": "BA9",
    "cell_type": "bulk_BA9",
    "condition": ["HD" if s.startswith("H_") else "control" for s in expr.index],
    "n_cells": 1,
})

expr.to_parquet(PROCESSED / "pseudobulk_expression.parquet")
meta.to_parquet(PROCESSED / "pseudobulk_metadata.parquet", index=False)

print("Wrote:")
print("  data/processed/pseudobulk_expression.parquet", expr.shape)
print("  data/processed/pseudobulk_metadata.parquet", meta.shape)
print(meta["condition"].value_counts())
