from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class PseudobulkResult:
    expression: pd.DataFrame
    metadata: pd.DataFrame
    counts: pd.DataFrame


def pseudobulk_from_frame(
    expr: pd.DataFrame,
    metadata: pd.DataFrame,
    group_fields: Sequence[str],
    min_cells: int = 25,
    agg: str = "mean",
) -> PseudobulkResult:
    """Aggregate a cell × gene DataFrame into group-level pseudobulk expression."""
    missing = [c for c in group_fields if c not in metadata.columns]
    if missing:
        raise KeyError(f"Metadata is missing required group fields: {missing}")
    if not expr.index.equals(metadata.index):
        metadata = metadata.reindex(expr.index)
    group_key = metadata[list(group_fields)].astype(str).agg("|".join, axis=1)
    cell_counts = group_key.value_counts().rename("n_cells")
    valid_groups = cell_counts[cell_counts >= min_cells].index
    mask = group_key.isin(valid_groups)
    expr_valid = expr.loc[mask]
    meta_valid = metadata.loc[mask]
    group_key_valid = group_key.loc[mask]
    if agg == "sum":
        grouped = expr_valid.groupby(group_key_valid).sum()
    else:
        grouped = expr_valid.groupby(group_key_valid).mean()
    meta_rows = (
        meta_valid.assign(_group=group_key_valid)
        .groupby("_group")[list(group_fields)]
        .first()
        .reset_index()
        .rename(columns={"_group": "pseudobulk_id"})
    )
    counts = cell_counts.loc[valid_groups].reset_index()
    counts.columns = ["pseudobulk_id", "n_cells"]
    grouped.index.name = "pseudobulk_id"
    return PseudobulkResult(expression=grouped, metadata=meta_rows, counts=counts)


def pseudobulk_from_anndata(
    adata,
    group_fields: Sequence[str] = ("dataset_id", "donor_id", "brain_region", "cell_type", "condition"),
    min_cells: int = 25,
    agg: str = "mean",
) -> PseudobulkResult:
    """Aggregate AnnData into pseudobulk features."""
    x = adata.X
    if sparse.issparse(x):
        expr = pd.DataFrame.sparse.from_spmatrix(x, index=adata.obs_names, columns=adata.var_names)
    else:
        expr = pd.DataFrame(x, index=adata.obs_names, columns=adata.var_names)
    return pseudobulk_from_frame(expr, adata.obs.copy(), group_fields=group_fields, min_cells=min_cells, agg=agg)


def attach_signature_features(expression: pd.DataFrame, signature_scores: pd.DataFrame) -> pd.DataFrame:
    """Join expression and signature scores with safe prefixes."""
    expr = expression.copy()
    expr.columns = [f"gene::{c}" for c in expr.columns]
    sig = signature_scores.copy()
    sig.columns = [f"signature::{c}" for c in sig.columns]
    return expr.join(sig, how="left")
