from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def expression_qc_table(expr: pd.DataFrame, sample_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return sample-level QC metrics for a sample × gene expression matrix."""
    if expr.empty:
        raise ValueError("Expression matrix is empty.")
    numeric = expr.apply(pd.to_numeric, errors="coerce").fillna(0)
    qc = pd.DataFrame(index=numeric.index)
    qc["n_detected_genes"] = (numeric > 0).sum(axis=1)
    qc["library_size"] = numeric.sum(axis=1)
    qc["zero_fraction"] = (numeric == 0).mean(axis=1)
    qc["mean_expression"] = numeric.mean(axis=1)
    qc["std_expression"] = numeric.std(axis=1)
    if sample_metadata is not None:
        qc = qc.join(sample_metadata, how="left")
    return qc.reset_index(names="sample_id")


def anndata_qc_table(adata, mito_prefix: tuple[str, ...] = ("MT-", "mt-")) -> pd.DataFrame:
    """Compute lightweight AnnData QC without requiring scanpy preprocessing."""
    x = adata.X
    if sparse.issparse(x):
        lib = np.asarray(x.sum(axis=1)).ravel()
        detected = np.asarray((x > 0).sum(axis=1)).ravel()
    else:
        lib = np.asarray(x).sum(axis=1)
        detected = (np.asarray(x) > 0).sum(axis=1)
    var_names = np.asarray(adata.var_names.astype(str))
    mito_mask = np.array([g.startswith(mito_prefix) for g in var_names])
    if mito_mask.any():
        if sparse.issparse(x):
            mito = np.asarray(x[:, mito_mask].sum(axis=1)).ravel()
        else:
            mito = np.asarray(x)[:, mito_mask].sum(axis=1)
        mito_pct = 100 * mito / np.maximum(lib, 1)
    else:
        mito_pct = np.zeros_like(lib, dtype=float)
    qc = adata.obs.copy()
    qc["n_detected_genes"] = detected
    qc["library_size"] = lib
    qc["mito_pct"] = mito_pct
    qc.insert(0, "cell_id", adata.obs_names.astype(str))
    return qc
