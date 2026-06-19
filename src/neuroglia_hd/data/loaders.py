from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def read_table_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV/TSV/TXT files with separator inference."""
    path = Path(path)
    compression = "gzip" if path.suffix == ".gz" else None
    if path.suffix.lower() in {".tsv", ".txt"} or path.name.endswith(".tsv.gz") or path.name.endswith(".txt.gz"):
        sep = "\t"
    else:
        sep = kwargs.pop("sep", None)
    return pd.read_csv(path, sep=sep, compression=compression, engine="python", **kwargs)


def find_files(root: str | Path, extensions: Iterable[str]) -> list[Path]:
    root = Path(root)
    exts = tuple(e.lower() for e in extensions)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(exts))


def read_expression_matrix(path: str | Path) -> pd.DataFrame:
    """Read a gene × sample or sample × gene expression matrix from a flat file."""
    df = read_table_auto(path, index_col=0)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all(axis=None):
        raise ValueError(f"No numeric expression values found in {path}")
    return numeric


def load_anndata_optional(path: str | Path):
    """Load AnnData/H5AD when scanpy/anndata is installed."""
    path = Path(path)
    try:
        import scanpy as sc  # type: ignore
    except Exception as exc:  # pragma: no cover - optional heavy dependency
        raise ImportError("Install the full extras to load AnnData: pip install -e '.[full]'") from exc
    return sc.read(path)


def sparse_to_frame(matrix: sparse.spmatrix | np.ndarray, obs_names: list[str], var_names: list[str]) -> pd.DataFrame:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return pd.DataFrame(matrix, index=obs_names, columns=var_names)


def detect_metadata_columns(meta: pd.DataFrame, candidates: dict[str, list[str]]) -> dict[str, str | None]:
    """Map semantic fields to likely metadata columns by case-insensitive matching."""
    lower_to_original = {c.lower(): c for c in meta.columns}
    mapping: dict[str, str | None] = {}
    for semantic, names in candidates.items():
        hit = None
        for name in names:
            if name.lower() in lower_to_original:
                hit = lower_to_original[name.lower()]
                break
        if hit is None:
            for col in meta.columns:
                col_l = col.lower()
                if any(name.lower() in col_l for name in names):
                    hit = col
                    break
        mapping[semantic] = hit
    return mapping
