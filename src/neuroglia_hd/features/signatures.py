from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_GENE_SETS: dict[str, list[str]] = {
    "DNA_REPAIR_CAG_MODIFIERS": ["FAN1", "MLH1", "MSH3", "PMS1", "PMS2", "LIG1", "MSH2", "MSH6", "EXO1"],
    "MICROGLIA_HOMEOSTASIS": ["P2RY12", "TMEM119", "CX3CR1", "CSF1R", "TREM2", "AIF1"],
    "MICROGLIA_ACTIVATION_INFLAMMATION": ["IL1B", "TNF", "IL6", "NLRP3", "CCL2", "CXCL10", "IRF7", "ISG15"],
    "ASTROCYTE_REACTIVITY": ["GFAP", "AQP4", "ALDH1L1", "SLC1A2", "SLC1A3", "VIM", "C3", "CLU"],
    "OLIGODENDROCYTE_MATURATION": ["PDGFRA", "CSPG4", "OLIG1", "OLIG2", "SOX10", "MBP", "MOG", "MAG", "PLP1", "MOBP"],
}


def load_gene_sets(path: str | Path | None = None) -> dict[str, list[str]]:
    if path is None:
        return DEFAULT_GENE_SETS
    p = Path(path)
    if not p.exists():
        return DEFAULT_GENE_SETS
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    gene_sets: dict[str, list[str]] = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            genes = value.get("genes", [])
        else:
            genes = value
        gene_sets[name] = [str(g).upper() for g in genes]
    return gene_sets


def _standardise_gene_columns(expr: pd.DataFrame) -> pd.DataFrame:
    out = expr.copy()
    out.columns = [str(c).upper() for c in out.columns]
    return out


def zscore_by_gene(expr: pd.DataFrame) -> pd.DataFrame:
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0)
    mean = expr.mean(axis=0)
    std = expr.std(axis=0).replace(0, np.nan)
    z = (expr - mean) / std
    return z.fillna(0)


def score_signatures(expr: pd.DataFrame, gene_sets: Mapping[str, list[str]]) -> pd.DataFrame:
    """Compute mean z-score signature features for a sample × gene matrix."""
    expr_std = _standardise_gene_columns(expr)
    z = zscore_by_gene(expr_std)
    scores = pd.DataFrame(index=expr.index)
    coverage_rows = []
    for name, genes in gene_sets.items():
        genes_u = [g.upper() for g in genes]
        present = [g for g in genes_u if g in z.columns]
        if present:
            scores[name] = z[present].mean(axis=1)
        else:
            scores[name] = 0.0
        coverage_rows.append({"signature": name, "n_genes": len(genes_u), "n_present": len(present), "coverage": len(present) / max(len(genes_u), 1)})
    scores.attrs["coverage"] = pd.DataFrame(coverage_rows)
    return scores


def select_top_variable_genes(expr: pd.DataFrame, n: int = 2000) -> list[str]:
    numeric = expr.apply(pd.to_numeric, errors="coerce").fillna(0)
    variances = numeric.var(axis=0).sort_values(ascending=False)
    return list(variances.head(min(n, len(variances))).index)
