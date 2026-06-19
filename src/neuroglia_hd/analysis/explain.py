from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(model, x: pd.DataFrame, y: pd.Series, n_repeats: int = 20, random_state: int = 42) -> pd.DataFrame:
    result = permutation_importance(model, x, y, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc", n_jobs=-1)
    return (
        pd.DataFrame({
            "feature": x.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def ablate_feature_prefixes(x: pd.DataFrame, prefixes: list[str]) -> dict[str, list[str]]:
    """Return feature masks for systematic feature-family ablations."""
    masks = {}
    cols = list(x.columns)
    for prefix in prefixes:
        masks[prefix] = [c for c in cols if not c.startswith(prefix)]
    return masks


def latent_shift_table(embedding: pd.DataFrame, metadata: pd.DataFrame, condition_col: str = "condition") -> pd.DataFrame:
    joined = embedding.join(metadata.set_index(embedding.index.name or metadata.index.name), how="left") if condition_col not in embedding.columns else embedding
    if condition_col not in joined.columns:
        return pd.DataFrame()
    rows = []
    for col in embedding.columns:
        groups = joined.groupby(condition_col)[col].agg(["mean", "std", "count"])
        if groups.shape[0] >= 2:
            ordered = groups.sort_index()
            diff = ordered.iloc[-1]["mean"] - ordered.iloc[0]["mean"]
            rows.append({"latent_dim": col, "condition_delta": float(diff)})
    return pd.DataFrame(rows).sort_values("condition_delta", key=lambda s: np.abs(s), ascending=False)
