from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_metric_bar(metrics: pd.DataFrame, output: str | Path, metric: str = "roc_auc") -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics.sort_values(metric).plot.barh(x="model", y=metric, ax=ax, legend=False)
    ax.set_xlabel(metric)
    ax.set_ylabel("model")
    ax.set_title(f"Model comparison: {metric}")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def save_latent_scatter(embedding: pd.DataFrame, metadata: pd.DataFrame, output: str | Path, label_col: str = "condition") -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    joined = embedding.join(metadata, how="left")
    labels = joined[label_col].astype(str) if label_col in joined else pd.Series("unknown", index=joined.index)
    for label, frame in joined.groupby(labels):
        ax.scatter(frame.iloc[:, 0], frame.iloc[:, 1], s=24, alpha=0.75, label=label)
    ax.set_xlabel(embedding.columns[0])
    ax.set_ylabel(embedding.columns[1] if embedding.shape[1] > 1 else embedding.columns[0])
    ax.legend(title=label_col, frameon=False)
    ax.set_title("Latent neuroglial state space")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def save_top_features(features: pd.DataFrame, output: str | Path, n: int = 25) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    top = features.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * n)))
    ax.barh(top["feature"], top["importance_mean"])
    ax.set_xlabel("permutation importance")
    ax.set_title("Top model features")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output
