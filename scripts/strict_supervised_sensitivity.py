from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
TABLES = Path("reports/tables")
TABLES.mkdir(parents=True, exist_ok=True)

x = pd.read_parquet(PROCESSED / "feature_matrix.parquet")
meta = pd.read_parquet(PROCESSED / "sample_metadata.parquet")

eligible = []
for dataset, df in meta.groupby("dataset_id"):
    labels = set(df["condition"].astype(str))
    if {"HD", "control"}.issubset(labels):
        eligible.append(dataset)

meta = meta[meta["dataset_id"].isin(eligible)].reset_index(drop=True)

if "pseudobulk_id" in meta.columns and set(meta["pseudobulk_id"]).issubset(set(x.index.astype(str))):
    x = x.loc[meta["pseudobulk_id"].astype(str)].reset_index(drop=True)
else:
    x = x.iloc[meta.index].reset_index(drop=True)

y = (meta["condition"].astype(str) == "HD").astype(int).reset_index(drop=True)
groups = meta["donor_id"].astype(str).reset_index(drop=True)

feature_caps = [100, 250, 500, 1000, 2000, 4000, min(8000, x.shape[1])]

models = {
    "logistic_regularized": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=5000)),
    ]),
    "hist_gradient_boosting_regularized": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            max_iter=160,
            random_state=42,
        )),
    ]),
}

rows = []

for cap in feature_caps:
    top = x.var(axis=0).sort_values(ascending=False).head(int(cap)).index
    xs = x[top]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        probs = np.zeros(len(y))
        train_aucs = []

        for train_idx, test_idx in cv.split(xs, y, groups):
            model.fit(xs.iloc[train_idx], y.iloc[train_idx])
            train_prob = model.predict_proba(xs.iloc[train_idx])[:, 1]
            test_prob = model.predict_proba(xs.iloc[test_idx])[:, 1]
            train_aucs.append(roc_auc_score(y.iloc[train_idx], train_prob))
            probs[test_idx] = test_prob

        pred = (probs >= 0.5).astype(int)

        rows.append({
            "model": model_name,
            "n_features": int(cap),
            "mean_train_auc": float(np.mean(train_aucs)),
            "test_roc_auc": roc_auc_score(y, probs),
            "auc_gap": float(np.mean(train_aucs)) - roc_auc_score(y, probs),
            "average_precision": average_precision_score(y, probs),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
            "brier": brier_score_loss(y, probs),
            "n_samples": len(y),
            "n_hd": int(y.sum()),
            "n_control": int((1 - y).sum()),
            "eligible_datasets": ";".join(eligible),
        })

out = pd.DataFrame(rows)
out.to_csv(TABLES / "strict_supervised_feature_sensitivity.csv", index=False)
print(out.to_string(index=False))
