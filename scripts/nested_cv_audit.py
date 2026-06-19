from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")
TABLES = Path("reports/tables")
TABLES.mkdir(parents=True, exist_ok=True)

x = pd.read_parquet(PROCESSED / "feature_matrix.parquet")
meta = pd.read_parquet(PROCESSED / "sample_metadata.parquet")

y = meta["condition"].astype(str).str.lower().isin(
    ["hd", "huntington", "disease", "case", "r6/2", "r62", "1", "true"]
).astype(int)

groups = meta["donor_id"].astype(str)

outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=123)

models = {
    "logistic_regularized": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=5000))
        ]),
        {
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        }
    ),
    "random_forest": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1))
        ]),
        {
            "clf__n_estimators": [300, 700],
            "clf__max_depth": [None, 3, 5],
            "clf__min_samples_leaf": [1, 2, 4]
        }
    ),
    "hist_gradient_boosting": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(random_state=42))
        ]),
        {
            "clf__max_iter": [100, 250],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__l2_regularization": [0.0, 0.1, 1.0],
            "clf__max_leaf_nodes": [7, 15, 31]
        }
    ),
}

rows = []
pred_rows = []

for model_name, (pipe, grid) in models.items():
    print(f"Nested CV: {model_name}")

    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups)):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train = groups.iloc[train_idx]

        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )

        search.fit(x_train, y_train, groups=g_train)

        prob = search.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        rows.append({
            "model": model_name,
            "outer_fold": fold,
            "inner_best_roc_auc": search.best_score_,
            "outer_roc_auc": roc_auc_score(y_test, prob),
            "outer_average_precision": average_precision_score(y_test, prob),
            "outer_balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "outer_f1": f1_score(y_test, pred, zero_division=0),
            "outer_brier": brier_score_loss(y_test, prob),
            "best_params": search.best_params_,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

        for idx, p in zip(test_idx, prob):
            pred_rows.append({
                "model": model_name,
                "outer_fold": fold,
                "sample_index": int(idx),
                "donor_id": groups.iloc[idx],
                "y_true": int(y.iloc[idx]),
                "y_prob": float(p),
            })

nested = pd.DataFrame(rows)
preds = pd.DataFrame(pred_rows)

nested.to_csv(TABLES / "nested_cv_metrics.csv", index=False)
preds.to_csv(TABLES / "nested_cv_predictions.csv", index=False)

summary = (
    nested.groupby("model")
    .agg(
        mean_outer_roc_auc=("outer_roc_auc", "mean"),
        std_outer_roc_auc=("outer_roc_auc", "std"),
        mean_outer_average_precision=("outer_average_precision", "mean"),
        mean_outer_balanced_accuracy=("outer_balanced_accuracy", "mean"),
        mean_outer_f1=("outer_f1", "mean"),
        mean_outer_brier=("outer_brier", "mean"),
    )
    .reset_index()
    .sort_values("mean_outer_roc_auc", ascending=False)
)

summary.to_csv(TABLES / "nested_cv_summary.csv", index=False)

print("\nNested CV summary:")
print(summary.to_string(index=False))
