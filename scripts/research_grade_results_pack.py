from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
REPORT = ROOT / "reports" / "research_grade_analysis_report.md"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def load_xy():
    x_path = PROCESSED / "feature_matrix.parquet"
    meta_path = PROCESSED / "sample_metadata.parquet"

    if not x_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Expected data/processed/feature_matrix.parquet and "
            "data/processed/sample_metadata.parquet. Run neurogliahd pseudobulk first."
        )

    x = pd.read_parquet(x_path)
    meta = pd.read_parquet(meta_path)

    if len(x) != len(meta):
        raise ValueError(f"Feature/meta row mismatch: {x.shape} versus {meta.shape}")

    condition = meta["condition"].astype(str).str.lower()
    y = condition.isin(["hd", "huntington", "disease", "case", "r6/2", "r62", "1", "true"]).astype(int)

    group_col = "donor_id" if "donor_id" in meta.columns else meta.columns[0]
    groups = meta[group_col].astype(str)

    return x, meta, y, groups, group_col


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def safe_ap(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def get_proba(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(x)
        z = (z - np.min(z)) / (np.max(z) - np.min(z) + 1e-12)
        return z
    return model.predict(x)


def make_models():
    return {
        "logistic_regularized": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=700,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        l2_regularization=0.1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def dataset_audit(x, meta, y, groups):
    rows = {
        "n_samples": len(x),
        "n_features": x.shape[1],
        "n_positive_hd": int(y.sum()),
        "n_negative_control": int((1 - y).sum()),
        "positive_fraction": float(y.mean()),
        "n_groups": int(groups.nunique()),
        "n_duplicate_feature_rows": int(pd.DataFrame(x).duplicated().sum()),
        "n_missing_values": int(pd.DataFrame(x).isna().sum().sum()),
    }
    out = pd.DataFrame([rows])
    out.to_csv(TABLES / "dataset_audit.csv", index=False)

    class_balance = (
        pd.DataFrame({"condition_binary": y, "group": groups})
        .groupby("condition_binary")
        .agg(n_samples=("condition_binary", "size"), n_groups=("group", "nunique"))
        .reset_index()
    )
    class_balance.to_csv(TABLES / "class_balance.csv", index=False)

    plt.figure(figsize=(6, 4))
    class_balance.plot(x="condition_binary", y="n_samples", kind="bar", legend=False)
    plt.title("Class balance")
    plt.xlabel("condition: 0=control, 1=HD")
    plt.ylabel("samples")
    savefig(FIGURES / "class_balance.png")

    return out, class_balance


def crossval_overfitting_check(x, y, groups):
    min_class = int(pd.Series(y).value_counts().min())
    n_groups = int(pd.Series(groups).nunique())
    n_splits = max(2, min(5, min_class, n_groups))

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(cv.split(x, y, groups))

    fold_rows = []
    prediction_rows = []

    for model_name, estimator in make_models().items():
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            model = clone(estimator)
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(x_train, y_train)

            train_prob = get_proba(model, x_train)
            test_prob = get_proba(model, x_test)
            test_pred = (test_prob >= 0.5).astype(int)

            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "train_roc_auc": safe_auc(y_train, train_prob),
                    "test_roc_auc": safe_auc(y_test, test_prob),
                    "auc_gap": safe_auc(y_train, train_prob) - safe_auc(y_test, test_prob),
                    "test_accuracy": accuracy_score(y_test, test_pred),
                    "test_balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
                    "test_f1": f1_score(y_test, test_pred, zero_division=0),
                    "test_average_precision": safe_ap(y_test, test_prob),
                    "test_brier": brier_score_loss(y_test, test_prob),
                }
            )

            for idx, prob in zip(test_idx, test_prob):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_id,
                        "sample_index": int(idx),
                        "group": groups.iloc[idx],
                        "y_true": int(y.iloc[idx]),
                        "y_prob": float(prob),
                    }
                )

        final_model = clone(estimator)
        final_model.fit(x, y)
        joblib.dump(final_model, MODELS / f"{model_name}_final.joblib")

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.DataFrame(prediction_rows)

    folds_df.to_csv(TABLES / "crossval_train_test_metrics.csv", index=False)
    preds_df.to_csv(TABLES / "crossval_predictions_recomputed.csv", index=False)

    summary = (
        folds_df.groupby("model")
        .agg(
            mean_train_auc=("train_roc_auc", "mean"),
            mean_test_auc=("test_roc_auc", "mean"),
            mean_auc_gap=("auc_gap", "mean"),
            max_auc_gap=("auc_gap", "max"),
            mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
            mean_test_f1=("test_f1", "mean"),
            mean_test_average_precision=("test_average_precision", "mean"),
            mean_test_brier=("test_brier", "mean"),
        )
        .reset_index()
        .sort_values("mean_test_auc", ascending=False)
    )

    summary.to_csv(TABLES / "overfitting_diagnostics.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(summary["model"], summary["mean_auc_gap"])
    plt.title("Overfitting diagnostic: train-test ROC-AUC gap")
    plt.xlabel("mean train AUC - mean test AUC")
    plt.ylabel("model")
    savefig(FIGURES / "overfitting_auc_gap.png")

    return folds_df, preds_df, summary


def plot_prediction_figures(preds_df):
    plt.figure(figsize=(7, 6))
    for model_name, df in preds_df.groupby("model"):
        if len(df["y_true"].unique()) < 2:
            continue
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        auc = roc_auc_score(df["y_true"], df["y_prob"])
        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.title("ROC curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=8)
    savefig(FIGURES / "roc_curves.png")

    plt.figure(figsize=(7, 6))
    for model_name, df in preds_df.groupby("model"):
        if len(df["y_true"].unique()) < 2:
            continue
        precision, recall, _ = precision_recall_curve(df["y_true"], df["y_prob"])
        ap = average_precision_score(df["y_true"], df["y_prob"])
        plt.plot(recall, precision, label=f"{model_name} AP={ap:.3f}")
    plt.title("Precision-recall curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    savefig(FIGURES / "precision_recall_curves.png")

    plt.figure(figsize=(7, 6))
    for model_name, df in preds_df.groupby("model"):
        frac_pos, mean_pred = calibration_curve(
            df["y_true"], df["y_prob"], n_bins=min(8, max(3, len(df) // 10)), strategy="quantile"
        )
        plt.plot(mean_pred, frac_pos, marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.title("Calibration curves")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.legend(fontsize=8)
    savefig(FIGURES / "calibration_curves.png")

    for model_name, df in preds_df.groupby("model"):
        y_pred = (df["y_prob"] >= 0.5).astype(int)
        cm = confusion_matrix(df["y_true"], y_pred, labels=[0, 1])

        plt.figure(figsize=(5, 4))
        plt.imshow(cm)
        plt.title(f"Confusion matrix: {model_name}")
        plt.xticks([0, 1], ["pred control", "pred HD"])
        plt.yticks([0, 1], ["true control", "true HD"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        savefig(FIGURES / f"confusion_matrix_{model_name}.png")

    plt.figure(figsize=(8, 5))
    metrics = []
    for model_name, df in preds_df.groupby("model"):
        y_pred = (df["y_prob"] >= 0.5).astype(int)
        metrics.append(
            {
                "model": model_name,
                "roc_auc": safe_auc(df["y_true"], df["y_prob"]),
                "average_precision": safe_ap(df["y_true"], df["y_prob"]),
                "balanced_accuracy": balanced_accuracy_score(df["y_true"], y_pred),
                "f1": f1_score(df["y_true"], y_pred, zero_division=0),
            }
        )
    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(TABLES / "recomputed_model_metrics.csv", index=False)

    metric_df.set_index("model").plot(kind="bar", figsize=(9, 5))
    plt.title("Model comparison across metrics")
    plt.ylabel("score")
    plt.ylim(0, 1.05)
    savefig(FIGURES / "multi_metric_model_comparison.png")


def permutation_label_control(x, y, groups):
    estimator = make_models()["logistic_regularized"]

    min_class = int(pd.Series(y).value_counts().min())
    n_groups = int(pd.Series(groups).nunique())
    n_splits = max(2, min(5, min_class, n_groups))

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(cv.split(x, y, groups))

    n_permutations = int(os.environ.get("N_PERMUTATIONS", "50"))
    rows = []

    for perm_id in range(n_permutations):
        y_perm = pd.Series(rng.permutation(y.values), index=y.index)
        probs = np.zeros(len(y_perm), dtype=float)

        for train_idx, test_idx in folds:
            model = clone(estimator)
            model.fit(x.iloc[train_idx], y_perm.iloc[train_idx])
            probs[test_idx] = get_proba(model, x.iloc[test_idx])

        rows.append(
            {
                "permutation": perm_id,
                "roc_auc": safe_auc(y_perm, probs),
                "average_precision": safe_ap(y_perm, probs),
            }
        )

    perm_df = pd.DataFrame(rows)
    perm_df.to_csv(TABLES / "permutation_label_control.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.hist(perm_df["roc_auc"].dropna(), bins=15)
    plt.axvline(0.5, linestyle=":")
    plt.title("Permutation-label control")
    plt.xlabel("ROC-AUC after label shuffle")
    plt.ylabel("count")
    savefig(FIGURES / "permutation_label_control.png")

    return perm_df


def feature_leakage_screen(x, y):
    rows = []
    for col in x.columns:
        s = pd.to_numeric(x[col], errors="coerce")
        if s.nunique(dropna=True) < 2:
            continue

        mask = s.notna()
        if mask.sum() < 5 or pd.Series(y[mask]).nunique() < 2:
            continue

        try:
            auc = roc_auc_score(y[mask], s[mask])
        except Exception:
            continue

        directionless_auc = max(auc, 1 - auc)
        rows.append(
            {
                "feature": col,
                "single_feature_auc_directionless": directionless_auc,
                "raw_auc": auc,
                "missing_fraction": float(1 - mask.mean()),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )

    audit = pd.DataFrame(rows).sort_values("single_feature_auc_directionless", ascending=False)
    audit.to_csv(TABLES / "single_feature_leakage_screen.csv", index=False)

    top = audit.head(25).iloc[::-1]
    if len(top):
        plt.figure(figsize=(8, max(5, len(top) * 0.25)))
        plt.barh(top["feature"], top["single_feature_auc_directionless"])
        plt.title("Single-feature separation screen")
        plt.xlabel("directionless single-feature ROC-AUC")
        plt.ylabel("feature")
        savefig(FIGURES / "single_feature_leakage_screen.png")

    return audit


def run_optuna_tuning(x, y, groups):
    try:
        from neuroglia_hd.config import load_config, get_seed
        from neuroglia_hd.models.tune import tune_with_optuna

        cfg = load_config("configs/default.yaml")
        n_trials = int(os.environ.get("N_TRIALS", cfg.get("models", {}).get("tune", {}).get("n_trials", 50)))
        seed = get_seed(cfg)

        study = tune_with_optuna(
            x=x,
            y=y,
            groups=groups,
            n_trials=n_trials,
            seed=seed,
        )

        trials = study.trials_dataframe()
        trials.to_csv(TABLES / "optuna_trials.csv", index=False)

        with open(TABLES / "best_optuna_model.txt", "w") as f:
            f.write(f"Best ROC-AUC: {study.best_value}\n")
            f.write(f"Best params: {study.best_params}\n")

        if "value" in trials.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(trials["number"], trials["value"], marker="o")
            plt.title("Optuna optimization history")
            plt.xlabel("trial")
            plt.ylabel("validation ROC-AUC")
            savefig(FIGURES / "optuna_optimization_history.png")

        return study.best_value, study.best_params

    except Exception as e:
        with open(TABLES / "optuna_error.txt", "w") as f:
            f.write(repr(e))
        return None, None


def latent_figures():
    hist_path = TABLES / "latent_training_history.csv"
    recon_path = TABLES / "latent_reconstruction_error.csv"

    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist.to_csv(TABLES / "latent_training_history_reviewed.csv", index=False)

        numeric_cols = [c for c in hist.columns if pd.api.types.is_numeric_dtype(hist[c])]
        if numeric_cols:
            plt.figure(figsize=(8, 5))
            for c in numeric_cols:
                if c.lower() not in {"epoch"}:
                    x_axis = hist["epoch"] if "epoch" in hist.columns else np.arange(len(hist))
                    plt.plot(x_axis, hist[c], label=c)
            plt.title("Latent model training history")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(fontsize=8)
            savefig(FIGURES / "latent_training_history.png")

    if recon_path.exists():
        recon = pd.read_csv(recon_path)
        numeric_cols = [c for c in recon.columns if pd.api.types.is_numeric_dtype(recon[c])]
        if numeric_cols:
            plt.figure(figsize=(7, 5))
            recon[numeric_cols[0]].hist(bins=20)
            plt.title("Latent reconstruction error")
            plt.xlabel(numeric_cols[0])
            plt.ylabel("count")
            savefig(FIGURES / "latent_reconstruction_error.png")


def write_report(dataset, class_balance, overfit, perm, leakage, optuna_best, optuna_params):
    def md_table(df, n=20):
        if df is None or len(df) == 0:
            return "_No rows available._"
        return df.head(n).to_markdown(index=False)

    figure_files = sorted(p.name for p in FIGURES.glob("*.png"))
    table_files = sorted(p.name for p in TABLES.glob("*.csv"))

    red_flags = []
    if len(overfit) and overfit["mean_auc_gap"].max() > 0.08:
        red_flags.append("At least one model shows a train-test ROC-AUC gap greater than 0.08.")
    if len(leakage) and leakage["single_feature_auc_directionless"].iloc[0] > 0.98:
        red_flags.append("At least one individual feature is almost perfectly separating the labels; inspect for biological plausibility or leakage.")
    if len(perm) and perm["roc_auc"].mean() > 0.65:
        red_flags.append("Permutation-label control is higher than expected; inspect split design and feature construction.")

    if not red_flags:
        red_flags.append("No automatic red-flag threshold was triggered, but independent validation is still required.")

    lines = [
        "# NeuroGlia-HD Atlas: Research-Grade Analysis Pack",
        "",
        "## Dataset audit",
        md_table(dataset),
        "",
        "## Class balance",
        md_table(class_balance),
        "",
        "## Cross-validation and overfitting diagnostics",
        "The table below compares train and held-out performance. Large train-test gaps suggest memorisation or leakage risk.",
        "",
        md_table(overfit),
        "",
        "## Hyperparameter tuning",
    ]

    if optuna_best is not None:
        lines += [
            f"Best Optuna validation ROC-AUC: **{optuna_best:.4f}**",
            "",
            f"Best parameters: `{optuna_params}`",
        ]
    else:
        lines += ["Optuna tuning did not complete. See `reports/tables/optuna_error.txt`."]

    lines += [
        "",
        "## Permutation-label control",
        "A shuffled-label model should collapse toward chance. If it remains strong, the split or feature matrix needs investigation.",
        "",
        md_table(perm.describe().reset_index()),
        "",
        "## Single-feature separation / leakage screen",
        "High values here are not automatically wrong. In gene-expression disease classification, some genes or signatures may separate strongly. Still, near-perfect single-feature separation must be audited.",
        "",
        md_table(leakage, n=25),
        "",
        "## Automatic red-flag summary",
    ]

    lines += [f"- {x}" for x in red_flags]

    lines += [
        "",
        "## Generated figures",
    ]

    lines += [f"- `reports/figures/{f}`" for f in figure_files]

    lines += [
        "",
        "## Generated tables",
    ]

    lines += [f"- `reports/tables/{f}`" for f in table_files]

    lines += [
        "",
        "## Interpretation guardrails",
        "These outputs are exploratory. Strong discrimination must be validated with independent cohorts, strict donor-level or cohort-level splits, biological review of top features, and functional/experimental confirmation where appropriate.",
        "",
    ]

    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT}")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    x, meta, y, groups, group_col = load_xy()

    dataset, class_balance = dataset_audit(x, meta, y, groups)
    folds, preds, overfit = crossval_overfitting_check(x, y, groups)
    plot_prediction_figures(preds)
    perm = permutation_label_control(x, y, groups)
    leakage = feature_leakage_screen(x, y)
    optuna_best, optuna_params = run_optuna_tuning(x, y, groups)
    latent_figures()

    write_report(dataset, class_balance, overfit, perm, leakage, optuna_best, optuna_params)

    print("\nDone. Main outputs:")
    print("  reports/research_grade_analysis_report.md")
    print("  reports/tables/overfitting_diagnostics.csv")
    print("  reports/tables/optuna_trials.csv")
    print("  reports/tables/permutation_label_control.csv")
    print("  reports/tables/single_feature_leakage_screen.csv")
    print("  reports/figures/roc_curves.png")
    print("  reports/figures/precision_recall_curves.png")
    print("  reports/figures/calibration_curves.png")
    print("  reports/figures/overfitting_auc_gap.png")


if __name__ == "__main__":
    main()
