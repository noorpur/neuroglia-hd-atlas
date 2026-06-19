from __future__ import annotations

import gzip
import json
import os
import re
import tarfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.io import mmread

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
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
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
REPORT = ROOT / "reports" / "full_scale_research_report.md"

for p in [INTERIM, PROCESSED, TABLES, FIGURES, MODELS]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def extract_tars() -> list[Path]:
    extracted_dirs = []
    for tar_path in RAW.rglob("*.tar"):
        out_dir = INTERIM / "extracted" / tar_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        marker = out_dir / ".extracted"
        if marker.exists():
            extracted_dirs.append(out_dir)
            continue

        print(f"Extracting {tar_path} -> {out_dir}")
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(out_dir)

        marker.write_text("ok\n")
        extracted_dirs.append(out_dir)

    return extracted_dirs


def infer_col(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    return None


def load_gse64810_ba9() -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = list(RAW.rglob("GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz"))
    if not matches:
        print("GSE64810 processed count file not found. Skipping BA9.")
        return pd.DataFrame(), pd.DataFrame()

    path = matches[0]
    print(f"Loading BA9 bulk counts: {path}")
    df = pd.read_csv(path, sep="\t", compression="gzip")

    gene_col = df.columns[0]
    df = df.set_index(gene_col)
    df.index = df.index.astype(str)

    sample_cols = [c for c in df.columns if re.match(r"^[CH]_\d+", str(c))]
    if len(sample_cols) < 10:
        raise ValueError(f"Could not detect BA9 sample columns in {path}")

    expr = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).T
    if float(expr.max().max()) > 50:
        expr = np.log1p(expr)

    expr.index = [f"GSE64810::{s}" for s in expr.index]
    expr.index.name = "pseudobulk_id"

    meta = pd.DataFrame({
        "pseudobulk_id": expr.index,
        "dataset_id": "GSE64810",
        "sample_id": [s.split("::", 1)[1] for s in expr.index],
        "donor_id": expr.index,
        "species": "human",
        "brain_region": "BA9",
        "cell_type": "bulk_BA9",
        "condition": ["HD" if "::H_" in s else "control" for s in expr.index],
        "n_cells": 1,
        "matrix_source": str(path),
    })

    return expr, meta


def load_10x_directory(dir_path: Path):
    matrix_files = list(dir_path.glob("matrix.mtx")) + list(dir_path.glob("matrix.mtx.gz"))
    if not matrix_files:
        return None

    matrix_path = matrix_files[0]

    feature_candidates = (
        list(dir_path.glob("features.tsv")) +
        list(dir_path.glob("features.tsv.gz")) +
        list(dir_path.glob("genes.tsv")) +
        list(dir_path.glob("genes.tsv.gz"))
    )
    barcode_candidates = list(dir_path.glob("barcodes.tsv")) + list(dir_path.glob("barcodes.tsv.gz"))

    if not feature_candidates or not barcode_candidates:
        return None

    feature_path = feature_candidates[0]
    barcode_path = barcode_candidates[0]

    print(f"Reading 10x matrix from {dir_path}")

    mat = mmread(str(matrix_path)).tocsr()

    features = pd.read_csv(feature_path, sep="\t", header=None, compression="gzip" if feature_path.suffix == ".gz" else None)
    barcodes = pd.read_csv(barcode_path, sep="\t", header=None, compression="gzip" if barcode_path.suffix == ".gz" else None)

    genes = features.iloc[:, 1].astype(str).values if features.shape[1] > 1 else features.iloc[:, 0].astype(str).values
    cells = barcodes.iloc[:, 0].astype(str).values

    if mat.shape[0] == len(genes) and mat.shape[1] == len(cells):
        mat = mat.T
    elif mat.shape[0] == len(cells) and mat.shape[1] == len(genes):
        pass
    else:
        raise ValueError(f"Matrix shape does not match features/barcodes in {dir_path}: {mat.shape}")

    return mat, genes, cells


def find_10x_dirs() -> list[Path]:
    dirs = set()
    for matrix_file in INTERIM.rglob("matrix.mtx*"):
        dirs.add(matrix_file.parent)
    return sorted(dirs)


def load_metadata_files() -> pd.DataFrame:
    frames = []
    for p in RAW.rglob("*metadata*.csv.gz"):
        try:
            df = pd.read_csv(p, compression="gzip")
            df["__metadata_source"] = str(p)
            frames.append(df)
            print(f"Loaded metadata: {p} {df.shape}")
        except Exception as e:
            print(f"Could not read metadata {p}: {e}")

    for p in RAW.rglob("*metadata*.csv"):
        try:
            df = pd.read_csv(p)
            df["__metadata_source"] = str(p)
            frames.append(df)
            print(f"Loaded metadata: {p} {df.shape}")
        except Exception as e:
            print(f"Could not read metadata {p}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def standardize_condition(value: str) -> str:
    v = str(value).lower()
    if any(x in v for x in ["hd", "huntington", "r6", "case", "disease", "mutant"]):
        if not any(x in v for x in ["control", "wild", "wt", "normal"]):
            return "HD"
    if any(x in v for x in ["control", "wild", "wt", "normal", "healthy"]):
        return "control"
    return str(value)


def pseudobulk_single_cell(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dirs = find_10x_dirs()
    if not dirs:
        print("No 10x directories found after extraction. Skipping single-cell pseudobulk.")
        return pd.DataFrame(), pd.DataFrame()

    if metadata.empty:
        print("No metadata found. Single-cell matrices cannot be safely labelled. Skipping them.")
        return pd.DataFrame(), pd.DataFrame()

    meta_cols = list(metadata.columns)

    barcode_col = infer_col(meta_cols, ["barcode", "cell"])
    sample_col = infer_col(meta_cols, ["sample", "orig.ident", "library", "donor"])
    donor_col = infer_col(meta_cols, ["donor", "individual", "patient", "subject"])
    condition_col = infer_col(meta_cols, ["condition", "diagnosis", "disease", "genotype", "status"])
    celltype_col = infer_col(meta_cols, ["celltype", "cell_type", "subcluster", "annotation", "anno", "cluster"])
    region_col = infer_col(meta_cols, ["region", "brain", "area"])

    if barcode_col is None:
        print("Could not detect barcode/cell column in metadata. Skipping single-cell pseudobulk.")
        print("Metadata columns:", meta_cols[:50])
        return pd.DataFrame(), pd.DataFrame()

    print("Metadata column mapping:")
    print({
        "barcode": barcode_col,
        "sample": sample_col,
        "donor": donor_col,
        "condition": condition_col,
        "cell_type": celltype_col,
        "region": region_col,
    })

    metadata = metadata.copy()
    metadata["__barcode_key"] = metadata[barcode_col].astype(str)
    metadata["__barcode_short"] = metadata["__barcode_key"].str.replace(r"-\d+$", "", regex=True)

    all_expr = []
    all_meta = []

    max_cells_per_dir = int(os.environ.get("MAX_CELLS_PER_10X_DIR", "250000"))

    for d in dirs:
        loaded = load_10x_directory(d)
        if loaded is None:
            continue

        mat, genes, cells = loaded

        if mat.shape[0] > max_cells_per_dir:
            idx = rng.choice(mat.shape[0], size=max_cells_per_dir, replace=False)
            mat = mat[idx]
            cells = cells[idx]

        cell_df = pd.DataFrame({
            "__cell_local": cells.astype(str),
            "__barcode_key": cells.astype(str),
            "__barcode_short": pd.Series(cells.astype(str)).str.replace(r"-\d+$", "", regex=True).values,
        })

        merged = cell_df.merge(metadata, on="__barcode_key", how="left", suffixes=("", "_meta"))
        missing = merged[condition_col].isna().mean() if condition_col else 1.0

        if missing > 0.75:
            merged = cell_df.merge(metadata, on="__barcode_short", how="left", suffixes=("", "_meta"))
            missing = merged[condition_col].isna().mean() if condition_col else 1.0

        if condition_col is None or missing > 0.75:
            print(f"Too many unlabelled cells in {d} missing={missing:.2f}. Skipping.")
            continue

        merged["condition_std"] = merged[condition_col].map(standardize_condition)
        merged["sample_std"] = merged[sample_col].astype(str) if sample_col else d.name
        merged["donor_std"] = merged[donor_col].astype(str) if donor_col else merged["sample_std"]
        merged["cell_type_std"] = merged[celltype_col].astype(str) if celltype_col else "unknown_celltype"
        merged["region_std"] = merged[region_col].astype(str) if region_col else "unknown_region"

        valid = merged["condition_std"].isin(["HD", "control"])
        if valid.sum() < 100:
            print(f"Too few labelled HD/control cells in {d}. Skipping.")
            continue

        mat = mat[valid.values]
        merged = merged.loc[valid].reset_index(drop=True)

        group_cols = ["condition_std", "donor_std", "sample_std", "cell_type_std", "region_std"]
        group_keys = merged[group_cols].fillna("unknown").agg("||".join, axis=1)

        unique_groups = sorted(group_keys.unique())

        # Collapse duplicate genes by summing after pseudobulk construction.
        gene_names = pd.Index(genes.astype(str))
        gene_df = pd.DataFrame({"gene": gene_names})

        for key in unique_groups:
            idx = np.where(group_keys.values == key)[0]
            if len(idx) < int(os.environ.get("MIN_CELLS_PER_PSEUDOBULK", "20")):
                continue

            vec = np.asarray(mat[idx].sum(axis=0)).ravel()
            pb = pd.DataFrame({"gene": gene_names, "value": vec}).groupby("gene")["value"].sum()

            condition, donor, sample, cell_type, region = key.split("||")
            pb_id = f"singlecell::{d.name}::{condition}::{donor}::{sample}::{cell_type}::{region}"

            all_expr.append(pb.rename(pb_id))
            all_meta.append({
                "pseudobulk_id": pb_id,
                "dataset_id": "single_cell_downloaded",
                "sample_id": sample,
                "donor_id": donor,
                "species": "unknown",
                "brain_region": region,
                "cell_type": cell_type,
                "condition": condition,
                "n_cells": int(len(idx)),
                "matrix_source": str(d),
            })

    if not all_expr:
        print("No safely labelled single-cell pseudobulks were built.")
        return pd.DataFrame(), pd.DataFrame()

    expr = pd.concat(all_expr, axis=1).T.fillna(0.0)
    expr = np.log1p(expr)
    meta = pd.DataFrame(all_meta)

    return expr, meta


def build_integrated_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    extract_tars()
    metadata = load_metadata_files()

    expr_frames = []
    meta_frames = []

    ba9_expr, ba9_meta = load_gse64810_ba9()
    if not ba9_expr.empty:
        expr_frames.append(ba9_expr)
        meta_frames.append(ba9_meta)

    sc_expr, sc_meta = pseudobulk_single_cell(metadata)
    if not sc_expr.empty:
        expr_frames.append(sc_expr)
        meta_frames.append(sc_meta)

    if not expr_frames:
        raise RuntimeError("No datasets could be converted into a labelled modelling matrix.")

    common_genes = sorted(set.intersection(*[set(x.columns) for x in expr_frames]))
    if len(common_genes) < 50:
        # Use union if platforms differ too much.
        print("Low shared-gene count across datasets. Using union with zeros.")
        expr = pd.concat(expr_frames, axis=0, sort=True).fillna(0.0)
    else:
        expr = pd.concat([x[common_genes] for x in expr_frames], axis=0).fillna(0.0)

    meta = pd.concat(meta_frames, axis=0, ignore_index=True, sort=False)
    meta["condition"] = meta["condition"].map(standardize_condition)
    keep = meta["condition"].isin(["HD", "control"])
    expr = expr.loc[keep.values]
    meta = meta.loc[keep].reset_index(drop=True)

    expr.index = meta["pseudobulk_id"].astype(str)
    expr.index.name = "pseudobulk_id"

    # Remove constant and ultra-sparse features.
    nonzero_frac = (expr != 0).mean(axis=0)
    expr = expr.loc[:, nonzero_frac >= 0.05]
    variances = expr.var(axis=0)
    expr = expr.loc[:, variances > 0]

    # Keep top variable genes to make M2 run tractable.
    max_features = int(os.environ.get("MAX_FEATURES", "6000"))
    if expr.shape[1] > max_features:
        top = expr.var(axis=0).sort_values(ascending=False).head(max_features).index
        expr = expr[top]

    expr.to_parquet(PROCESSED / "pseudobulk_expression.parquet")
    meta.to_parquet(PROCESSED / "pseudobulk_metadata.parquet", index=False)

    # Also write directly as modelling matrix.
    expr.to_parquet(PROCESSED / "feature_matrix.parquet")
    meta.to_parquet(PROCESSED / "sample_metadata.parquet", index=False)

    audit = pd.DataFrame([{
        "n_pseudobulk_samples": len(expr),
        "n_features": expr.shape[1],
        "n_hd": int((meta["condition"] == "HD").sum()),
        "n_control": int((meta["condition"] == "control").sum()),
        "datasets": ";".join(sorted(meta["dataset_id"].astype(str).unique())),
        "cell_types": int(meta["cell_type"].astype(str).nunique()),
        "regions": int(meta["brain_region"].astype(str).nunique()),
        "donors": int(meta["donor_id"].astype(str).nunique()),
    }])
    audit.to_csv(TABLES / "full_scale_dataset_audit.csv", index=False)

    print("Integrated matrix:")
    print(audit.to_string(index=False))

    if len(expr) < 80:
        print("\nWARNING: Integrated matrix is still small. This is not yet a large-scale glial atlas run.")
    if meta["dataset_id"].nunique() < 2:
        print("\nWARNING: Only one dataset was converted. External validation is not available yet.")

    return expr, meta


def make_models():
    return {
        "logistic_regularized": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=5000)),
            ]),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        "random_forest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
            ]),
            {
                "clf__n_estimators": [300, 700],
                "clf__max_depth": [None, 3, 5],
                "clf__min_samples_leaf": [1, 2, 4],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(random_state=SEED)),
            ]),
            {
                "clf__max_iter": [100, 250],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__l2_regularization": [0.0, 0.1, 1.0],
                "clf__max_leaf_nodes": [7, 15, 31],
            },
        ),
    }


def labels(meta: pd.DataFrame) -> pd.Series:
    return (meta["condition"].astype(str) == "HD").astype(int)


def safe_auc(y, p):
    if pd.Series(y).nunique() < 2:
        return np.nan
    return roc_auc_score(y, p)


def safe_ap(y, p):
    if pd.Series(y).nunique() < 2:
        return np.nan
    return average_precision_score(y, p)


def nested_cv_train_tune_test(x: pd.DataFrame, meta: pd.DataFrame):
    y = labels(meta)
    groups = meta["donor_id"].astype(str)

    min_class = int(y.value_counts().min())
    n_groups = int(groups.nunique())
    n_outer = max(2, min(5, min_class, n_groups))
    n_inner = max(2, min(3, min_class, n_groups))

    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=SEED + 1)

    rows = []
    preds = []

    for model_name, (pipe, grid) in make_models().items():
        print(f"\nNested train/tune/test: {model_name}")

        for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups)):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            g_train = groups.iloc[train_idx]

            search = GridSearchCV(
                pipe,
                grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_train, y_train, groups=g_train)

            train_prob = search.predict_proba(x_train)[:, 1]
            test_prob = search.predict_proba(x_test)[:, 1]
            test_pred = (test_prob >= 0.5).astype(int)

            row = {
                "model": model_name,
                "fold": fold,
                "inner_best_roc_auc": search.best_score_,
                "train_roc_auc": safe_auc(y_train, train_prob),
                "test_roc_auc": safe_auc(y_test, test_prob),
                "auc_gap": safe_auc(y_train, train_prob) - safe_auc(y_test, test_prob),
                "test_average_precision": safe_ap(y_test, test_prob),
                "test_accuracy": accuracy_score(y_test, test_pred),
                "test_balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
                "test_f1": f1_score(y_test, test_pred, zero_division=0),
                "test_brier": brier_score_loss(y_test, test_prob),
                "best_params": json.dumps(search.best_params_),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
            rows.append(row)

            for idx, prob in zip(test_idx, test_prob):
                preds.append({
                    "model": model_name,
                    "fold": fold,
                    "sample_index": int(idx),
                    "pseudobulk_id": meta.iloc[idx]["pseudobulk_id"],
                    "dataset_id": meta.iloc[idx]["dataset_id"],
                    "cell_type": meta.iloc[idx]["cell_type"],
                    "brain_region": meta.iloc[idx]["brain_region"],
                    "donor_id": meta.iloc[idx]["donor_id"],
                    "y_true": int(y.iloc[idx]),
                    "y_prob": float(prob),
                })

        final_model = clone(pipe)
        final_model.fit(x, y)
        joblib.dump(final_model, MODELS / f"full_scale_{model_name}.joblib")

    metrics = pd.DataFrame(rows)
    pred_df = pd.DataFrame(preds)

    metrics.to_csv(TABLES / "full_scale_nested_cv_metrics.csv", index=False)
    pred_df.to_csv(TABLES / "full_scale_nested_cv_predictions.csv", index=False)

    summary = (
        metrics.groupby("model")
        .agg(
            mean_train_auc=("train_roc_auc", "mean"),
            mean_test_auc=("test_roc_auc", "mean"),
            std_test_auc=("test_roc_auc", "std"),
            mean_auc_gap=("auc_gap", "mean"),
            max_auc_gap=("auc_gap", "max"),
            mean_average_precision=("test_average_precision", "mean"),
            mean_balanced_accuracy=("test_balanced_accuracy", "mean"),
            mean_f1=("test_f1", "mean"),
            mean_brier=("test_brier", "mean"),
        )
        .reset_index()
        .sort_values("mean_test_auc", ascending=False)
    )
    summary.to_csv(TABLES / "full_scale_nested_cv_summary.csv", index=False)

    return metrics, pred_df, summary


def permutation_control(x: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    y = labels(meta)
    groups = meta["donor_id"].astype(str)

    pipe, grid = make_models()["logistic_regularized"]
    pipe = clone(pipe)

    min_class = int(y.value_counts().min())
    n_groups = int(groups.nunique())
    n_splits = max(2, min(5, min_class, n_groups))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    n_perm = int(os.environ.get("N_PERMUTATIONS", "100"))
    rows = []

    for i in range(n_perm):
        y_perm = pd.Series(rng.permutation(y.values), index=y.index)
        probs = np.zeros(len(y_perm))

        for train_idx, test_idx in cv.split(x, y_perm, groups):
            m = clone(pipe)
            m.fit(x.iloc[train_idx], y_perm.iloc[train_idx])
            probs[test_idx] = m.predict_proba(x.iloc[test_idx])[:, 1]

        rows.append({
            "permutation": i,
            "roc_auc": safe_auc(y_perm, probs),
            "average_precision": safe_ap(y_perm, probs),
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "full_scale_permutation_control.csv", index=False)
    return out


def leakage_screen(x: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    x = x.reset_index(drop=True)
    y = labels(meta).reset_index(drop=True)
    rows = []

    for col in x.columns:
        s = pd.to_numeric(x[col], errors="coerce")
        if s.nunique(dropna=True) < 2:
            continue
        mask = s.notna()
        if mask.sum() < 10 or y[mask].nunique() < 2:
            continue

        auc = roc_auc_score(y[mask], s[mask])
        rows.append({
            "feature": col,
            "single_feature_auc_directionless": max(auc, 1 - auc),
            "raw_auc": auc,
            "n_unique": int(s.nunique(dropna=True)),
            "missing_fraction": float(1 - mask.mean()),
        })

    out = pd.DataFrame(rows).sort_values("single_feature_auc_directionless", ascending=False)
    out.to_csv(TABLES / "full_scale_single_feature_leakage_screen.csv", index=False)
    return out


def make_figures(x, meta, pred_df, summary, perm, leakage):
    # Dataset composition.
    comp = meta.groupby(["dataset_id", "condition"]).size().reset_index(name="n")
    comp.to_csv(TABLES / "full_scale_dataset_composition.csv", index=False)

    plt.figure(figsize=(8, 5))
    pivot = comp.pivot(index="dataset_id", columns="condition", values="n").fillna(0)
    pivot.plot(kind="bar", figsize=(8, 5))
    plt.title("Dataset composition")
    plt.ylabel("pseudobulk samples")
    savefig(FIGURES / "full_scale_dataset_composition.png")

    # ROC.
    plt.figure(figsize=(7, 6))
    for model, df in pred_df.groupby("model"):
        if df["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        auc = roc_auc_score(df["y_true"], df["y_prob"])
        plt.plot(fpr, tpr, label=f"{model} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.title("Full-scale nested CV ROC curves")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=8)
    savefig(FIGURES / "full_scale_roc_curves.png")

    # PR.
    plt.figure(figsize=(7, 6))
    for model, df in pred_df.groupby("model"):
        if df["y_true"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(df["y_true"], df["y_prob"])
        ap = average_precision_score(df["y_true"], df["y_prob"])
        plt.plot(recall, precision, label=f"{model} AP={ap:.3f}")
    plt.title("Full-scale nested CV precision-recall curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    savefig(FIGURES / "full_scale_precision_recall_curves.png")

    # Calibration.
    plt.figure(figsize=(7, 6))
    for model, df in pred_df.groupby("model"):
        bins = min(8, max(3, len(df) // 10))
        frac_pos, mean_pred = calibration_curve(df["y_true"], df["y_prob"], n_bins=bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=model)
    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.title("Full-scale calibration curves")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.legend(fontsize=8)
    savefig(FIGURES / "full_scale_calibration_curves.png")

    # Overfitting gap.
    ordered = summary.sort_values("mean_auc_gap")
    plt.figure(figsize=(8, 4.8))
    bars = plt.barh(ordered["model"], ordered["mean_auc_gap"])
    plt.axvline(0.02, linestyle=":", label="low concern threshold")
    plt.axvline(0.08, linestyle="--", label="high concern threshold")
    for bar, val in zip(bars, ordered["mean_auc_gap"]):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center")
    plt.title("Full-scale overfitting audit")
    plt.xlabel("mean train ROC-AUC - mean test ROC-AUC")
    plt.ylabel("model")
    plt.legend(fontsize=8)
    savefig(FIGURES / "full_scale_overfitting_auc_gap.png")

    # Permutation.
    plt.figure(figsize=(7, 5))
    plt.hist(perm["roc_auc"].dropna(), bins=20)
    plt.axvline(0.5, linestyle=":")
    plt.title("Full-scale permutation-label control")
    plt.xlabel("ROC-AUC after label shuffle")
    plt.ylabel("count")
    savefig(FIGURES / "full_scale_permutation_control.png")

    # Leakage.
    top = leakage.head(25).iloc[::-1]
    if len(top):
        plt.figure(figsize=(8, max(5, len(top) * 0.28)))
        plt.barh(top["feature"], top["single_feature_auc_directionless"])
        plt.axvline(0.98, linestyle="--")
        plt.title("Single-feature separation screen")
        plt.xlabel("directionless single-feature ROC-AUC")
        plt.ylabel("feature")
        savefig(FIGURES / "full_scale_single_feature_screen.png")

    # PCA.
    x_small = x.copy()
    if x_small.shape[1] > 2000:
        top_var = x_small.var(axis=0).sort_values(ascending=False).head(2000).index
        x_small = x_small[top_var]

    z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(x_small))
    pcs = PCA(n_components=2, random_state=SEED).fit_transform(z)
    pc = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "condition": meta["condition"].values,
        "dataset_id": meta["dataset_id"].values,
        "cell_type": meta["cell_type"].values,
    })
    pc.to_csv(TABLES / "full_scale_pca_coordinates.csv", index=False)

    plt.figure(figsize=(7, 6))
    for cond, df in pc.groupby("condition"):
        plt.scatter(df["PC1"], df["PC2"], label=cond, alpha=0.75)
    plt.title("PCA by condition")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    savefig(FIGURES / "full_scale_pca_condition.png")


def write_report(meta, summary, perm, leakage):
    def md(df, n=30):
        if df is None or len(df) == 0:
            return "_No rows._"
        return df.head(n).to_markdown(index=False)

    audit = pd.read_csv(TABLES / "full_scale_dataset_audit.csv")
    comp = pd.read_csv(TABLES / "full_scale_dataset_composition.csv")

    warnings_list = []

    if audit["n_pseudobulk_samples"].iloc[0] < 80:
        warnings_list.append("The converted matrix has fewer than 80 pseudobulk samples. Treat this as a pilot-scale benchmark, not a complete atlas-scale result.")
    if meta["dataset_id"].nunique() < 2:
        warnings_list.append("Only one labelled dataset was converted, so independent external validation is not yet available.")
    if summary["mean_auc_gap"].max() > 0.08:
        warnings_list.append("At least one model shows a high train-test AUC gap.")
    if len(perm) and perm["roc_auc"].mean() > 0.65:
        warnings_list.append("Permutation-label control is higher than expected and needs split/leakage investigation.")
    if len(leakage) and leakage["single_feature_auc_directionless"].iloc[0] > 0.98:
        warnings_list.append("A single feature nearly separates the labels. This may be biological, but it requires manual review.")

    if not warnings_list:
        warnings_list.append("No automatic red-flag threshold was triggered. Independent validation remains required.")

    figs = sorted(p.name for p in FIGURES.glob("full_scale*.png"))
    tabs = sorted(p.name for p in TABLES.glob("full_scale*.csv"))

    lines = [
        "# Full-Scale NeuroGlia-HD Atlas Research Report",
        "",
        "## Dataset audit",
        md(audit),
        "",
        "## Dataset composition",
        md(comp),
        "",
        "## Nested cross-validation summary",
        "Hyperparameters were selected inside inner folds and evaluated on held-out outer folds.",
        "",
        md(summary),
        "",
        "## Permutation-label control",
        md(perm.describe().reset_index()),
        "",
        "## Single-feature separation screen",
        md(leakage, 25),
        "",
        "## Automatic issue flags",
    ]

    lines.extend([f"- {w}" for w in warnings_list])

    lines.extend([
        "",
        "## Figures",
    ])
    lines.extend([f"- `reports/figures/{f}`" for f in figs])

    lines.extend([
        "",
        "## Tables",
    ])
    lines.extend([f"- `reports/tables/{t}`" for t in tabs])

    lines.extend([
        "",
        "## Interpretation note",
        "Near-perfect discrimination is treated as an audit trigger rather than a final conclusion. The modelling evidence is strongest when it remains stable under nested cross-validation, permutation controls collapse toward chance, calibration is acceptable, single-feature screens are biologically plausible, and external datasets replicate the signal.",
        "",
    ])

    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT}")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("Building integrated large-scale matrix...")
    x, meta = build_integrated_matrix()

    print("\nRunning nested train/tune/test...")
    metrics, pred_df, summary = nested_cv_train_tune_test(x, meta)

    print("\nRunning permutation-label control...")
    perm = permutation_control(x, meta)

    print("\nRunning leakage screen...")
    leakage = leakage_screen(x, meta)

    print("\nMaking figures...")
    make_figures(x, meta, pred_df, summary, perm, leakage)

    print("\nWriting report...")
    write_report(meta, summary, perm, leakage)

    print("\nDONE.")
    print("Main report:", REPORT)
    print("Main summary:", TABLES / "full_scale_nested_cv_summary.csv")
    print("Main figure:", FIGURES / "full_scale_roc_curves.png")


if __name__ == "__main__":
    main()
