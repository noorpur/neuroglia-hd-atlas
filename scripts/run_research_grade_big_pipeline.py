#!/usr/bin/env python3
"""Research-grade NeuroGlia-HD pipeline.

This script is intentionally conservative. It converts every public dataset it can
parse, builds an atlas-level pseudobulk matrix, then trains supervised models only
on datasets with both HD and control labels. It refuses to call a run "full-scale"
when only a single small benchmark dataset was converted.

Outputs
-------
reports/research_grade_big_report.md
reports/tables/*
reports/figures/*
models/*
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tarfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

for d in [RAW, INTERIM, PROCESSED, REPORTS, TABLES, FIGURES, MODELS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class MatrixBundle:
    x: pd.DataFrame
    meta: pd.DataFrame


def log(msg: str) -> None:
    print(msg, flush=True)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(x)).strip("_")


def standard_condition(value: object) -> str:
    v = str(value).strip().lower()
    if v in {"", "nan", "none", "unknown", "na"}:
        return "unknown"
    control_tokens = ["control", "ctrl", "normal", "wt", "wild", "healthy", "con"]
    hd_tokens = ["hd", "huntington", "disease", "case", "mutant", "r6/2", "r62"]
    if any(tok in v for tok in control_tokens) and not any(tok in v for tok in ["hd", "huntington"]):
        return "control"
    if any(tok in v for tok in hd_tokens):
        return "HD"
    return "unknown"


def infer_from_filename(path: Path) -> dict[str, str]:
    name = path.name
    stem = re.sub(r"(_filtered|_raw)?_feature_bc_matrix\.h5$", "", name)
    parts = stem.split("_")
    gsm = parts[0] if parts else stem
    # common region abbreviations in the downloaded filenames
    region = "unknown_region"
    for token in reversed(parts):
        if token in {"CB", "HC", "IFG", "CN", "STR", "CTX", "BA9", "ST", "Cortex", "Striatum"}:
            region = token
            break
    cond = standard_condition(name)
    sample = stem
    donor = stem
    return {
        "gsm": gsm,
        "sample_id": sample,
        "donor_id": donor,
        "condition": cond,
        "brain_region": region,
    }


def extract_archives() -> None:
    """Extract GEO RAW tar archives once, safely enough for GEO local files."""
    for tar_path in RAW.rglob("*.tar"):
        out_dir = INTERIM / "extracted" / tar_path.stem
        marker = out_dir / ".extracted"
        if marker.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"Extracting {tar_path} -> {out_dir}")
        with tarfile.open(tar_path, "r:*") as tar:
            # Python 3.14 will change tar defaults; data filter is best when available.
            try:
                tar.extractall(out_dir, filter="data")
            except TypeError:
                tar.extractall(out_dir)
        marker.write_text("ok\n")


def inventory_files() -> pd.DataFrame:
    roots = [RAW, INTERIM / "extracted"]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                rows.append({"path": str(p), "size_bytes": p.stat().st_size, "suffixes": "".join(p.suffixes[-3:])})
    inv = pd.DataFrame(rows).sort_values("path") if rows else pd.DataFrame(columns=["path", "size_bytes", "suffixes"])
    inv.to_csv(TABLES / "downloaded_file_inventory.csv", index=False)
    return inv


def read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    if path.name.endswith(".gz"):
        return pd.read_csv(path, compression="gzip", **kwargs)
    return pd.read_csv(path, **kwargs)


def convert_gse64810_ba9(max_features: int | None = None) -> MatrixBundle | None:
    matches = list(RAW.rglob("GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz"))
    if not matches:
        log("GSE64810 BA9 processed counts not found. Skipping BA9 benchmark.")
        return None
    path = matches[0]
    log(f"Converting GSE64810 BA9 bulk matrix: {path}")
    df = pd.read_csv(path, sep="\t", compression="gzip")
    df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    sample_cols = [c for c in df.columns if re.match(r"^[CH]_\d+", str(c))]
    if len(sample_cols) < 10:
        raise ValueError(f"Could not detect C_/H_ sample columns in {path}")
    x = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).T
    if float(x.max().max()) > 50:
        x = np.log1p(x)
    x.index = [f"GSE64810::{s}" for s in x.index]
    x.index.name = "pseudobulk_id"
    meta = pd.DataFrame(
        {
            "pseudobulk_id": x.index,
            "dataset_id": "GSE64810",
            "sample_id": [s.split("::", 1)[1] for s in x.index],
            "donor_id": x.index,
            "species": "human",
            "brain_region": "BA9",
            "cell_type": "bulk_BA9",
            "condition": ["HD" if "::H_" in s else "control" for s in x.index],
            "n_cells": 1,
            "matrix_source": str(path),
            "supervised_eligible_hint": True,
        }
    )
    return MatrixBundle(x=x, meta=meta)


def find_10x_h5_files() -> list[Path]:
    files = sorted((INTERIM / "extracted").rglob("*_feature_bc_matrix.h5"))
    if not files:
        files = sorted(RAW.rglob("*_feature_bc_matrix.h5"))
    filtered = [p for p in files if "filtered_feature_bc_matrix" in p.name]
    return filtered or files


def load_gse281069_metadata() -> pd.DataFrame:
    hits = list(RAW.rglob("GSE281069_seurat_metadata.csv.gz")) + list((INTERIM / "extracted").rglob("*metadata*.csv*"))
    frames = []
    for path in hits:
        try:
            df = read_csv_any(path)
            df["__metadata_source"] = str(path)
            frames.append(df)
            log(f"Loaded metadata {path} {df.shape}")
        except Exception as e:  # pragma: no cover - depends on user downloads
            log(f"Could not read metadata {path}: {e}")
    if not frames:
        return pd.DataFrame()
    # Prefer the GEO Seurat metadata if present; otherwise combine.
    return pd.concat(frames, ignore_index=True, sort=False)


def infer_col(columns: Iterable[str], patterns: list[str]) -> str | None:
    cols = list(columns)
    for pat in patterns:
        hits = [c for c in cols if re.search(pat, c, flags=re.I)]
        if hits:
            return hits[0]
    return None


def prepare_metadata_index(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if meta.empty:
        return meta, {}
    mapping = {
        "barcode": infer_col(meta.columns, [r"barcode", r"cell.*id", r"cell"]),
        "condition": infer_col(meta.columns, [r"condition", r"diagnosis", r"disease", r"status", r"group", r"genotype", r"dx"]),
        "donor": infer_col(meta.columns, [r"donor", r"individual", r"subject", r"patient", r"case"]),
        "sample": infer_col(meta.columns, [r"orig.ident", r"sample", r"library", r"specimen", r"gsm"]),
        "cell_type": infer_col(meta.columns, [r"cell.?type", r"subcluster", r"annotation", r"anno", r"cluster", r"class"]),
        "region": infer_col(meta.columns, [r"region", r"brain", r"area", r"tissue"]),
    }
    out = meta.copy()
    # If barcode is not an explicit column, row names were lost in CSV; still keep row number key.
    if mapping["barcode"]:
        out["__barcode_full"] = out[mapping["barcode"]].astype(str)
    else:
        out["__barcode_full"] = out.index.astype(str)
    out["__barcode_short"] = out["__barcode_full"].str.replace(r"-\d+$", "", regex=True)
    pd.DataFrame([mapping]).to_csv(TABLES / "gse281069_metadata_column_mapping.csv", index=False)
    pd.DataFrame({"column": list(meta.columns)}).to_csv(TABLES / "gse281069_metadata_columns.csv", index=False)
    meta.head(30).to_csv(TABLES / "gse281069_metadata_preview.csv", index=False)
    return out, mapping


def _read_10x_h5(path: Path):
    try:
        import scanpy as sc  # optional full dependency
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scanpy is required to read 10x .h5 matrices. Install with pip install -e '.[full]'.") from e
    adata = sc.read_10x_h5(path)
    adata.var_names_make_unique()
    return adata


def convert_gse281069_h5(max_cells_per_h5: int = 250_000, min_cells_per_pseudobulk: int = 30) -> MatrixBundle | None:
    h5_files = find_10x_h5_files()
    h5_files = [p for p in h5_files if "GSE281069" in str(p) or "GSM861" in p.name]
    if not h5_files:
        log("No GSE281069 10x H5 matrices found. Skipping H5 conversion.")
        return None
    log(f"Converting GSE281069 H5 matrices: {len(h5_files)} files")
    meta_raw = load_gse281069_metadata()
    meta, mapping = prepare_metadata_index(meta_raw)
    expr_rows: list[pd.Series] = []
    meta_rows: list[dict] = []
    conversion_rows = []

    for i, h5 in enumerate(h5_files, start=1):
        info = infer_from_filename(h5)
        log(f"[{i}/{len(h5_files)}] reading {h5.name}")
        try:
            adata = _read_10x_h5(h5)
        except Exception as e:
            log(f"  failed to read {h5}: {e}")
            conversion_rows.append({"path": str(h5), "status": "read_failed", "error": repr(e)})
            continue

        n_cells = adata.n_obs
        if n_cells == 0:
            continue
        if n_cells > max_cells_per_h5:
            idx = rng.choice(n_cells, size=max_cells_per_h5, replace=False)
            adata = adata[idx].copy()
            n_cells = adata.n_obs

        obs = pd.DataFrame({"barcode": adata.obs_names.astype(str)})
        obs["barcode_short"] = obs["barcode"].str.replace(r"-\d+$", "", regex=True)
        obs["condition"] = info["condition"]
        obs["donor_id"] = info["donor_id"]
        obs["sample_id"] = info["sample_id"]
        obs["brain_region"] = info["brain_region"]
        obs["cell_type"] = "snRNA_all_nuclei"
        obs["metadata_matched"] = False

        matched = False
        if not meta.empty and mapping:
            # Match on full barcode first, then barcode stripped of -1 suffix. Metadata may contain duplicates;
            # keep first annotation per barcode to prevent row inflation.
            for left_key, right_key in [("barcode", "__barcode_full"), ("barcode_short", "__barcode_short")]:
                ann = meta.drop_duplicates(right_key)
                merged = obs.merge(ann, left_on=left_key, right_on=right_key, how="left", suffixes=("", "_m"))
                condition_col = mapping.get("condition")
                match_frac = 0.0 if condition_col is None else merged[condition_col].notna().mean()
                if match_frac > 0.25:
                    obs = merged
                    if condition_col:
                        mapped_condition = obs[condition_col].map(standard_condition)
                        obs.loc[mapped_condition.isin(["HD", "control"]), "condition"] = mapped_condition[mapped_condition.isin(["HD", "control"])]
                    for target, col in [
                        ("donor_id", mapping.get("donor")),
                        ("sample_id", mapping.get("sample")),
                        ("cell_type", mapping.get("cell_type")),
                        ("brain_region", mapping.get("region")),
                    ]:
                        if col and col in obs.columns:
                            vals = obs[col].astype(str)
                            obs.loc[vals.notna() & (vals != "nan"), target] = vals[vals.notna() & (vals != "nan")]
                    obs["metadata_matched"] = True
                    matched = True
                    break

        # Keep known HD/control for supervised, but preserve unknown for atlas representation.
        group_cols = ["condition", "donor_id", "sample_id", "cell_type", "brain_region"]
        group_key = obs[group_cols].fillna("unknown").astype(str).agg("||".join, axis=1)
        counts_by_group = group_key.value_counts()
        keep_groups = counts_by_group[counts_by_group >= min_cells_per_pseudobulk].index.tolist()
        if not keep_groups:
            conversion_rows.append({"path": str(h5), "status": "too_few_cells", "n_cells": n_cells, "matched": matched})
            continue

        xmat = adata.X
        if not sparse.issparse(xmat):
            xmat = sparse.csr_matrix(xmat)
        genes = pd.Index(adata.var_names.astype(str))
        for group in keep_groups:
            cell_idx = np.where(group_key.values == group)[0]
            vec = np.asarray(xmat[cell_idx, :].sum(axis=0)).ravel()
            cond, donor, sample, cell_type, region = group.split("||")
            pb_id = "::".join(["GSE281069", safe_name(cond), safe_name(donor), safe_name(sample), safe_name(cell_type), safe_name(region)])
            # If duplicate group occurs across files, add filename stem.
            pb_id = f"{pb_id}::{safe_name(h5.stem)}"
            expr_rows.append(pd.Series(vec, index=genes, name=pb_id, dtype="float32"))
            meta_rows.append(
                {
                    "pseudobulk_id": pb_id,
                    "dataset_id": "GSE281069",
                    "sample_id": sample,
                    "donor_id": donor,
                    "species": "human",
                    "brain_region": region,
                    "cell_type": cell_type,
                    "condition": cond if cond in {"HD", "control"} else "unknown",
                    "n_cells": int(len(cell_idx)),
                    "matrix_source": str(h5),
                    "metadata_matched": bool(matched),
                    "supervised_eligible_hint": cond in {"HD", "control"},
                }
            )
        conversion_rows.append({"path": str(h5), "status": "converted", "n_cells": n_cells, "matched": matched, "n_groups": len(keep_groups)})
        del adata

    pd.DataFrame(conversion_rows).to_csv(TABLES / "gse281069_h5_conversion_manifest.csv", index=False)
    if not expr_rows:
        log("GSE281069 H5 conversion produced no pseudobulk groups.")
        return None
    x = pd.concat(expr_rows, axis=1).T.fillna(0.0)
    if float(x.max().max()) > 50:
        x = np.log1p(x)
    meta_df = pd.DataFrame(meta_rows)
    x.index.name = "pseudobulk_id"
    return MatrixBundle(x=x, meta=meta_df)


def convert_mouse_gse180294_csv(min_cells_hint: int = 1) -> MatrixBundle | None:
    """Best-effort mouse R6/2 conversion from extracted seurat_counts CSV files.

    These files are often gene x cell matrices without rich sample-level labels. We include them in
    atlas/latent analyses if they can be parsed, but supervised HD-vs-control uses only known labels.
    """
    files = sorted((INTERIM / "extracted" / "GSE180294_RAW").glob("*seurat_counts.csv.gz"))
    if not files:
        return None
    expr_rows = []
    meta_rows = []
    manifests = []
    log(f"Converting GSE180294 mouse count CSV files: {len(files)} files")
    for path in files:
        try:
            df = pd.read_csv(path, compression="gzip", index_col=0)
        except Exception as e:
            manifests.append({"path": str(path), "status": "read_failed", "error": repr(e)})
            continue
        # Assume rows are genes and columns are cells. If opposite, flip when there are far more rows than columns? 
        if df.shape[0] < df.shape[1]:
            genes = df.index.astype(str)
            vals = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        else:
            # Some exports are cell x gene; this keeps the larger gene dimension as columns.
            genes = df.columns.astype(str)
            vals = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=0)
        name = path.stem.replace(".csv", "")
        cond = standard_condition(name)
        pb_id = f"GSE180294::{safe_name(name)}"
        expr_rows.append(pd.Series(vals.values, index=genes, name=pb_id, dtype="float32"))
        meta_rows.append(
            {
                "pseudobulk_id": pb_id,
                "dataset_id": "GSE180294",
                "sample_id": name,
                "donor_id": name,
                "species": "mouse",
                "brain_region": "striatum" if re.search("striat", name, re.I) else "unknown_region",
                "cell_type": "mouse_snRNA_or_bulk_counts",
                "condition": cond,
                "n_cells": int(df.shape[1] if df.shape[0] < df.shape[1] else df.shape[0]),
                "matrix_source": str(path),
                "metadata_matched": False,
                "supervised_eligible_hint": cond in {"HD", "control"},
            }
        )
        manifests.append({"path": str(path), "status": "converted", "shape": str(df.shape), "condition": cond})
    pd.DataFrame(manifests).to_csv(TABLES / "gse180294_conversion_manifest.csv", index=False)
    if not expr_rows:
        return None
    x = pd.concat(expr_rows, axis=1).T.fillna(0.0)
    if float(x.max().max()) > 50:
        x = np.log1p(x)
    return MatrixBundle(x=x, meta=pd.DataFrame(meta_rows))


def merge_bundles(bundles: list[MatrixBundle], max_features: int = 8000, min_nonzero_frac: float = 0.03) -> MatrixBundle:
    if not bundles:
        raise RuntimeError("No datasets were converted into a matrix. Check reports/tables/downloaded_file_inventory.csv.")
    meta = pd.concat([b.meta for b in bundles], ignore_index=True, sort=False)
    # Use gene union, filling platform-specific missing genes with zero. This enables atlas-level representation;
    # supervised model filters to eligible labelled subsets later.
    x = pd.concat([b.x for b in bundles], axis=0, sort=True).fillna(0.0)
    x.index = meta["pseudobulk_id"].astype(str).values
    x.index.name = "pseudobulk_id"
    meta["condition"] = meta["condition"].map(standard_condition)
    meta["dataset_id"] = meta["dataset_id"].astype(str)
    meta["cell_type"] = meta["cell_type"].astype(str).replace({"nan": "unknown_celltype"})
    meta["brain_region"] = meta["brain_region"].astype(str).replace({"nan": "unknown_region"})
    meta["donor_id"] = meta["donor_id"].astype(str)

    nonzero_frac = (x != 0).mean(axis=0)
    x = x.loc[:, nonzero_frac >= min_nonzero_frac]
    variances = x.var(axis=0)
    x = x.loc[:, variances > 0]
    if x.shape[1] > max_features:
        top = x.var(axis=0).sort_values(ascending=False).head(max_features).index
        x = x[top]
    return MatrixBundle(x=x, meta=meta)


def write_matrix_outputs(bundle: MatrixBundle) -> None:
    bundle.x.to_parquet(PROCESSED / "atlas_feature_matrix.parquet")
    bundle.meta.to_parquet(PROCESSED / "atlas_sample_metadata.parquet", index=False)
    # Keep repo-compatible names too.
    bundle.x.to_parquet(PROCESSED / "feature_matrix.parquet")
    bundle.meta.to_parquet(PROCESSED / "sample_metadata.parquet", index=False)
    bundle.x.to_parquet(PROCESSED / "pseudobulk_expression.parquet")
    bundle.meta.to_parquet(PROCESSED / "pseudobulk_metadata.parquet", index=False)


def dataset_audits(bundle: MatrixBundle) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = bundle.meta.copy()
    audit = pd.DataFrame(
        [
            {
                "n_pseudobulk_samples": len(bundle.x),
                "n_features": bundle.x.shape[1],
                "n_hd": int((meta["condition"] == "HD").sum()),
                "n_control": int((meta["condition"] == "control").sum()),
                "n_unknown_condition": int((meta["condition"] == "unknown").sum()),
                "datasets": ";".join(sorted(meta["dataset_id"].unique())),
                "n_datasets": int(meta["dataset_id"].nunique()),
                "n_cell_types": int(meta["cell_type"].nunique()),
                "n_regions": int(meta["brain_region"].nunique()),
                "n_donors": int(meta["donor_id"].nunique()),
                "n_duplicate_feature_rows": int(bundle.x.duplicated().sum()),
                "n_missing_values": int(bundle.x.isna().sum().sum()),
            }
        ]
    )
    comp = meta.groupby(["dataset_id", "condition"], dropna=False).size().reset_index(name="n")
    cell_comp = meta.groupby(["dataset_id", "cell_type", "condition"], dropna=False).size().reset_index(name="n")
    audit.to_csv(TABLES / "atlas_dataset_audit.csv", index=False)
    comp.to_csv(TABLES / "atlas_dataset_condition_composition.csv", index=False)
    cell_comp.to_csv(TABLES / "atlas_celltype_condition_composition.csv", index=False)
    return audit, comp, cell_comp


def supervised_subset(bundle: MatrixBundle) -> MatrixBundle:
    meta = bundle.meta.copy()
    labelled = meta["condition"].isin(["HD", "control"])
    # Only keep datasets where both classes exist. This prevents the classic trap:
    # BA9 controls + glial HD-only -> model learns dataset/platform rather than disease.
    eligible_dataset = (
        meta[labelled].groupby("dataset_id")["condition"].nunique().loc[lambda s: s >= 2].index.tolist()
    )
    keep = labelled & meta["dataset_id"].isin(eligible_dataset)
    sub_meta = meta.loc[keep].reset_index(drop=True)
    sub_x = bundle.x.loc[keep.values].copy()
    sub_x.index = sub_meta["pseudobulk_id"].astype(str).values
    sub_x.index.name = "pseudobulk_id"
    sub_x.to_parquet(PROCESSED / "supervised_feature_matrix.parquet")
    sub_meta.to_parquet(PROCESSED / "supervised_sample_metadata.parquet", index=False)
    pd.DataFrame({"eligible_dataset_id": eligible_dataset}).to_csv(TABLES / "supervised_eligible_datasets.csv", index=False)
    return MatrixBundle(x=sub_x, meta=sub_meta)


def y_from_meta(meta: pd.DataFrame) -> pd.Series:
    return (meta["condition"].astype(str) == "HD").astype(int)


def safe_auc(y_true, y_score) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true, y_score) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def make_models() -> dict[str, tuple[Pipeline, dict[str, list]]]:
    return {
        "logistic_regularized": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)),
                ]
            ),
            {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
                ]
            ),
            {"clf__n_estimators": [300, 700], "clf__max_depth": [None, 3, 5], "clf__min_samples_leaf": [1, 2, 4]},
        ),
        "hist_gradient_boosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", HistGradientBoostingClassifier(random_state=SEED)),
                ]
            ),
            {
                "clf__max_iter": [100, 250],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__l2_regularization": [0.0, 0.1, 1.0],
                "clf__max_leaf_nodes": [7, 15, 31],
            },
        ),
    }


def nested_cv(supervised: MatrixBundle, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, meta = supervised.x, supervised.meta
    if len(x) < 10 or meta["condition"].nunique() < 2:
        empty = pd.DataFrame()
        empty.to_csv(TABLES / "nested_cv_summary.csv", index=False)
        empty.to_csv(TABLES / "nested_cv_metrics.csv", index=False)
        empty.to_csv(TABLES / "nested_cv_predictions.csv", index=False)
        return empty, empty, empty
    y = y_from_meta(meta).reset_index(drop=True)
    groups = meta["donor_id"].astype(str).reset_index(drop=True)
    x = x.reset_index(drop=True)
    min_class = int(y.value_counts().min())
    n_groups = int(groups.nunique())
    n_outer = max(2, min(n_splits, min_class, n_groups))
    n_inner = max(2, min(3, min_class, n_groups))
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=SEED)
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=SEED + 1)
    rows = []
    preds = []

    for model_name, (pipe, grid) in make_models().items():
        log(f"Nested CV train/tune/test: {model_name}")
        for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups)):
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
            search.fit(x.iloc[train_idx], y.iloc[train_idx], groups=groups.iloc[train_idx])
            train_prob = search.predict_proba(x.iloc[train_idx])[:, 1]
            test_prob = search.predict_proba(x.iloc[test_idx])[:, 1]
            test_pred = (test_prob >= 0.5).astype(int)
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "inner_best_roc_auc": float(search.best_score_),
                    "train_roc_auc": safe_auc(y.iloc[train_idx], train_prob),
                    "test_roc_auc": safe_auc(y.iloc[test_idx], test_prob),
                    "auc_gap": safe_auc(y.iloc[train_idx], train_prob) - safe_auc(y.iloc[test_idx], test_prob),
                    "test_average_precision": safe_ap(y.iloc[test_idx], test_prob),
                    "test_accuracy": float(accuracy_score(y.iloc[test_idx], test_pred)),
                    "test_balanced_accuracy": float(balanced_accuracy_score(y.iloc[test_idx], test_pred)),
                    "test_f1": float(f1_score(y.iloc[test_idx], test_pred, zero_division=0)),
                    "test_brier": float(brier_score_loss(y.iloc[test_idx], test_prob)),
                    "best_params": json.dumps(search.best_params_),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
            )
            for idx, prob in zip(test_idx, test_prob):
                rec = meta.iloc[idx].to_dict()
                rec.update({"model": model_name, "fold": fold, "sample_index": int(idx), "y_true": int(y.iloc[idx]), "y_prob": float(prob)})
                preds.append(rec)
        final_model = clone(pipe).fit(x, y)
        joblib.dump(final_model, MODELS / f"{model_name}_final.joblib")

    metrics = pd.DataFrame(rows)
    pred_df = pd.DataFrame(preds)
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
    metrics.to_csv(TABLES / "nested_cv_metrics.csv", index=False)
    pred_df.to_csv(TABLES / "nested_cv_predictions.csv", index=False)
    summary.to_csv(TABLES / "nested_cv_summary.csv", index=False)
    return metrics, pred_df, summary


def optuna_tuning(supervised: MatrixBundle, n_trials: int = 100) -> pd.DataFrame:
    try:
        import optuna
    except Exception:
        (TABLES / "optuna_error.txt").write_text("Optuna is not installed. Install with pip install -e '.[full]'.\n")
        return pd.DataFrame()
    if len(supervised.x) < 10 or supervised.meta["condition"].nunique() < 2:
        return pd.DataFrame()
    x = supervised.x.reset_index(drop=True)
    y = y_from_meta(supervised.meta).reset_index(drop=True)
    groups = supervised.meta["donor_id"].astype(str).reset_index(drop=True)
    min_class = int(y.value_counts().min())
    n_groups = int(groups.nunique())
    n_splits = max(2, min(5, min_class, n_groups))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    def objective(trial):
        model_type = trial.suggest_categorical("model_type", ["logistic", "rf", "hgb"])
        if model_type == "logistic":
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=trial.suggest_float("C", 1e-3, 10, log=True), class_weight="balanced", solver="lbfgs", max_iter=5000)),
                ]
            )
        elif model_type == "rf":
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=trial.suggest_int("n_estimators", 200, 900, step=100),
                            max_depth=trial.suggest_categorical("max_depth", [None, 3, 5, 8]),
                            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=SEED,
                        ),
                    ),
                ]
            )
        else:
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                            max_leaf_nodes=trial.suggest_categorical("max_leaf_nodes", [7, 15, 31]),
                            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 3.0, log=True),
                            max_iter=trial.suggest_int("max_iter", 80, 300, step=20),
                            random_state=SEED,
                        ),
                    ),
                ]
            )
        scores = []
        for tr, te in cv.split(x, y, groups):
            model.fit(x.iloc[tr], y.iloc[tr])
            prob = model.predict_proba(x.iloc[te])[:, 1]
            scores.append(safe_auc(y.iloc[te], prob))
        return float(np.nanmean(scores))

    study = optuna.create_study(direction="maximize", study_name="neuroglia_hd_optuna")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials = study.trials_dataframe()
    trials.to_csv(TABLES / "optuna_trials.csv", index=False)
    (TABLES / "best_optuna_model.txt").write_text(f"Best ROC-AUC: {study.best_value}\nBest params: {study.best_params}\n")
    return trials


def permutation_control(supervised: MatrixBundle, n_permutations: int = 100) -> pd.DataFrame:
    if len(supervised.x) < 10 or supervised.meta["condition"].nunique() < 2:
        return pd.DataFrame()
    x = supervised.x.reset_index(drop=True)
    y = y_from_meta(supervised.meta).reset_index(drop=True)
    groups = supervised.meta["donor_id"].astype(str).reset_index(drop=True)
    min_class = int(y.value_counts().min())
    n_groups = int(groups.nunique())
    n_splits = max(2, min(5, min_class, n_groups))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=5000)),
        ]
    )
    rows = []
    for perm in range(n_permutations):
        y_perm = pd.Series(rng.permutation(y.values))
        probs = np.zeros(len(y_perm))
        for tr, te in cv.split(x, y_perm, groups):
            m = clone(pipe)
            m.fit(x.iloc[tr], y_perm.iloc[tr])
            probs[te] = m.predict_proba(x.iloc[te])[:, 1]
        rows.append({"permutation": perm, "roc_auc": safe_auc(y_perm, probs), "average_precision": safe_ap(y_perm, probs)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "permutation_label_control.csv", index=False)
    return out


def leakage_screen(supervised: MatrixBundle) -> pd.DataFrame:
    if len(supervised.x) < 10 or supervised.meta["condition"].nunique() < 2:
        return pd.DataFrame()
    x = supervised.x.reset_index(drop=True)
    y = y_from_meta(supervised.meta).reset_index(drop=True)
    rows = []
    for col in x.columns:
        s = pd.to_numeric(x[col], errors="coerce").reset_index(drop=True)
        if s.nunique(dropna=True) < 2:
            continue
        mask = s.notna().reset_index(drop=True)
        if int(mask.sum()) < 10 or y.loc[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y.loc[mask], s.loc[mask])
        rows.append(
            {
                "feature": col,
                "single_feature_auc_directionless": float(max(auc, 1 - auc)),
                "raw_auc": float(auc),
                "n_unique": int(s.nunique(dropna=True)),
                "missing_fraction": float(1 - mask.mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("single_feature_auc_directionless", ascending=False) if rows else pd.DataFrame()
    out.to_csv(TABLES / "single_feature_leakage_screen.csv", index=False)
    return out


def dataset_proxy_screen(supervised: MatrixBundle) -> pd.DataFrame:
    meta = supervised.meta.copy()
    if meta.empty or meta["condition"].nunique() < 2:
        out = pd.DataFrame()
        out.to_csv(TABLES / "dataset_label_proxy_screen.csv", index=False)
        return out
    rows = []
    for col in ["dataset_id", "brain_region", "cell_type", "matrix_source"]:
        if col not in meta.columns:
            continue
        tab = pd.crosstab(meta[col].astype(str), meta["condition"].astype(str))
        for level, row in tab.iterrows():
            n = int(row.sum())
            frac_hd = float(row.get("HD", 0) / n) if n else float("nan")
            rows.append({"metadata_field": col, "level": level, "n": n, "hd_fraction": frac_hd})
    out = pd.DataFrame(rows).sort_values(["metadata_field", "n"], ascending=[True, False]) if rows else pd.DataFrame()
    out.to_csv(TABLES / "dataset_label_proxy_screen.csv", index=False)
    return out


def pca_atlas(bundle: MatrixBundle) -> pd.DataFrame:
    x = bundle.x.copy()
    if x.shape[1] > 2000:
        top = x.var(axis=0).sort_values(ascending=False).head(2000).index
        x = x[top]
    z = SimpleImputer(strategy="median").fit_transform(x)
    z = StandardScaler().fit_transform(z)
    pcs = PCA(n_components=2, random_state=SEED).fit_transform(z)
    out = bundle.meta[["pseudobulk_id", "dataset_id", "condition", "cell_type", "brain_region"]].copy()
    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]
    out.to_csv(TABLES / "atlas_pca_coordinates.csv", index=False)
    return out


def make_figures(bundle: MatrixBundle, supervised: MatrixBundle, pred_df: pd.DataFrame, summary: pd.DataFrame, perm: pd.DataFrame, leakage: pd.DataFrame) -> None:
    comp = pd.read_csv(TABLES / "atlas_dataset_condition_composition.csv")
    if not comp.empty:
        pivot = comp.pivot_table(index="dataset_id", columns="condition", values="n", aggfunc="sum", fill_value=0)
        plt.figure(figsize=(8, 5))
        pivot.plot(kind="bar", ax=plt.gca())
        plt.title("Atlas dataset composition")
        plt.ylabel("pseudobulk samples")
        savefig(FIGURES / "atlas_dataset_composition.png")

    coords = pca_atlas(bundle)
    if not coords.empty:
        plt.figure(figsize=(7, 6))
        for cond, df in coords.groupby("condition"):
            plt.scatter(df["PC1"], df["PC2"], label=cond, alpha=0.75)
        plt.title("Atlas PCA by condition")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(fontsize=8)
        savefig(FIGURES / "atlas_pca_condition.png")
        plt.figure(figsize=(7, 6))
        for ds, df in coords.groupby("dataset_id"):
            plt.scatter(df["PC1"], df["PC2"], label=ds, alpha=0.75)
        plt.title("Atlas PCA by dataset")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(fontsize=8)
        savefig(FIGURES / "atlas_pca_dataset.png")

    if pred_df is not None and not pred_df.empty:
        plt.figure(figsize=(7, 6))
        for model, df in pred_df.groupby("model"):
            if df["y_true"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
            auc = roc_auc_score(df["y_true"], df["y_prob"])
            plt.plot(fpr, tpr, label=f"{model} AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle=":")
        plt.title("Nested CV ROC curves")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend(fontsize=8)
        savefig(FIGURES / "nested_cv_roc_curves.png")

        plt.figure(figsize=(7, 6))
        for model, df in pred_df.groupby("model"):
            if df["y_true"].nunique() < 2:
                continue
            precision, recall, _ = precision_recall_curve(df["y_true"], df["y_prob"])
            ap = average_precision_score(df["y_true"], df["y_prob"])
            plt.plot(recall, precision, label=f"{model} AP={ap:.3f}")
        plt.title("Nested CV precision-recall curves")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(fontsize=8)
        savefig(FIGURES / "nested_cv_precision_recall_curves.png")

        plt.figure(figsize=(7, 6))
        for model, df in pred_df.groupby("model"):
            bins = min(8, max(3, len(df) // 10))
            frac_pos, mean_pred = calibration_curve(df["y_true"], df["y_prob"], n_bins=bins, strategy="quantile")
            plt.plot(mean_pred, frac_pos, marker="o", label=model)
        plt.plot([0, 1], [0, 1], linestyle=":")
        plt.title("Calibration curves")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed positive fraction")
        plt.legend(fontsize=8)
        savefig(FIGURES / "calibration_curves.png")

        for model, df in pred_df.groupby("model"):
            y_pred = (df["y_prob"] >= 0.5).astype(int)
            cm = confusion_matrix(df["y_true"], y_pred, labels=[0, 1])
            plt.figure(figsize=(5, 4))
            plt.imshow(cm)
            plt.title(f"Confusion matrix: {model}")
            plt.xticks([0, 1], ["pred control", "pred HD"])
            plt.yticks([0, 1], ["true control", "true HD"])
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center")
            savefig(FIGURES / f"confusion_matrix_{safe_name(model)}.png")

    if summary is not None and not summary.empty and "mean_auc_gap" in summary.columns:
        ordered = summary.sort_values("mean_auc_gap")
        plt.figure(figsize=(8, 4.8))
        bars = plt.barh(ordered["model"], ordered["mean_auc_gap"])
        plt.axvline(0.02, linestyle=":", label="low concern")
        plt.axvline(0.08, linestyle="--", label="high concern")
        for bar, val in zip(bars, ordered["mean_auc_gap"]):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center")
        plt.title("Overfitting audit: train-test ROC-AUC gap")
        plt.xlabel("mean train ROC-AUC - mean test ROC-AUC")
        plt.legend(fontsize=8)
        savefig(FIGURES / "overfitting_auc_gap_labelled.png")

    if perm is not None and not perm.empty:
        plt.figure(figsize=(7, 5))
        plt.hist(perm["roc_auc"].dropna(), bins=20)
        plt.axvline(0.5, linestyle=":")
        plt.title("Permutation-label control")
        plt.xlabel("ROC-AUC after shuffled labels")
        plt.ylabel("count")
        savefig(FIGURES / "permutation_label_control.png")

    if leakage is not None and not leakage.empty:
        top = leakage.head(25).iloc[::-1]
        plt.figure(figsize=(8, max(5, len(top) * 0.28)))
        plt.barh(top["feature"], top["single_feature_auc_directionless"])
        plt.axvline(0.98, linestyle="--")
        plt.title("Single-feature separation screen")
        plt.xlabel("directionless single-feature ROC-AUC")
        savefig(FIGURES / "single_feature_leakage_screen.png")


def write_report(bundle: MatrixBundle, supervised: MatrixBundle, summary: pd.DataFrame, perm: pd.DataFrame, leakage: pd.DataFrame, proxy: pd.DataFrame, optuna_trials: pd.DataFrame) -> None:
    audit = pd.read_csv(TABLES / "atlas_dataset_audit.csv")
    comp = pd.read_csv(TABLES / "atlas_dataset_condition_composition.csv")
    warnings_list = []
    n_datasets = int(audit["n_datasets"].iloc[0]) if not audit.empty else 0
    if n_datasets < 2:
        warnings_list.append("Only one dataset converted. Do not describe this as a full-scale multi-dataset run.")
    if len(supervised.x) < 30:
        warnings_list.append("Supervised labelled subset is small or unavailable; train/test metrics are not sufficiently stable.")
    if not comp.empty:
        for ds, df in comp.groupby("dataset_id"):
            known = df[df["condition"].isin(["HD", "control"])]
            if len(known) == 1:
                warnings_list.append(f"Dataset {ds} has only one supervised class and is excluded from supervised HD/control training.")
    if summary is not None and not summary.empty and summary["mean_auc_gap"].max() > 0.08:
        warnings_list.append("At least one model shows a high train-test ROC-AUC gap > 0.08.")
    if perm is not None and not perm.empty and perm["roc_auc"].mean() > 0.65:
        warnings_list.append("Permutation-label control is above expected chance; inspect split design and leakage.")
    if leakage is not None and not leakage.empty and leakage["single_feature_auc_directionless"].iloc[0] > 0.98:
        warnings_list.append("A single feature nearly separates the label; inspect whether this is biology, platform artefact, or leakage.")
    if not warnings_list:
        warnings_list.append("No automatic red-flag threshold triggered; independent validation remains required.")

    def md(df: pd.DataFrame, n: int = 20) -> str:
        if df is None or df.empty:
            return "_No rows._"
        return df.head(n).to_markdown(index=False)

    fig_files = sorted(str(p) for p in FIGURES.glob("*.png"))
    table_files = sorted(str(p) for p in TABLES.glob("*.csv"))
    lines = [
        "# NeuroGlia-HD Atlas Research-Grade Big Pipeline Report",
        "",
        "## Run status",
        "This report is generated by a conservative pipeline. Atlas-level conversions and supervised modelling are separated so that single-class datasets cannot create dataset-label leakage.",
        "",
        "## Atlas dataset audit",
        md(audit),
        "",
        "## Dataset and condition composition",
        md(comp, 50),
        "",
        "## Supervised nested cross-validation summary",
        md(summary),
        "",
        "## Optuna tuning",
        f"Optuna trials: {0 if optuna_trials is None else len(optuna_trials)}",
        "",
        "## Permutation-label control",
        md(perm.describe().reset_index() if perm is not None and not perm.empty else pd.DataFrame()),
        "",
        "## Single-feature leakage screen",
        md(leakage, 25),
        "",
        "## Metadata proxy / confounding screen",
        md(proxy, 40),
        "",
        "## Automatic issue flags",
    ]
    lines.extend([f"- {w}" for w in warnings_list])
    lines.extend(["", "## Figures"])
    lines.extend([f"- `{f}`" for f in fig_files])
    lines.extend(["", "## Tables"])
    lines.extend([f"- `{t}`" for t in table_files])
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "Strong HD/control discrimination should be interpreted only after nested tuning, permutation controls, calibration, single-feature screens, and independent/cohort-level validation are reviewed. Near-perfect AUC is an audit trigger, not a finish line.",
            "",
        ]
    )
    (REPORTS / "research_grade_big_report.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    if args.extract:
        extract_archives()
    inventory_files()
    bundles: list[MatrixBundle] = []
    if args.include_ba9:
        b = convert_gse64810_ba9()
        if b is not None:
            bundles.append(b)
    if args.include_human_glia:
        b = convert_gse281069_h5(max_cells_per_h5=args.max_cells_per_h5, min_cells_per_pseudobulk=args.min_cells_per_pseudobulk)
        if b is not None:
            bundles.append(b)
    if args.include_mouse:
        b = convert_mouse_gse180294_csv()
        if b is not None:
            bundles.append(b)
    atlas = merge_bundles(bundles, max_features=args.max_features, min_nonzero_frac=args.min_nonzero_frac)
    write_matrix_outputs(atlas)
    dataset_audits(atlas)
    supervised = supervised_subset(atlas)
    pd.DataFrame(
        [
            {
                "n_supervised_samples": len(supervised.x),
                "n_supervised_features": supervised.x.shape[1] if len(supervised.x) else 0,
                "n_hd": int((supervised.meta["condition"] == "HD").sum()) if len(supervised.meta) else 0,
                "n_control": int((supervised.meta["condition"] == "control").sum()) if len(supervised.meta) else 0,
                "datasets": ";".join(sorted(supervised.meta["dataset_id"].unique())) if len(supervised.meta) else "",
            }
        ]
    ).to_csv(TABLES / "supervised_dataset_audit.csv", index=False)

    metrics, pred_df, summary = nested_cv(supervised, n_splits=args.n_splits)
    optuna_trials = optuna_tuning(supervised, n_trials=args.n_trials) if args.n_trials > 0 else pd.DataFrame()
    perm = permutation_control(supervised, n_permutations=args.n_permutations) if args.n_permutations > 0 else pd.DataFrame()
    leakage = leakage_screen(supervised)
    proxy = dataset_proxy_screen(supervised)
    make_figures(atlas, supervised, pred_df, summary, perm, leakage)
    write_report(atlas, supervised, summary, perm, leakage, proxy, optuna_trials)
    log("\nDone.")
    log("Main report: reports/research_grade_big_report.md")
    log("Dataset audit: reports/tables/atlas_dataset_audit.csv")
    log("Supervised audit: reports/tables/supervised_dataset_audit.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the research-grade NeuroGlia-HD big pipeline.")
    p.add_argument("--extract", action="store_true", help="Extract downloaded GEO tar archives before conversion.")
    p.add_argument("--include-ba9", action="store_true", default=True)
    p.add_argument("--include-human-glia", action="store_true", default=True)
    p.add_argument("--include-mouse", action="store_true", default=True)
    p.add_argument("--max-features", type=int, default=int(os.getenv("MAX_FEATURES", "8000")))
    p.add_argument("--min-nonzero-frac", type=float, default=float(os.getenv("MIN_NONZERO_FRAC", "0.03")))
    p.add_argument("--max-cells-per-h5", type=int, default=int(os.getenv("MAX_CELLS_PER_H5", "250000")))
    p.add_argument("--min-cells-per-pseudobulk", type=int, default=int(os.getenv("MIN_CELLS_PER_PSEUDOBULK", "30")))
    p.add_argument("--n-splits", type=int, default=int(os.getenv("N_SPLITS", "5")))
    p.add_argument("--n-trials", type=int, default=int(os.getenv("N_TRIALS", "100")))
    p.add_argument("--n-permutations", type=int, default=int(os.getenv("N_PERMUTATIONS", "100")))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
