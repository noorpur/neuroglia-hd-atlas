from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neuroglia_hd.analysis.figures import save_metric_bar
from neuroglia_hd.analysis.report import write_markdown_report
from neuroglia_hd.config import get_seed, load_config, resolve_paths
from neuroglia_hd.data.geo import download_geo_record
from neuroglia_hd.data.registry import get_record, list_records
from neuroglia_hd.features.pseudobulk import attach_signature_features
from neuroglia_hd.features.signatures import (
    load_gene_sets,
    score_signatures,
    select_top_variable_genes,
)
from neuroglia_hd.logging_utils import setup_logging
from neuroglia_hd.models.baselines import run_grouped_baselines


def cmd_registry(args: argparse.Namespace) -> None:
    records = [r.__dict__ for r in list_records()]
    df = pd.DataFrame(records)
    print(df[["dataset_id", "accession", "organism", "modality", "role", "primary_url"]].to_string(index=False))


def cmd_download(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    logger = setup_logging(Path("logs") / "download.log")
    dataset_ids = args.datasets or list(cfg.get("datasets", {}).keys())
    manifests = []
    for dataset_id in dataset_ids:
        record = get_record(dataset_id)
        logger.info("Downloading %s (%s)", dataset_id, record.accession)
        manifests.append(download_geo_record(record, paths.raw_dir, overwrite=args.overwrite))
    if manifests:
        out = pd.concat(manifests, ignore_index=True)
        out.to_csv(paths.tables_dir / "download_manifest.csv", index=False)
        logger.info("Wrote %s", paths.tables_dir / "download_manifest.csv")


def _load_processed_matrix(paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = paths.processed_dir / "feature_matrix.parquet"
    meta_path = paths.processed_dir / "sample_metadata.parquet"
    if not feature_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Expected data/processed/feature_matrix.parquet and sample_metadata.parquet. "
            "Run pseudobulk construction first, or place prepared matrices there."
        )
    return pd.read_parquet(feature_path), pd.read_parquet(meta_path)


def cmd_pseudobulk(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    logger = setup_logging(Path("logs") / "pseudobulk.log")
    logger.info("Pseudobulk construction requires parsed AnnData or flat expression matrices.")
    logger.info("This command creates a feature manifest and checks for existing processed matrices.")
    expr_path = paths.processed_dir / "pseudobulk_expression.parquet"
    meta_path = paths.processed_dir / "pseudobulk_metadata.parquet"
    if not expr_path.exists() or not meta_path.exists():
        template = pd.DataFrame(
            columns=["pseudobulk_id", "dataset_id", "donor_id", "brain_region", "cell_type", "condition", "n_cells"]
        )
        template.to_csv(paths.processed_dir / "pseudobulk_metadata_TEMPLATE.csv", index=False)
        raise FileNotFoundError(
            "No pseudobulk matrices found yet. Use src/neuroglia_hd/features/pseudobulk.py after loading AnnData, "
            "or export pseudobulk_expression.parquet and pseudobulk_metadata.parquet into data/processed/. "
            "A metadata template has been written."
        )
    expr = pd.read_parquet(expr_path)
    meta = pd.read_parquet(meta_path)
    n_top = int(cfg.get("features", {}).get("top_variable_genes", 2000))
    genes = select_top_variable_genes(expr, n_top)
    expr_top = expr[genes]
    gene_sets = load_gene_sets(cfg.get("features", {}).get("signature_file"))
    sig = score_signatures(expr, gene_sets)
    feature_matrix = attach_signature_features(expr_top, sig)
    feature_matrix.to_parquet(paths.processed_dir / "feature_matrix.parquet")
    meta.to_parquet(paths.processed_dir / "sample_metadata.parquet")
    coverage = sig.attrs.get("coverage")
    if coverage is not None:
        coverage.to_csv(paths.tables_dir / "signature_coverage.csv", index=False)
    logger.info("Wrote feature matrix with shape %s", feature_matrix.shape)


def cmd_train_baselines(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    seed = get_seed(cfg)
    logger = setup_logging(Path("logs") / "train_baselines.log")
    x, meta = _load_processed_matrix(paths)
    condition_col = args.target_col
    group_col = cfg.get("splitting", {}).get("group_field", "donor_id")
    if condition_col not in meta.columns:
        raise KeyError(f"Target column {condition_col!r} not found in metadata.")
    if group_col not in meta.columns:
        raise KeyError(f"Group column {group_col!r} not found in metadata.")
    y = meta[condition_col].astype(str).str.lower().isin(["hd", "huntington", "disease", "case", "r6/2", "r62", "1", "true"]).astype(int)
    groups = meta[group_col].astype(str)
    model_names = cfg.get("models", {}).get("baselines", ["logistic_elasticnet", "random_forest"])
    runs = run_grouped_baselines(x, y, groups, model_names, n_splits=int(cfg.get("splitting", {}).get("n_splits", 5)), random_state=seed, model_dir=str(paths.model_dir))
    metrics = pd.DataFrame([{"model": r.name, **r.metrics} for r in runs])
    preds = pd.concat([r.predictions for r in runs], ignore_index=True)
    metrics.to_csv(paths.tables_dir / "baseline_metrics.csv", index=False)
    preds.to_csv(paths.tables_dir / "baseline_predictions.csv", index=False)
    save_metric_bar(metrics, paths.figures_dir / "baseline_model_comparison.png", metric="roc_auc")
    logger.info("Wrote baseline metrics and predictions.")


def cmd_train_latent(args: argparse.Namespace) -> None:
    from neuroglia_hd.models.latent import train_autoencoder
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    seed = get_seed(cfg)
    x, meta = _load_processed_matrix(paths)
    latent_cfg = cfg.get("models", {}).get("latent", {})
    result = train_autoencoder(
        x,
        latent_dim=int(latent_cfg.get("latent_dim", 16)),
        hidden_dim=int((latent_cfg.get("hidden_dims") or [128])[0]),
        epochs=int(latent_cfg.get("epochs", 100)),
        batch_size=int(latent_cfg.get("batch_size", 64)),
        learning_rate=float(latent_cfg.get("learning_rate", 1e-3)),
        device=cfg.get("compute", {}).get("device", "auto"),
        seed=seed,
    )
    result.embedding.to_parquet(paths.processed_dir / "latent_embedding.parquet")
    result.reconstruction_error.to_csv(paths.tables_dir / "latent_reconstruction_error.csv", index=False)
    result.history.to_csv(paths.tables_dir / "latent_training_history.csv", index=False)
    meta.to_parquet(paths.processed_dir / "latent_metadata.parquet")
    print(f"Wrote latent embedding: {result.embedding.shape}")


def cmd_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    metrics_path = paths.tables_dir / "baseline_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    write_markdown_report(paths.reports_dir / "analysis_report.md", metrics=metrics)
    print(f"Wrote {paths.reports_dir / 'analysis_report.md'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurogliahd", description="NeuroGlia-HD Atlas command-line interface")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("registry", help="Show dataset registry")
    s.set_defaults(func=cmd_registry)

    s = sub.add_parser("download", help="Download GEO supplementary files")
    s.add_argument("--config", default="configs/default.yaml")
    s.add_argument("--datasets", nargs="*", default=None)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("pseudobulk", help="Build feature matrices from pseudobulk expression")
    s.add_argument("--config", default="configs/default.yaml")
    s.set_defaults(func=cmd_pseudobulk)

    s = sub.add_parser("train-baselines", help="Train grouped baseline models")
    s.add_argument("--config", default="configs/default.yaml")
    s.add_argument("--target-col", default="condition")
    s.set_defaults(func=cmd_train_baselines)

    s = sub.add_parser("train-latent", help="Train neural latent model")
    s.add_argument("--config", default="configs/default.yaml")
    s.set_defaults(func=cmd_train_latent)

    s = sub.add_parser("report", help="Generate markdown report")
    s.add_argument("--config", default="configs/default.yaml")
    s.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
