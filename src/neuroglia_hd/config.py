from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path
    tables_dir: Path
    model_dir: Path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_paths(config: dict[str, Any], root: str | Path | None = None) -> ProjectPaths:
    """Resolve project directories from configuration."""
    root_path = Path(root or ".").resolve()
    data_dir = root_path / config.get("project", {}).get("data_dir", "data")
    reports_dir = root_path / config.get("project", {}).get("output_dir", "reports")
    paths = ProjectPaths(
        root=root_path,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        interim_dir=data_dir / "interim",
        processed_dir=data_dir / "processed",
        reports_dir=reports_dir,
        figures_dir=reports_dir / "figures",
        tables_dir=reports_dir / "tables",
        model_dir=root_path / "models",
    )
    for p in paths.__dict__.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return paths


def get_seed(config: dict[str, Any]) -> int:
    return int(config.get("project", {}).get("random_seed", 42))
