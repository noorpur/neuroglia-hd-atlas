from __future__ import annotations

import gzip
import hashlib
import re
import shutil
import urllib.request
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from tqdm import tqdm

from neuroglia_hd.data.registry import DatasetRecord


def geo_series_prefix(accession: str) -> str:
    """Return GEO series FTP prefix, e.g. GSE281069 -> GSE281nnn."""
    if not re.match(r"^GSE\d+$", accession):
        raise ValueError(f"Expected GEO series accession like GSE12345, got {accession}")
    return accession[:-3] + "nnn"


def geo_supplementary_url(accession: str) -> str:
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_series_prefix(accession)}/{accession}/suppl/"


def list_geo_supplementary(accession: str) -> list[str]:
    """List supplementary-file URLs for a GEO series FTP directory."""
    base = geo_supplementary_url(accession)
    with urllib.request.urlopen(base, timeout=60) as response:  # noqa: S310 public scientific data
        html = response.read().decode("utf-8", errors="replace")
    hrefs = re.findall(r'href="([^"]+)"', html)
    urls = []
    for href in hrefs:
        if href in {"../", "/"} or href.startswith("?"):
            continue
        if any(href.lower().endswith(ext) for ext in (".gz", ".txt", ".tsv", ".csv", ".h5", ".h5ad", ".mtx", ".tar", ".zip", ".rds", ".xlsx")):
            urls.append(urljoin(base, href))
    return sorted(set(urls))


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def download_url(url: str, destination: Path, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 public scientific data
        total = int(response.headers.get("Content-Length", 0))
        with destination.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as bar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    return destination


def maybe_decompress_gzip(path: Path, keep_original: bool = True) -> Path:
    if path.suffix != ".gz":
        return path
    output = path.with_suffix("")
    if output.exists():
        return output
    with gzip.open(path, "rb") as f_in, output.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    if not keep_original:
        path.unlink()
    return output


def download_geo_record(record: DatasetRecord, raw_dir: Path, overwrite: bool = False) -> pd.DataFrame:
    """Download all supplementary files for a dataset record and return a manifest."""
    dataset_dir = raw_dir / record.dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    urls = list_geo_supplementary(record.accession)
    rows = []
    for url in urls:
        fname = url.rstrip("/").split("/")[-1]
        local = download_url(url, dataset_dir / fname, overwrite=overwrite)
        rows.append(
            {
                **asdict(record),
                "url": url,
                "local_path": str(local),
                "bytes": local.stat().st_size,
                "sha256": sha256_file(local),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = dataset_dir / "download_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest
