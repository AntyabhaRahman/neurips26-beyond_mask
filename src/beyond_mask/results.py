from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import polars as pl

_SLUG_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _slugify(s: str) -> str:
    return _SLUG_SAFE.sub("-", s).strip("-._")


def dataset_slug(dataset_id: str, config: str | None, split: str) -> str:
    parts = [dataset_id.replace("/", "__"), config or "", split]
    return "__".join(_slugify(p) for p in parts if p)


def model_slug(model_id: str) -> str:
    return _slugify(model_id.replace("/", "__"))


def run_paths(
    results_root: Path,
    dataset_slug: str,
    model_slug: str,
    n: int,
    started_at: datetime,
) -> tuple[Path, Path]:
    folder = results_root / dataset_slug
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{started_at.strftime('%Y%m%d-%H%M%S')}__{model_slug}__n{n}"
    return folder / f"{stem}.parquet", folder / f"{stem}.json"


def write_run(
    df: pl.DataFrame,
    metadata: dict,
    parquet_path: Path,
    json_path: Path,
) -> None:
    df.write_parquet(parquet_path)
    json_path.write_text(json.dumps(metadata, indent=2, default=str))


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None
