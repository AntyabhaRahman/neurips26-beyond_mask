from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from beyond_mask.openrouter import OpenRouterClient
from beyond_mask.results import model_slug

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "mailbench/.cache/openrouter"
RESULTS_ROOT = REPO_ROOT / "results"
SCENARIO_DIR = REPO_ROOT / "mailbench/scenarios/handwritten"


def make_client(concurrency: int = 8) -> OpenRouterClient:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing (expected in repo-root .env)")
    return OpenRouterClient(api_key=key, concurrency=concurrency)


def episode_dir_name(scenario_id: str, variant: str | None, model: str) -> str:
    slug = f"{scenario_id}@{variant}" if variant else scenario_id
    return f"{slug}__{model_slug(model)}"


def run_root(results_root: Path, run_id: str) -> Path:
    return results_root / "mailbench" / run_id
