"""Run mailbench episodes: scenarios x variants x models -> results/mailbench/<run_id>/."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from beyond_mask.mailenv.episode import run_episode
from beyond_mask.mailenv.scenario import load_scenario
from beyond_mask.results import git_sha

from _common import (
    CACHE_DIR,
    RESULTS_ROOT,
    SCENARIO_DIR,
    episode_dir_name,
    make_client,
    run_root,
)


async def main(args: argparse.Namespace) -> None:
    run_id = (
        "test"
        if args.test
        else (args.run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    )
    root = run_root(RESULTS_ROOT, run_id)
    paths = sorted(Path(args.scenarios).glob("*.yaml"))
    if args.test:
        paths = paths[:1]
    jobs = []
    for path in paths:
        base = load_scenario(path)
        for variant in (None, *base.variant_names):
            for model in args.models.split(","):
                jobs.append((path, variant, model))
    client = make_client()
    totals = {"episodes": 0, "skipped": 0, "cost_usd": 0.0}
    try:
        for path, variant, model in jobs:
            scenario = load_scenario(path, variant=variant)
            out_dir = root / "episodes" / episode_dir_name(scenario.id, variant, model)
            if (out_dir / "events.jsonl").exists():
                print(f"skip {out_dir.name} (exists)")
                totals["skipped"] += 1
                continue
            result = await run_episode(
                client, model, scenario, out_dir, cache_dir=CACHE_DIR
            )
            totals["episodes"] += 1
            totals["cost_usd"] += result.cost_usd
            print(
                f"{out_dir.name}: {result.end_reason} in {result.turns} turns "
                f"(${result.cost_usd:.4f})"
            )
    finally:
        await client.aclose()
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "models": args.models.split(","),
                "git_sha": git_sha(),
                "scenario_files": [p.name for p in paths],
                **totals,
            },
            indent=2,
        )
    )
    print(json.dumps(totals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", required=True, help="comma-separated OpenRouter model ids"
    )
    parser.add_argument("--scenarios", default=str(SCENARIO_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--test", action="store_true", help="run only the first scenario; run_id=test"
    )
    asyncio.run(main(parser.parse_args()))
