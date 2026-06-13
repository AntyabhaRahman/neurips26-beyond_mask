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

try:
    from mailbench._common import (
        CACHE_DIR,
        RESULTS_ROOT,
        SCENARIO_DIR,
        episode_dir_name,
        make_client,
        run_root,
    )
except ModuleNotFoundError:
    from _common import (
        CACHE_DIR,
        RESULTS_ROOT,
        SCENARIO_DIR,
        episode_dir_name,
        make_client,
        run_root,
    )


async def main(args: argparse.Namespace) -> None:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
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
            for model in models:
                jobs.append((path, variant, model))
    client = make_client()
    totals = {"episodes": 0, "skipped": 0, "errors": 0, "cost_usd": 0.0}
    try:
        for path, variant, model in jobs:
            out_name = f"{path.stem}/{variant}/{model}"
            try:
                scenario = load_scenario(path, variant=variant)
                out_dir = (
                    root / "episodes" / episode_dir_name(scenario.id, variant, model)
                )
                out_name = out_dir.name
                if (out_dir / "events.jsonl").exists():
                    print(f"skip {out_name} (exists)")
                    totals["skipped"] += 1
                    continue
                result = await run_episode(
                    client, model, scenario, out_dir, cache_dir=CACHE_DIR
                )
                totals["episodes"] += 1
                totals["cost_usd"] += result.cost_usd
                print(
                    f"{out_name}: {result.end_reason} in {result.turns} turns "
                    f"(${result.cost_usd:.4f})"
                )
            except Exception as exc:
                print(f"ERROR {out_name}: {exc}")
                totals["errors"] += 1
    finally:
        await client.aclose()
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "models": models,
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
