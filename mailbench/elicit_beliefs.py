"""Elicit model beliefs about each scenario proposition (MASK-style repeated runs).

For every scenario x proposition x model we ask a neutral model what it believes
about the ground-truth proposition, decoupled from the pressured episode. The first
(direct) belief prompt is sampled DIRECT_K times at temperature 1.0 with distinct
seeds; each follow-up (consistency) prompt is sampled CONSISTENCY_K times. Every raw
response is then classified by a judge model into A/B/C and aggregated into a single
true/false/unsure belief label via beyond_mask.mailenv.beliefs.aggregate_belief.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from beyond_mask.mailenv.beliefs import (
    BELIEF_JUDGE_PROMPT,
    CONSISTENCY_K,
    DIRECT_K,
    aggregate_belief,
    parse_answer_letter,
)
from beyond_mask.mailenv.scenario import load_scenario
from beyond_mask.results import git_sha, model_slug

try:
    from mailbench._common import (
        CACHE_DIR,
        RESULTS_ROOT,
        SCENARIO_DIR,
        make_client,
        run_root,
    )
except ModuleNotFoundError:
    from _common import CACHE_DIR, RESULTS_ROOT, SCENARIO_DIR, make_client, run_root

BELIEF_SYSTEM = "You are a helpful assistant."
JUDGE_SEED = 0
DIRECT_SEED_BASE = 1000


async def elicit_one(
    client,
    model: str,
    judge_model: str,
    proposition_text: str,
    prompt: str,
    seed: int,
) -> dict:
    """Sample the subject model once, then classify the response with the judge."""
    result = await client.chat(
        model,
        [
            {"role": "system", "content": BELIEF_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        max_tokens=1024,
        cache_dir=CACHE_DIR,
        seed=seed,
    )
    response = result.text or ""
    judge = await client.chat(
        judge_model,
        [
            {"role": "system", "content": BELIEF_SYSTEM},
            {
                "role": "user",
                "content": BELIEF_JUDGE_PROMPT.format(
                    proposition=proposition_text, response=response
                ),
            },
        ],
        temperature=0.0,
        max_tokens=256,
        cache_dir=CACHE_DIR,
        seed=JUDGE_SEED,
    )
    judge_raw = judge.text or ""
    return {
        "response": response,
        "judge_raw": judge_raw,
        "letter": parse_answer_letter(judge_raw),
    }


async def elicit_for_proposition(
    client,
    model: str,
    judge_model: str,
    proposition_text: str,
    prompts: tuple[str, ...],
) -> dict:
    """Run the direct + consistency prompts for one proposition and aggregate."""
    runs: list[dict] = []
    direct_letters: list[str] = []
    consistency_letters: list[str] = []
    for prompt_index, prompt in enumerate(prompts):
        if prompt_index == 0:
            for run in range(DIRECT_K):
                seed = DIRECT_SEED_BASE + run
                out = await elicit_one(
                    client, model, judge_model, proposition_text, prompt, seed
                )
                runs.append(
                    {"prompt_index": prompt_index, "run": run, "seed": seed, **out}
                )
                if out["letter"] is not None:
                    direct_letters.append(out["letter"])
        else:
            for run in range(CONSISTENCY_K):
                seed = DIRECT_SEED_BASE + run
                out = await elicit_one(
                    client, model, judge_model, proposition_text, prompt, seed
                )
                runs.append(
                    {"prompt_index": prompt_index, "run": run, "seed": seed, **out}
                )
                if out["letter"] is not None:
                    consistency_letters.append(out["letter"])
    belief = aggregate_belief(direct_letters, consistency_letters)
    return {"belief": belief, "runs": runs}


async def main(args: argparse.Namespace) -> None:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_id = (
        "test"
        if args.test
        else (args.run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    )
    root = run_root(RESULTS_ROOT, run_id)
    beliefs_dir = root / "beliefs"
    paths = sorted(Path(args.scenarios).glob("*.yaml"))
    if args.test:
        paths = paths[:1]
    client = make_client()
    totals = {"models": 0, "skipped": 0, "propositions": 0}
    try:
        for model in models:
            out_path = beliefs_dir / f"{model_slug(model)}.json"
            if out_path.exists():
                print(f"skip {out_path.name} (exists)")
                totals["skipped"] += 1
                continue
            by_scenario: dict[str, dict] = {}
            for path in paths:
                scenario = load_scenario(path)
                prop_text = {p.id: p.proposition for p in scenario.ground_truth}
                by_prop: dict[str, dict] = {}
                for prop_id, prompts in scenario.belief_elicitation.items():
                    by_prop[prop_id] = await elicit_for_proposition(
                        client, model, args.judge_model, prop_text[prop_id], prompts
                    )
                    totals["propositions"] += 1
                by_scenario[scenario.id] = by_prop
            beliefs_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(by_scenario, indent=2))
            totals["models"] += 1
            print(f"{out_path.name}: {len(by_scenario)} scenario(s)")
    finally:
        await client.aclose()
    root.mkdir(parents=True, exist_ok=True)
    (root / "beliefs.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "models": models,
                "judge_model": args.judge_model,
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
    parser.add_argument("--judge-model", default="google/gemini-3.1-flash-lite")
    parser.add_argument("--scenarios", default=str(SCENARIO_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--test", action="store_true", help="run only the first scenario; run_id=test"
    )
    asyncio.run(main(parser.parse_args()))
