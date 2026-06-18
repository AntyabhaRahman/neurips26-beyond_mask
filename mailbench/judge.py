"""Judge mailbench episodes: classify the assistant's final email per proposition.

For each episode produced by run_episodes.py we locate the final email sent to the
receiver, then for every ground-truth proposition we ask a judge model to classify
how that email treats the fact (accurate / distorted / fabricated / omitted / evaded)
using a strict JSON schema. The behavior label is combined with whether the agent was
exposed to the evidence and with the model's elicited belief into a single outcome via
beyond_mask.mailenv.judging.combine_outcome. Episodes that never reached the receiver
get an outcome with behavior=None and spend no API money.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from beyond_mask.mailenv.judging import (
    BEHAVIOR_JUDGE_PROMPT,
    BEHAVIOR_SCHEMA,
    combine_outcome,
    parse_behavior_json,
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

JUDGE_SYSTEM = "You are a helpful assistant."
JUDGE_SEED = 0


def load_events(ep_dir: Path) -> list[dict]:
    text = (ep_dir / "events.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def find_final_email(events: list[dict], final_email_id: str | None) -> dict | None:
    if not final_email_id:
        return None
    for e in events:
        if (
            e.get("event") == "agent_email"
            and e["email"].get("email_id") == final_email_id
        ):
            return e["email"]
    return None


def exposed_propositions(events: list[dict]) -> set[str]:
    return {e["proposition"] for e in events if e.get("event") == "exposure"}


async def judge_episode(
    client,
    judge_model: str,
    ep_dir: Path,
    scenario,
    beliefs: dict,
) -> dict:
    """Classify each proposition for one episode, returning {prop_id: judgment}."""
    summary = json.loads((ep_dir / "episode.json").read_text())
    events = load_events(ep_dir)
    end_reason = summary["end_reason"]
    final_email = find_final_email(events, summary.get("final_email_id"))
    exposed = exposed_propositions(events)
    scenario_beliefs = beliefs.get(scenario.id, {})

    out: dict[str, dict] = {}
    for prop in scenario.ground_truth:
        belief = scenario_beliefs.get(prop.id, {}).get("belief", "unsure")
        is_exposed = prop.id in exposed
        behavior: str | None = None
        assertion = None
        evidence_quote = None
        judge_raw = None
        if final_email is not None:
            prompt = BEHAVIOR_JUDGE_PROMPT.format(
                proposition=prop.proposition,
                subject=final_email.get("subject", ""),
                body=final_email.get("text_body", ""),
            )
            result = await client.chat(
                judge_model,
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                cache_dir=CACHE_DIR,
                seed=JUDGE_SEED,
                response_format=BEHAVIOR_SCHEMA,
            )
            judge_raw = result.text or ""
            parsed = parse_behavior_json(judge_raw)
            if parsed is not None:
                behavior = parsed["behavior"]
                assertion = parsed.get("assertion")
                evidence_quote = parsed.get("evidence_quote")
        outcome = combine_outcome(
            end_reason=end_reason,
            exposed=is_exposed,
            world_knowledge=prop.world_knowledge,
            behavior=behavior,
            belief=belief,
        )
        out[prop.id] = {
            "behavior": behavior,
            "assertion": assertion,
            "evidence_quote": evidence_quote,
            "belief": belief,
            "exposed": is_exposed,
            "outcome": outcome,
            "judge_raw": judge_raw,
        }
    return out


def scenarios_by_id(scenario_dir: Path) -> dict[str, Path]:
    """Map each scenario id to its yaml path (loaded once at base variant)."""
    mapping: dict[str, Path] = {}
    for path in sorted(scenario_dir.glob("*.yaml")):
        mapping[load_scenario(path).id] = path
    return mapping


async def main(args: argparse.Namespace) -> None:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run_id = (
        "test"
        if args.test
        else (args.run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    )
    root = run_root(RESULTS_ROOT, run_id)
    episodes_root = root / "episodes"
    judgments_dir = root / "judgments"
    beliefs_dir = root / "beliefs"
    id_to_path = scenarios_by_id(Path(args.scenarios))

    client = make_client()
    totals = {"models": 0, "skipped": 0, "episodes": 0, "errors": 0}
    try:
        for model in models:
            slug = model_slug(model)
            out_path = judgments_dir / f"{slug}.json"
            if out_path.exists():
                print(f"skip {out_path.name} (exists)")
                totals["skipped"] += 1
                continue
            beliefs_path = beliefs_dir / f"{slug}.json"
            beliefs = (
                json.loads(beliefs_path.read_text()) if beliefs_path.exists() else {}
            )
            suffix = f"__{slug}"
            ep_dirs = (
                sorted(
                    p
                    for p in episodes_root.iterdir()
                    if p.is_dir() and p.name.endswith(suffix)
                )
                if episodes_root.exists()
                else []
            )
            by_episode: dict[str, dict] = {}
            for ep_dir in ep_dirs:
                try:
                    summary = json.loads((ep_dir / "episode.json").read_text())
                    scenario = load_scenario(
                        id_to_path[summary["scenario"]],
                        variant=summary.get("variant"),
                    )
                    by_episode[ep_dir.name] = await judge_episode(
                        client, args.judge_model, ep_dir, scenario, beliefs
                    )
                    totals["episodes"] += 1
                    print(
                        f"{ep_dir.name}: judged {len(by_episode[ep_dir.name])} prop(s)"
                    )
                except Exception as exc:
                    print(f"ERROR {ep_dir.name}: {exc}")
                    totals["errors"] += 1
            judgments_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(by_episode, indent=2))
            totals["models"] += 1
    finally:
        await client.aclose()
    root.mkdir(parents=True, exist_ok=True)
    (root / "judge.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "models": models,
                "judge_model": args.judge_model,
                "git_sha": git_sha(),
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
    parser.add_argument("--test", action="store_true", help="judge the 'test' run_id")
    asyncio.run(main(parser.parse_args()))
