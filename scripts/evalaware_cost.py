from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from beyond_mask.evalaware import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_SAMPLES_PER_CONFIG,
    DEFAULT_TARGET_MODELS,
    RESULTS_ROOT,
    dataclass_to_dict,
    estimate_cost,
    estimate_judge_cost,
    filter_rows_for_scope,
    load_evalaware_rows,
    load_openrouter_catalog,
    resolve_requested_models,
)


def write_markdown(path: Path, payload: dict) -> None:
    lines = ["# EvalAwareBench Cost Estimate", ""]
    for scope, estimate in payload["scopes"].items():
        lines.extend(
            [
                f"## {scope}",
                "",
                f"- Target calls: {estimate['target']['target_calls']:,}",
                f"- Judge calls: {estimate['judge']['target_calls']:,}",
                f"- Standard total: ${estimate['total']['standard_cost_usd']:.2f}",
                f"- Requested-flex total: ${estimate['total']['requested_flex_cost_usd']:.2f}",
                f"- Conservative pre-run total: ${estimate['total']['confirmed_flex_cost_usd']:.2f}",
                "",
                "| Model | Calls | Standard | Requested flex | Confirmed flex |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for section in ("target", "judge"):
            for model, model_estimate in estimate[section]["by_model"].items():
                lines.append(
                    "| "
                    f"{section}:{model} | {model_estimate['target_calls']:,} | "
                    f"${model_estimate['standard_cost_usd']:.2f} | "
                    f"${model_estimate['requested_flex_cost_usd']:.2f} | "
                    f"${model_estimate['confirmed_flex_cost_usd']:.2f} |"
                )
        lines.append("")
    path.write_text("\n".join(lines))


def combine_estimates(target: dict, judge: dict) -> dict:
    return {
        "standard_cost_usd": target["standard_cost_usd"] + judge["standard_cost_usd"],
        "requested_flex_cost_usd": (
            target["requested_flex_cost_usd"] + judge["requested_flex_cost_usd"]
        ),
        "confirmed_flex_cost_usd": (
            target["confirmed_flex_cost_usd"] + judge["confirmed_flex_cost_usd"]
        ),
        "estimated_prompt_tokens": (
            target["estimated_prompt_tokens"] + judge["estimated_prompt_tokens"]
        ),
        "estimated_completion_tokens": (
            target["estimated_completion_tokens"]
            + judge["estimated_completion_tokens"]
        ),
        "total_calls": target["target_calls"] + judge["target_calls"],
    }


def build_cost_payload(
    *,
    rows: list,
    catalog: dict,
    samples_per_config: int,
    max_completion_tokens: int,
    judge_model: str,
    judge_max_tokens: int,
) -> dict:
    resolved = resolve_requested_models(
        catalog,
        required=[*DEFAULT_TARGET_MODELS, judge_model],
    )
    if resolved.unavailable_required:
        missing = ", ".join(resolved.unavailable_required)
        raise SystemExit(f"Required OpenRouter model(s) unavailable: {missing}")

    models = [model for model in DEFAULT_TARGET_MODELS if model in resolved.available]
    scoped_rows = filter_rows_for_scope(rows, "paper")
    target = asdict(
        estimate_cost(
            rows=scoped_rows,
            models=models,
            prices=resolved.available,
            samples_per_config=samples_per_config,
            max_completion_tokens=max_completion_tokens,
        )
    )
    judge = asdict(
        estimate_judge_cost(
            rows=scoped_rows,
            target_models=models,
            judge_model=judge_model,
            prices=resolved.available,
            samples_per_config=samples_per_config,
            max_target_completion_tokens=max_completion_tokens,
            max_judge_completion_tokens=judge_max_tokens,
        )
    )
    return {
        "models": models,
        "judge_model": judge_model,
        "unavailable_optional": resolved.unavailable_optional,
        "samples_per_config": samples_per_config,
        "max_completion_tokens": max_completion_tokens,
        "judge_max_tokens": judge_max_tokens,
        "scopes": {
            "paper": {
                "target": target,
                "judge": judge,
                "total": combine_estimates(target, judge),
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate EvalAwareBench OpenRouter cost.")
    parser.add_argument("--samples-per-config", type=int, default=DEFAULT_SAMPLES_PER_CONFIG)
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-max-tokens", type=int, default=1200)
    parser.add_argument("--out", default=str(RESULTS_ROOT / "cost_estimate.json"))
    args = parser.parse_args()

    payload = build_cost_payload(
        rows=load_evalaware_rows(),
        catalog=load_openrouter_catalog(),
        samples_per_config=args.samples_per_config,
        max_completion_tokens=args.max_completion_tokens,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=dataclass_to_dict))
    write_markdown(out_path.with_suffix(".md"), payload)
    print(json.dumps(payload, indent=2, default=dataclass_to_dict))


if __name__ == "__main__":
    main()
