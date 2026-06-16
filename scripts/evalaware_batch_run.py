from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from beyond_mask.evalaware import (
    DEFAULT_SAMPLES_PER_CONFIG,
    DEFAULT_SERVICE_TIER,
    RESULTS_ROOT,
    build_work_items,
    completed_ids_from_jsonl,
    filter_rows_for_scope,
    load_openrouter_catalog,
    load_evalaware_rows,
    resolve_requested_models,
    config_name,
    upsert_jsonl,
    write_jsonl,
)
from beyond_mask.openrouter import OpenRouterClient

load_dotenv()


def parse_models(value: str | None) -> list[str]:
    if value:
        return [model.strip() for model in value.split(",") if model.strip()]
    resolved = resolve_requested_models(load_openrouter_catalog())
    if resolved.unavailable_required:
        missing = ", ".join(resolved.unavailable_required)
        raise SystemExit(f"Required OpenRouter model(s) unavailable: {missing}")
    if resolved.unavailable_optional:
        print(
            json.dumps(
                {"unavailable_optional": resolved.unavailable_optional},
                indent=2,
            )
        )
    return list(resolved.available)


def target_result_record(item, result) -> dict:
    return {
        "custom_id": item.custom_id,
        "kind": item.kind,
        "model": item.model,
        "row": asdict(item.row),
        "task_id": item.row.task_id,
        "task_name": item.row.task_name,
        "valence": item.row.valence,
        "config_name": config_name(item.row),
        "sample_idx": item.sample_idx,
        "prompt": item.row.prompt,
        "response": result.text,
        "reasoning": result.reasoning,
        "reasoning_details": result.reasoning_details,
        "service_tier_requested": item.service_tier,
        "service_tier": result.service_tier,
        "cache_hit": result.cached,
        "request_hash": result.request_hash,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.cost_usd,
        "error": result.error,
    }


async def fetch_target_result(client: OpenRouterClient, item, args: argparse.Namespace):
    result = await client.chat(
        item.model,
        item.messages,
        temperature=args.temperature,
        max_tokens=args.max_completion_tokens,
        cache_dir=RESULTS_ROOT / "cache" / "target",
        reasoning={"effort": "high"},
        service_tier=item.service_tier,
    )
    return item, result


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set.")

    rows = filter_rows_for_scope(load_evalaware_rows(), args.scope)
    if args.limit:
        rows = rows[: args.limit]
    models = parse_models(args.models)
    out_path = RESULTS_ROOT / "responses" / f"{args.scope}.jsonl"
    completed = completed_ids_from_jsonl(out_path)
    items = build_work_items(
        rows=rows,
        models=models,
        samples_per_config=args.samples_per_config,
        service_tier=args.service_tier,
        kind="target",
        max_tokens=args.max_completion_tokens,
        completed_ids=completed,
        force=args.force,
    )

    print(
        json.dumps(
            {
                "scope": args.scope,
                "models": models,
                "rows": len(rows),
                "queued": len(items),
                "completed": len(completed),
                "output": str(out_path),
            },
            indent=2,
        )
    )

    async with OpenRouterClient(
        api_key=api_key,
        concurrency=args.concurrency,
        timeout=args.timeout,
    ) as client:
        for start in range(0, len(items), args.chunk_size):
            chunk = items[start : start + args.chunk_size]
            tasks = [fetch_target_result(client, item, args) for item in chunk]
            saved_in_chunk = 0
            for completed in asyncio.as_completed(tasks):
                item, result = await completed
                record = target_result_record(item, result)
                if args.force:
                    upsert_jsonl(out_path, [record])
                else:
                    write_jsonl(out_path, [record])
                saved_in_chunk += 1
                print(
                    f"saved {min(start + saved_in_chunk, len(items))}/{len(items)}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvalAwareBench target calls.")
    parser.add_argument("--scope", choices=("paper", "full"), default="paper")
    parser.add_argument("--models", default=None, help="comma-separated model ids")
    parser.add_argument("--samples-per-config", type=int, default=DEFAULT_SAMPLES_PER_CONFIG)
    parser.add_argument("--service-tier", default=DEFAULT_SERVICE_TIER)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
