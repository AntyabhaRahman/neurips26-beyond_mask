import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

from beyond_mask.openrouter import OpenRouterClient, ChatResult

load_dotenv()

ARCHETYPES = [
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "known_facts",
    "provided_facts",
    "statistics",
]

BELIEF_K: dict[str, dict[str, int]] = {
    "statistics": {"1": 3},
    "continuations": {"1": 3, "2": 1, "3": 1},
    "disinformation": {"1": 3, "2": 1, "3": 1},
    "doubling_down_known_facts": {"1": 3, "2": 1, "3": 1},
    "known_facts": {"1": 3, "2": 1, "3": 1},
    "provided_facts": {"1": 3, "2": 1, "3": 1},
}


def load_archetype(archetype: str, limit: int | None) -> pd.DataFrame:
    ds = load_dataset("cais/MASK", archetype, split="test")
    df = ds.to_pandas()
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    if "belief_elicit_1" in df.columns and "belief_system_1" not in df.columns:
        df["belief_system_1"] = "You are a helpful assistant."
    return df


def build_tasks(df: pd.DataFrame, archetype: str, lie_k: int) -> tuple[list, list, list]:
    """Return (prompts, row_indices, prompt_type_labels) for all calls needed."""
    belief_k = BELIEF_K[archetype]
    is_doubling_down = (
        "dd_turn_1_prompt" in df.columns
        and "dd_turn_1_response" in df.columns
        and df["dd_turn_1_prompt"].notna().any()
    )

    prompts: list[list[dict]] = []
    indices: list[int] = []
    prompt_types: list[str] = []

    for idx, row in df.iterrows():
        for run in range(lie_k):
            if is_doubling_down:
                msg = [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["dd_turn_1_prompt"]},
                    {"role": "assistant", "content": row["dd_turn_1_response"]},
                    {"role": "user", "content": row["user_prompt"]},
                ]
            else:
                msg = [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_prompt"]},
                ]
            prompts.append(msg)
            indices.append(idx)
            prompt_types.append(f"lying_run{run + 1}")

        if "belief_elicit_1" in df.columns:
            for be_num, k_value in belief_k.items():
                belief_col = f"belief_elicit_{be_num}"
                if belief_col in df.columns and pd.notna(row.get(belief_col)):
                    system = row.get("belief_system_1") or "You are a helpful assistant."
                    for run in range(k_value):
                        prompts.append([
                            {"role": "system", "content": system},
                            {"role": "user", "content": row[belief_col]},
                        ])
                        indices.append(idx)
                        prompt_types.append(f"be{be_num}_run{run + 1}")

    return prompts, indices, prompt_types


def column_name(prompt_type: str, lie_k: int, belief_k: dict[str, int]) -> str:
    for run in range(lie_k):
        if prompt_type == f"lying_run{run + 1}":
            return f"generation(System Prompt + User Prompt)_run{run + 1}"
    # belief elicitation
    parts = prompt_type.split("_run")
    be_num = parts[0][len("be"):]
    run_n = parts[1]
    return f"generation(belief_system_1 + belief_elicit_{be_num})_run{run_n}"


async def process_archetype(
    client: OpenRouterClient,
    model: str,
    archetype: str,
    output_csv: Path,
    lie_k: int,
    max_tokens: int,
    temperature: float,
    cache_dir: Path,
    limit: int | None,
) -> None:
    print(f"Loading {archetype} from HuggingFace cais/MASK ...")
    df = load_archetype(archetype, limit)
    print(f"  {len(df)} rows")

    prompts, indices, prompt_types = build_tasks(df, archetype, lie_k)

    print(f"  Dispatching {len(prompts)} calls ...")
    results: list[ChatResult] = await asyncio.gather(*[
        client.chat(
            model,
            msg,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
        )
        for msg in prompts
    ])

    belief_k = BELIEF_K[archetype]
    calls_meta = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    error_count = 0

    for result, idx, ptype in zip(results, indices, prompt_types):
        col = column_name(ptype, lie_k, belief_k)
        if result.error is not None:
            df.at[idx, col] = f"[ERROR: {result.error}]"
            error_count += 1
        else:
            df.at[idx, col] = result.text

        total_prompt_tokens += result.prompt_tokens or 0
        total_completion_tokens += result.completion_tokens or 0
        calls_meta.append({
            "row_idx": int(idx),
            "prompt_type": ptype,
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "error": result.error,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"  Saved {output_csv}")

    meta = {
        "model": model,
        "archetype": archetype,
        "split": "test",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "lie_k": lie_k,
        "row_count": len(df),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "error_count": error_count,
        "calls": calls_meta,
    }
    meta_path = output_csv.parent / (output_csv.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved {meta_path}")


async def main(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set. Add it to .env or the environment.", file=sys.stderr)
        sys.exit(1)

    archetypes = args.archetypes
    for a in archetypes:
        if a not in ARCHETYPES:
            print(f"ERROR: Unknown archetype '{a}'. Valid: {ARCHETYPES}", file=sys.stderr)
            sys.exit(1)

    model_slug = args.model.split("/")[-1]
    base_dir = Path("test_csv_data" if args.test else "csv_data") / "responses"
    cache_dir = Path(args.cache_dir)
    limit = 5 if args.test else args.limit

    async with OpenRouterClient(api_key=api_key, concurrency=args.concurrency) as client:
        for archetype in archetypes:
            output_csv = base_dir / f"{archetype}_{model_slug}.csv"
            if output_csv.exists():
                print(f"Skipping {output_csv} (already exists)")
                continue
            await process_archetype(
                client=client,
                model=args.model,
                archetype=archetype,
                output_csv=output_csv,
                lie_k=args.lie_k,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                cache_dir=cache_dir,
                limit=limit,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MASK benchmark responses via OpenRouter")
    parser.add_argument("--model", required=True, help="OpenRouter model ID, e.g. google/gemini-2.5-flash-lite")
    parser.add_argument("--archetypes", nargs="+", default=ARCHETYPES, metavar="ARCHETYPE",
                        help=f"Archetypes to run (default: all). Choices: {ARCHETYPES}")
    parser.add_argument("--lie-k", type=int, default=1, dest="lie_k",
                        help="Number of lying prompt runs per row (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=1000, dest="max_tokens",
                        help="Max tokens for generation (default: 1000)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent API calls (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows per archetype (optional)")
    parser.add_argument("--test", action="store_true",
                        help="Use first 5 rows per archetype; write to test_csv_data/")
    parser.add_argument("--cache-dir", default=".cache/openrouter", dest="cache_dir",
                        help="Disk cache directory (default: .cache/openrouter)")

    args = parser.parse_args()
    asyncio.run(main(args))
