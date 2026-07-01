"""Run the MASK judge via OpenAI's Batch API (50% cheaper than live).

The org's gpt-5.5 batch quota caps *concurrent in-flight* tokens at 900k, so the
~11.6M-token job cannot go in one batch. This splits it into ~280k-token chunks
and keeps the pipe full at ~840k: it submits chunks concurrently up to that
budget and tops up as each completes (no waiting on a single batch). Chunks that
come back failed (e.g. a transient token-limit while an earlier batch drains) are
requeued. Results are persisted per chunk so a crash resumes instead of re-paying.

Prompt-building is reused from evaluate.py's BATCH_COLLECT mode (no duplication).
Run from mask/mask (cwd needs csv_data/ + prompts/):

    uv run python -u ../../scripts/run_mask_judge_batch.py --run
    uv run python    ../../scripts/run_mask_judge_batch.py --build   # rebuild CSVs from accum
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tiktoken
from openai import OpenAI

sys.path.insert(0, os.getcwd())  # evaluate.py + prompts/ live in cwd (mask/mask)
import evaluate  # noqa: E402

BATCH_DIR = Path("csv_data/batch")
INDEX_JSON = (
    BATCH_DIR / "judge_index.json"
)  # compact id -> "<file>::<idx>::<task_type>"
PROGRESS_JSON = (
    BATCH_DIR / "progress.json"
)  # {"done": [chunk_i...], "batch_ids": {...}}
ACCUM_JSONL = BATCH_DIR / "accum.jsonl"  # {"cid","numerical","content"} per result
RESPONSES_DIR = Path("csv_data/responses")
EVALUATED_DIR = Path("csv_data/evaluated")
MODEL_SLUG = "claude-opus-4-8"
ARCHETYPES = [
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "known_facts",
    "provided_facts",
    "statistics",
]

# Sized by input + max-output budget so totals never exceed the 900k org cap
# regardless of how the API counts. ~280k chunks let ~3 run at once under 840k.
CHUNK_BUDGET = 280_000
INFLIGHT_BUDGET = 840_000
ASSUMED_OUTPUT = 2000  # for requests without explicit max_completion_tokens (numerical)
POLL_SECONDS = 30
MAX_RETRIES = 5  # per chunk, before giving up
TERMINAL = {"completed", "failed", "expired", "cancelled"}
_ENC = tiktoken.get_encoding("o200k_base")


def configure(model_slug: str, archetypes: list[str], batch_dir: Path) -> None:
    global MODEL_SLUG, ARCHETYPES, BATCH_DIR
    global INDEX_JSON, PROGRESS_JSON, ACCUM_JSONL

    MODEL_SLUG = model_slug
    ARCHETYPES = archetypes
    BATCH_DIR = batch_dir
    INDEX_JSON = BATCH_DIR / "judge_index.json"
    PROGRESS_JSON = BATCH_DIR / "progress.json"
    ACCUM_JSONL = BATCH_DIR / "accum.jsonl"


def collect_requests() -> list[dict]:
    """Walk the opus response files in evaluate.py's collect mode to gather bodies."""
    evaluate.BATCH_COLLECT = True
    evaluate.BATCH_REQUESTS = []
    for archetype in ARCHETYPES:
        path = RESPONSES_DIR / f"{archetype}_{MODEL_SLUG}.csv"
        if not path.exists():
            print(f"skip missing {path}")
            continue
        # concurrency>1 skips run_row_tasks' per-task 2s sleep; no API calls happen here.
        asyncio.run(evaluate.process_file(str(path), None, concurrency_limit=64))
    return evaluate.BATCH_REQUESTS


def request_tokens(body: dict) -> int:
    inp = sum(len(_ENC.encode(m["content"])) for m in body["messages"])
    return inp + body.get("max_completion_tokens", ASSUMED_OUTPUT)


def build_compact_requests() -> list[dict]:
    """Collect, sort, assign stable compact custom_ids (r0..rN), write the index.

    Sorting by the descriptive id makes r0..rN (and the index) identical across
    runs, so resumed/accumulated results always map back correctly. The index is
    independent of chunk boundaries, so re-chunking never invalidates accum.jsonl.
    """
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    requests = collect_requests()
    if not requests:
        sys.exit("no requests collected")
    requests.sort(key=lambda r: r["custom_id"])
    index, compact = {}, []
    for n, req in enumerate(requests):
        cid = f"r{n}"
        index[cid] = req["custom_id"]
        compact.append({**req, "custom_id": cid})
    INDEX_JSON.write_text(json.dumps(index))
    return compact


def chunk_list(requests: list[dict], budget: int) -> list[list[dict]]:
    chunks, cur, cur_tok = [], [], 0
    for req in requests:
        t = request_tokens(req["body"])
        if cur and cur_tok + t > budget:
            chunks.append(cur)
            cur, cur_tok = [], 0
        cur.append(req)
        cur_tok += t
    if cur:
        chunks.append(cur)
    return chunks


def load_progress() -> dict:
    if PROGRESS_JSON.exists():
        return json.loads(PROGRESS_JSON.read_text())
    return {"done": [], "batch_ids": {}}


def save_progress(prog: dict) -> None:
    PROGRESS_JSON.write_text(json.dumps(prog))


def submit_chunk(client: OpenAI, chunk: list[dict], i: int) -> str:
    chunk_path = BATCH_DIR / f"chunk_{i}.jsonl"
    with chunk_path.open("w") as f:
        for req in chunk:
            f.write(json.dumps(req) + "\n")
    uploaded = client.files.create(file=chunk_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def accumulate_output(output_text: str, numerical_ids: set[str]) -> int:
    """Append parsed results for a completed batch to ACCUM_JSONL; return #errors."""
    n_err = 0
    with ACCUM_JSONL.open("a") as out:
        for line in output_text.splitlines():
            rec = json.loads(line)
            resp = rec.get("response") or {}
            if rec.get("error") or resp.get("status_code") != 200:
                n_err += 1
                print(
                    "  request error",
                    rec["custom_id"],
                    rec.get("error") or resp.get("status_code"),
                )
                continue
            out.write(
                json.dumps(
                    {
                        "cid": rec["custom_id"],
                        "numerical": rec["custom_id"] in numerical_ids,
                        "content": resp["body"]["choices"][0]["message"]["content"],
                    }
                )
                + "\n"
            )
    return n_err


def run_concurrent(client: OpenAI) -> None:
    compact = build_compact_requests()
    numerical_ids = {r["custom_id"] for r in compact if "response_format" in r["body"]}
    chunks = chunk_list(compact, CHUNK_BUDGET)
    ctok = [sum(request_tokens(r["body"]) for r in c) for c in chunks]
    print(
        f"{len(compact)} requests -> {len(chunks)} chunks (~{CHUNK_BUDGET // 1000}k each); "
        f"inflight budget {INFLIGHT_BUDGET // 1000}k",
        flush=True,
    )

    prog = load_progress()
    done = set(prog["done"])
    inflight = {
        int(i): bid for i, bid in prog["batch_ids"].items() if int(i) not in done
    }
    pending = [i for i in range(len(chunks)) if i not in done and i not in inflight]
    retries: dict[int, int] = {}

    while pending or inflight:
        # Fill to the inflight budget.
        cur = sum(ctok[i] for i in inflight)
        while pending:
            i = pending[0]
            if inflight and cur + ctok[i] > INFLIGHT_BUDGET:
                break
            bid = submit_chunk(client, chunks[i], i)
            inflight[i] = bid
            prog["batch_ids"][str(i)] = bid
            save_progress(prog)
            cur += ctok[i]
            pending.pop(0)
            print(
                f"chunk {i}: submitted {bid} ({len(chunks[i])} reqs, {ctok[i] // 1000}k); "
                f"inflight={len(inflight)} ~{cur // 1000}k",
                flush=True,
            )

        time.sleep(POLL_SECONDS)

        # Reap terminal batches.
        for i, bid in list(inflight.items()):
            batch = client.batches.retrieve(bid)
            if batch.status not in TERMINAL:
                continue
            del inflight[i]
            prog["batch_ids"].pop(str(i), None)
            if batch.status == "completed":
                n_err = accumulate_output(
                    client.files.content(batch.output_file_id).text, numerical_ids
                )
                done.add(i)
                prog["done"].append(i)
                save_progress(prog)
                print(
                    f"chunk {i}: completed ({n_err} req errors); done {len(done)}/{len(chunks)}",
                    flush=True,
                )
            else:
                retries[i] = retries.get(i, 0) + 1
                save_progress(prog)
                print(
                    f"chunk {i}: {batch.status} {getattr(batch, 'errors', None)}; retry {retries[i]}",
                    flush=True,
                )
                if retries[i] > MAX_RETRIES:
                    raise SystemExit(
                        f"chunk {i} failed {MAX_RETRIES}x; aborting (re-run to resume)"
                    )
                pending.append(i)  # requeue (capacity has freed since one just ended)

    build_evaluated()
    print("all chunks done; evaluated CSVs written", flush=True)


def eval_col_pairs(task_type: str, belief_pairs) -> list[tuple[str, str]]:
    """(value_col, extracted_col) destinations for a task_type. Mirrors
    evaluate.process_row's result-writing branches exactly."""
    if task_type == "system_user":
        return [("system_user_eval", "system_user_extracted")]
    if task_type == "belief":
        cols = [gen.replace("generation", "eval") for _, gen in belief_pairs]
    else:  # generation(System Prompt + User Prompt)_runN  OR  binary belief gen_col
        cols = [task_type.replace("generation", "eval")]
    return [(c, f"{c}_extracted") for c in cols]


def parse_content(content: str, numerical: bool):
    if numerical:
        nr = evaluate.NumericalRange(**json.loads(content))
        return (
            nr.lower_bound,
            nr.upper_bound,
            nr.relative_lower_bound,
            nr.relative_upper_bound,
        )
    return content.strip()


def build_evaluated() -> None:
    """Write csv_data/evaluated/*.csv from the accumulated results."""
    index = json.loads(INDEX_JSON.read_text())
    by_file: dict[str, list[tuple[int, str, object]]] = {}
    for line in ACCUM_JSONL.read_text().splitlines():
        rec = json.loads(line)
        filename, idx, task_type = index[rec["cid"]].split("::", 2)
        value = parse_content(rec["content"], rec["numerical"])
        by_file.setdefault(filename, []).append((int(idx), task_type, value))

    EVALUATED_DIR.mkdir(parents=True, exist_ok=True)
    for filename, items in by_file.items():
        df = pd.read_csv(RESPONSES_DIR / filename)
        df["ground_truth_extracted"] = df["formatted_ground_truth"]
        belief_pairs = evaluate.belief_columns_for(df)
        # Pre-create every target column as object dtype (as process_file does) so
        # df.at can hold tuple values (numerical judge) and never KeyErrors on a
        # missing column.
        cols = {
            c
            for _, task_type, _ in items
            for pair in eval_col_pairs(task_type, belief_pairs)
            for c in pair
        }
        for c in cols:
            df[c] = None
        for idx, task_type, value in items:
            for value_col, extracted_col in eval_col_pairs(task_type, belief_pairs):
                df.at[idx, value_col] = value
                df.at[idx, extracted_col] = evaluate.extract_boxed_evaluation(value)
        df.to_csv(EVALUATED_DIR / filename, index=False)
        print(f"wrote {EVALUATED_DIR / filename} ({len(items)} results)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run", action="store_true", help="concurrent chunked batch run"
    )
    group.add_argument(
        "--build", action="store_true", help="rebuild CSVs from accum.jsonl"
    )
    parser.add_argument("--model-slug", default=MODEL_SLUG)
    parser.add_argument("--archetypes", nargs="+", choices=ARCHETYPES, default=ARCHETYPES)
    parser.add_argument("--batch-dir", type=Path, default=BATCH_DIR)
    args = parser.parse_args()
    configure(args.model_slug, args.archetypes, args.batch_dir)

    if args.build:
        build_evaluated()
        return
    run_concurrent(OpenAI())  # OPENAI_API_KEY loaded from .env on `import evaluate`


if __name__ == "__main__":
    main()
