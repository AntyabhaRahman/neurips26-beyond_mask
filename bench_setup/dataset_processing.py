from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import datasets
from dotenv import load_dotenv

MASK_SUBSETS = [
    "known_facts",
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "provided_facts",
    "statistics",
]

RESPONSES_DIR = Path("mask/mask/csv_data/responses")
LOGGER = logging.getLogger(__name__)


def build_request_items(
    rows: list[dict[str, Any]], *, config: str, lie_k: int
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    belief_k = {"1": 3} if config == "statistics" else {"1": 3, "2": 1, "3": 1}

    for row_index, row in enumerate(rows):
        is_doubling_down = present(row.get("dd_turn_1_prompt")) and present(
            row.get("dd_turn_1_response")
        )
        for run in range(1, lie_k + 1):
            if is_doubling_down:
                messages = [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["dd_turn_1_prompt"]},
                    {"role": "assistant", "content": row["dd_turn_1_response"]},
                    {"role": "user", "content": row["user_prompt"]},
                ]
            else:
                messages = [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_prompt"]},
                ]
            add_item(items, row_index, f"lying_run{run}", messages)

        if "belief_elicit_1" not in row:
            continue

        belief_system = row.get("belief_system_1") or "You are a helpful assistant."
        for belief_num, count in belief_k.items():
            belief_col = f"belief_elicit_{belief_num}"
            if not present(row.get(belief_col)):
                continue
            for run in range(1, count + 1):
                add_item(
                    items,
                    row_index,
                    f"be{belief_num}_run{run}",
                    [
                        {"role": "system", "content": belief_system},
                        {"role": "user", "content": row[belief_col]},
                    ],
                )

    return items


def present(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def add_item(
    items: list[dict[str, Any]],
    row_index: int,
    prompt_type: str,
    messages: list[dict[str, str]],
) -> None:
    items.append(
        {
            "row_index": row_index,
            "prompt_type": prompt_type,
            "messages": messages,
            "output_column": output_column(prompt_type),
            "custom_id": f"row{row_index}_{prompt_type}",
        }
    )


def output_column(prompt_type: str) -> str:
    if prompt_type.startswith("lying_run"):
        run = prompt_type.removeprefix("lying_run")
        return f"generation(System Prompt + User Prompt)_run{run}"
    belief, run = prompt_type.split("_run", 1)
    belief_num = belief.removeprefix("be")
    return f"generation(belief_system_1 + belief_elicit_{belief_num})_run{run}"


def apply_generation_results(
    rows: list[dict[str, Any]],
    items: list[dict[str, Any]],
    results: dict[str, str],
) -> None:
    for item in items:
        rows[item["row_index"]][item["output_column"]] = results.get(
            item["custom_id"], "[ERROR: Generation failed or timed out]"
        )


def run_model_batch(
    model: str,
    items: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    effort: str,
    poll_interval: int,
    timeout_seconds: int,
) -> dict[str, str]:
    if model.startswith("openai/"):
        from openai import OpenAI

        return run_openai_batch(
            OpenAI(),
            model=model,
            items=items,
            max_tokens=max_tokens,
            temperature=temperature,
            effort=effort,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )
    if model.startswith("anthropic/"):
        from anthropic import Anthropic

        return run_anthropic_batch(
            Anthropic(),
            model=model,
            items=items,
            max_tokens=max_tokens,
            temperature=temperature,
            effort=effort,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Model must start with openai/ or anthropic/: {model}")


def run_openai_batch(
    client: Any,
    *,
    model: str,
    items: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    effort: str,
    poll_interval: int,
    timeout_seconds: int,
) -> dict[str, str]:
    batch_id = create_openai_batch(
        client,
        model=model,
        items=items,
        max_tokens=max_tokens,
        temperature=temperature,
        effort=effort,
    )
    final = poll(
        lambda: client.batches.retrieve(batch_id),
        lambda batch: get_value(batch, "status") == "completed",
        lambda batch: get_value(batch, "status") in {"failed", "expired", "canceled"},
        poll_interval,
        timeout_seconds,
        label=f"openai {batch_id}",
    )
    status = get_value(final, "status")
    if status != "completed":
        LOGGER.error("openai batch: %s finished with status %s", batch_id, status)
        return all_errors(items, f"batch {status or 'timed out'}")
    _, results = openai_batch_results(client, batch_id, items)
    return results or all_errors(items, "batch completed without downloadable results")


def create_openai_batch(
    client: Any,
    *,
    model: str,
    items: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    effort: str,
) -> str:
    api_model = strip_model_prefix(model, "openai/")
    LOGGER.info("openai batch: preparing %s requests for %s", len(items), api_model)
    lines = [
        json.dumps(
            {
                "custom_id": item["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": openai_body(
                    api_model, item["messages"], max_tokens, temperature, effort
                ),
            }
        )
        for item in items
    ]
    batch_file = io.BytesIO(("\n".join(lines) + "\n").encode())
    batch_file.name = "mask_batch.jsonl"
    uploaded = client.files.create(file=batch_file, purpose="batch")
    LOGGER.info("openai batch: uploaded input file %s", get_value(uploaded, "id"))
    batch = client.batches.create(
        input_file_id=get_value(uploaded, "id"),
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = get_value(batch, "id")
    LOGGER.info("openai batch: submitted %s", batch_id)
    return batch_id


def openai_batch_results(
    client: Any, batch_id: str, items: list[dict[str, Any]]
) -> tuple[str, dict[str, str] | None]:
    final = client.batches.retrieve(batch_id)
    status = get_value(final, "status")
    if status != "completed":
        if status in {"failed", "expired", "canceled"}:
            return status, all_errors(items, f"batch {status}")
        return status or "unknown", None
    results: dict[str, str] = {}
    output_file_id = get_value(final, "output_file_id")
    if output_file_id:
        LOGGER.info("openai batch: downloading output file %s", output_file_id)
        content = client.files.content(output_file_id)
        results.update(parse_openai_output(file_text(content)))
    error_file_id = get_value(final, "error_file_id")
    if error_file_id:
        LOGGER.warning("openai batch: downloading error file %s", error_file_id)
        content = client.files.content(error_file_id)
        results.update(parse_openai_output(file_text(content)))
    LOGGER.info("openai batch: collected %s/%s results", len(results), len(items))
    return status, results


def openai_body(
    api_model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    effort: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": api_model,
        "messages": openai_messages(api_model, messages),
        "temperature": temperature,
    }
    if api_model.startswith("gpt-5"):
        body["reasoning_effort"] = effort
    else:
        body["max_tokens"] = max_tokens
    return body


def openai_messages(
    api_model: str, messages: list[dict[str, str]]
) -> list[dict[str, str]]:
    copied = [dict(message) for message in messages]
    if (
        copied
        and copied[0]["role"] == "system"
        and any(marker in api_model for marker in ("gpt", "o1", "o3"))
    ):
        copied[0]["role"] = "developer"
    return copied


def parse_openai_output(text: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item["custom_id"]
        error = get_value(item, "error")
        if error:
            results[custom_id] = f"[ERROR: {error_message(error)}]"
            continue
        body = item.get("response", {}).get("body", {})
        if body.get("error"):
            results[custom_id] = f"[ERROR: {error_message(body['error'])}]"
            continue
        results[custom_id] = body["choices"][0]["message"]["content"]
    return results


def run_anthropic_batch(
    client: Any,
    *,
    model: str,
    items: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    effort: str,
    poll_interval: int,
    timeout_seconds: int,
) -> dict[str, str]:
    batch_id = create_anthropic_batch(
        client,
        model=model,
        items=items,
        max_tokens=max_tokens,
        temperature=temperature,
        effort=effort,
    )
    final = poll(
        lambda: client.messages.batches.retrieve(batch_id),
        lambda batch: get_value(batch, "processing_status") == "ended",
        lambda batch: (
            get_value(batch, "processing_status") in {"failed", "expired", "canceled"}
        ),
        poll_interval,
        timeout_seconds,
        label=f"anthropic {batch_id}",
    )
    status = get_value(final, "processing_status")
    if status != "ended":
        LOGGER.error("anthropic batch: %s finished with status %s", batch_id, status)
        return all_errors(items, f"batch {status or 'timed out'}")
    _, results = anthropic_batch_results(client, batch_id, items)
    return results or all_errors(items, "batch ended without downloadable results")


def create_anthropic_batch(
    client: Any,
    *,
    model: str,
    items: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    effort: str,
) -> str:
    api_model = strip_model_prefix(model, "anthropic/")
    LOGGER.info("anthropic batch: preparing %s requests for %s", len(items), api_model)
    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": item["custom_id"],
                "params": anthropic_params(
                    api_model, item["messages"], max_tokens, temperature, effort
                ),
            }
            for item in items
        ]
    )
    batch_id = get_value(batch, "id")
    LOGGER.info("anthropic batch: submitted %s", batch_id)
    return batch_id


def anthropic_batch_results(
    client: Any, batch_id: str, items: list[dict[str, Any]]
) -> tuple[str, dict[str, str] | None]:
    final = client.messages.batches.retrieve(batch_id)
    status = get_value(final, "processing_status")
    if status != "ended":
        if status in {"failed", "expired", "canceled"}:
            return status, all_errors(items, f"batch {status}")
        return status or "unknown", None
    try:
        results = parse_anthropic_results(client.messages.batches.results(batch_id))
    except Exception as exc:
        LOGGER.warning(
            "anthropic batch: SDK results stream failed for %s: %s; retrying raw HTTP",
            batch_id,
            exc,
        )
        results = parse_anthropic_results(download_anthropic_results(batch_id))
    LOGGER.info("anthropic batch: collected %s/%s results", len(results), len(items))
    return status, results


def download_anthropic_results(batch_id: str) -> list[dict[str, Any]]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required to download Anthropic batch results"
        )
    request = urllib.request.Request(
        f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        text = response.read().decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def anthropic_params(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    effort: str,
) -> dict[str, Any]:
    system = None
    remaining = [dict(message) for message in messages]
    if remaining and remaining[0]["role"] == "system":
        system = remaining.pop(0)["content"]
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": remaining,
        "temperature": temperature,
        "output_config": {"effort": effort},
    }
    if system is not None:
        params["system"] = system
    return params


def parse_anthropic_results(stream: Any) -> dict[str, str]:
    results: dict[str, str] = {}
    for item in stream:
        custom_id = get_value(item, "custom_id")
        result = get_value(item, "result")
        result_type = get_value(result, "type")
        if result_type == "succeeded":
            message = get_value(result, "message")
            content_blocks = get_value(message, "content") or []
            if not content_blocks:
                results[custom_id] = "[ERROR: empty response content]"
                continue
            text = get_value(content_blocks[0], "text")
            results[custom_id] = text or "[ERROR: missing response text]"
        else:
            results[custom_id] = (
                f"[ERROR: {error_message(get_value(result, 'error') or result_type)}]"
            )
    return results


def poll(
    retrieve: Any,
    done: Any,
    failed: Any,
    poll_interval: int,
    timeout_seconds: int,
    label: str = "batch",
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    while True:
        batch = retrieve()
        LOGGER.info("%s status: %s %s", label, batch_status(batch), batch_counts(batch))
        if done(batch) or failed(batch) or time.monotonic() >= deadline:
            return batch
        time.sleep(poll_interval)


def batch_status(batch: Any) -> str:
    return (
        get_value(batch, "status") or get_value(batch, "processing_status") or "unknown"
    )


def batch_counts(batch: Any) -> str:
    counts = get_value(batch, "request_counts")
    if counts is None:
        return ""
    parts = []
    for key in ("completed", "failed", "total", "succeeded", "errored", "processing"):
        value = get_value(counts, key)
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def all_errors(items: list[dict[str, Any]], message: str) -> dict[str, str]:
    return {item["custom_id"]: f"[ERROR: {message}]" for item in items}


def strip_model_prefix(model: str, prefix: str) -> str:
    if not model.startswith(prefix):
        raise ValueError(f"Model must start with {prefix}: {model}")
    return model.removeprefix(prefix)


def get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def error_message(error: Any) -> str:
    if isinstance(error, str):
        return error
    return get_value(error, "message") or get_value(error, "type") or str(error)


def file_text(content: Any) -> str:
    if hasattr(content, "text"):
        text = content.text
        return text() if callable(text) else text
    if hasattr(content, "content"):
        raw = content.content
        return raw.decode() if isinstance(raw, bytes) else raw
    if hasattr(content, "read"):
        raw = content.read()
        return raw.decode() if isinstance(raw, bytes) else raw
    return str(content)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("wrote %s rows to %s", len(rows), path)


def output_path(config: str, model: str) -> Path:
    # strip stray separators so a trailing comma can't leak into the filename
    slug = model.split("/")[-1].strip(", ")
    return RESPONSES_DIR / f"{config}_{slug}.csv"


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    load_dotenv()
    parser = argparse.ArgumentParser(description="Process the dataset")
    parser.add_argument("--dataset", default="cais/MASK")
    parser.add_argument("--split", default="test")
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("-n", "--num_rows", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["openai/gpt-4o-mini"])
    parser.add_argument("--config", default="known_facts")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--max_token", "--max_tokens", dest="max_tokens", type=int, default=4096
    )
    parser.add_argument("--lie_k", type=int, default=1)
    parser.add_argument("--effort_level", default="medium")
    parser.add_argument("--poll_interval", type=int, default=60)
    parser.add_argument("--timeout_seconds", type=int, default=86400)

    args = parser.parse_args(argv)
    if not args.all and args.num_rows <= 0:
        parser.error("--num_rows must be positive")
    if args.dataset == "cais/MASK" and args.config not in MASK_SUBSETS:
        parser.error(
            f"Invalid config for cais/MASK. Must be one of: {', '.join(MASK_SUBSETS)}"
        )
    for model in args.models:
        if not (model.startswith("openai/") or model.startswith("anthropic/")):
            parser.error(f"Model must start with openai/ or anthropic/: {model}")

    split = args.split if args.all else f"{args.split}[:{args.num_rows}]"
    dataset = datasets.load_dataset(
        args.dataset,
        args.config or None,
        split=split,
        keep_in_memory=True,
    )
    rows = [dict(row) for row in dataset]
    LOGGER.info(
        "loaded %s rows from %s/%s split=%s",
        len(rows),
        args.dataset,
        args.config,
        split,
    )

    for model in args.models:
        model_rows = [dict(row) for row in rows]
        items = build_request_items(model_rows, config=args.config, lie_k=args.lie_k)
        LOGGER.info(
            "starting generation: config=%s model=%s rows=%s requests=%s",
            args.config,
            model,
            len(model_rows),
            len(items),
        )
        results = run_model_batch(
            model,
            items,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            effort=args.effort_level,
            poll_interval=args.poll_interval,
            timeout_seconds=args.timeout_seconds,
        )
        apply_generation_results(model_rows, items, results)
        write_rows_csv(output_path(args.config, model), model_rows)


if __name__ == "__main__":
    main()
