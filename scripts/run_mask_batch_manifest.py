from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import datasets
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_setup import dataset_processing as dp

LOGGER = logging.getLogger(__name__)


def submit(argv: list[str] | None = None) -> Path:
    load_dotenv()
    args = submit_parser().parse_args(argv)
    validate_common(args)
    clients: dict[str, Any] = {}
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "all": args.all,
        "num_rows": args.num_rows,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "lie_k": args.lie_k,
        "effort_level": args.effort_level,
        "entries": [],
    }
    for config in args.configs:
        rows = load_rows(args.dataset, config, args.split, args.all, args.num_rows)
        items = dp.build_request_items(rows, config=config, lie_k=args.lie_k)
        for model in args.models:
            provider = provider_for(model)
            if provider not in clients:
                clients[provider] = make_client(provider)
            client = clients[provider]
            if provider == "openai":
                batch_id = dp.create_openai_batch(
                    client,
                    model=model,
                    items=items,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    effort=args.effort_level,
                )
            else:
                batch_id = dp.create_anthropic_batch(
                    client,
                    model=model,
                    items=items,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    effort=args.effort_level,
                )
            manifest["entries"].append(
                {
                    "provider": provider,
                    "model": model,
                    "config": config,
                    "batch_id": batch_id,
                    "status": "submitted",
                    "request_count": len(items),
                    "output_path": str(dp.output_path(config, model)),
                }
            )
            LOGGER.info("submitted %s %s %s", config, model, batch_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"mask_batches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json(path, manifest)
    LOGGER.info("wrote manifest %s", path)
    return path


def poll(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = poll_parser().parse_args(argv)
    clients: dict[str, Any] = {}
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        data = json.loads(args.manifest.read_text())
        remaining = 0
        for entry in data["entries"]:
            if entry["status"] == "completed":
                continue
            remaining += poll_entry(data, entry, clients)
        write_json(args.manifest, data)
        if remaining == 0 or not args.watch or time.monotonic() >= deadline:
            return
        time.sleep(args.poll_interval)


def poll_entry(data: dict[str, Any], entry: dict[str, Any], clients: dict[str, Any]) -> int:
    provider = entry["provider"]
    if provider not in clients:
        clients[provider] = make_client(provider)
    client = clients[provider]
    status = remote_status(provider, client, entry["batch_id"])
    if status not in {"completed", "ended", "failed", "expired", "canceled"}:
        entry["status"] = status
        LOGGER.info("%s %s %s: %s", entry["config"], entry["model"], entry["batch_id"], status)
        return 1
    rows = load_rows(
        data["dataset"], entry["config"], data["split"], data["all"], data["num_rows"]
    )
    items = dp.build_request_items(rows, config=entry["config"], lie_k=data["lie_k"])
    if provider == "openai":
        status, results = dp.openai_batch_results(client, entry["batch_id"], items)
    else:
        status, results = dp.anthropic_batch_results(client, entry["batch_id"], items)
    LOGGER.info("%s %s %s: %s", entry["config"], entry["model"], entry["batch_id"], status)
    if results is None:
        entry["status"] = status
        return 1
    dp.apply_generation_results(rows, items, results)
    dp.write_rows_csv(Path(entry["output_path"]), rows)
    entry["status"] = "completed" if status in {"completed", "ended"} else status
    return 0


def remote_status(provider: str, client: Any, batch_id: str) -> str:
    if provider == "openai":
        return dp.get_value(client.batches.retrieve(batch_id), "status") or "unknown"
    return dp.get_value(client.messages.batches.retrieve(batch_id), "processing_status") or "unknown"


def load_rows(dataset: str, config: str, split: str, all_rows: bool, num_rows: int) -> list[dict[str, Any]]:
    split_expr = split if all_rows else f"{split}[:{num_rows}]"
    loaded = datasets.load_dataset(dataset, config or None, split=split_expr, keep_in_memory=True)
    return [dict(row) for row in loaded]


def provider_for(model: str) -> str:
    if model.startswith("openai/"):
        return "openai"
    if model.startswith("anthropic/"):
        return "anthropic"
    raise ValueError(f"Model must start with openai/ or anthropic/: {model}")


def make_client(provider: str) -> Any:
    if provider == "openai":
        from openai import OpenAI

        return OpenAI()
    from anthropic import Anthropic

    return Anthropic()


def validate_common(args: argparse.Namespace) -> None:
    if not args.all and args.num_rows <= 0:
        raise SystemExit("--num_rows must be positive")
    for model in args.models:
        provider_for(model)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="cais/MASK")
    parser.add_argument("--split", default="test")
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("-n", "--num_rows", "--num-rows", dest="num_rows", type=int, default=5)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--configs", nargs="+", default=dp.MASK_SUBSETS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_token", "--max_tokens", dest="max_tokens", type=int, default=4096)
    parser.add_argument("--lie_k", type=int, default=1)
    parser.add_argument("--effort_level", default="medium")


def submit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit MASK provider batches without polling")
    add_common(parser)
    parser.add_argument("--out_dir", type=Path, default=Path("results/mask_batches"))
    return parser


def poll_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poll a MASK batch manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll_interval", type=int, default=60)
    parser.add_argument("--timeout_seconds", type=int, default=86400)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Submit or poll MASK batch manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit_cmd = subparsers.add_parser("submit")
    add_common(submit_cmd)
    submit_cmd.add_argument("--out_dir", type=Path, default=Path("results/mask_batches"))
    poll_cmd = subparsers.add_parser("poll")
    poll_cmd.add_argument("manifest", type=Path)
    poll_cmd.add_argument("--watch", action="store_true")
    poll_cmd.add_argument("--poll_interval", type=int, default=60)
    poll_cmd.add_argument("--timeout_seconds", type=int, default=86400)
    args = parser.parse_args(argv)
    if args.command == "submit":
        submit(vars_to_args(args))
    else:
        poll(vars_to_args(args))


def vars_to_args(args: argparse.Namespace) -> list[str]:
    values = vars(args).copy()
    values.pop("command", None)
    argv: list[str] = []
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
        elif isinstance(value, list):
            argv.append(f"--{key}")
            argv.extend(str(item) for item in value)
        elif key == "manifest":
            argv.append(str(value))
        else:
            argv.extend([f"--{key}", str(value)])
    return argv


if __name__ == "__main__":
    main()
