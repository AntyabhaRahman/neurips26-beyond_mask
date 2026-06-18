from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

SPLIT_DIRS = {
    "full": "csv_data",
    "test": "test_csv_data",
}
MAIN_RESPONSE_COL = "generation(System Prompt + User Prompt)_run1"
MAIN_RESPONSE_EVAL_COL = "eval(System Prompt + User Prompt)_run1"
MAIN_RESPONSE_TYPE_COL = "eval(System Prompt + User Prompt)_run1_extracted"
LOGGER = logging.getLogger(__name__)


def collect_summary_rows(
    root: Path, *, splits: list[str], normalize: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        metrics_dir = root / SPLIT_DIRS[split] / "metrics"
        if not metrics_dir.exists():
            LOGGER.info("skipping missing metrics dir %s", metrics_dir)
            continue

        LOGGER.info("processing MASK metrics split=%s dir=%s", split, metrics_dir)
        run_mask_process_metrics(root, split=split, normalize=normalize)
        results = read_mask_results(root, split=split, normalize=normalize)
        rows.extend(flatten_mask_results(results, source_split=split))
    return rows


def prepare_mask_metrics(root: Path, *, splits: list[str], concurrency_limit: int) -> None:
    for split in splits:
        split_dir = SPLIT_DIRS[split]
        response_dir = root / split_dir / "responses"
        if not response_dir.exists():
            continue
        ran_evaluate = False
        for response_file in sorted(response_dir.glob("*.csv")):
            if response_file.name.endswith(",.csv"):
                LOGGER.warning("skipping comma-suffixed response file %s", response_file)
                continue
            evaluated_file = root / split_dir / "evaluated" / response_file.name
            if evaluated_file.exists() and evaluated_file.stat().st_mtime >= response_file.stat().st_mtime:
                continue
            rel_input = response_file.relative_to(root)
            cmd = [
                sys.executable,
                "evaluate.py",
                "--input_file",
                str(rel_input),
                "--concurrency_limit",
                str(concurrency_limit),
            ]
            LOGGER.info("running %s in %s", " ".join(cmd), root)
            subprocess.run(cmd, cwd=root, check=True)
            ran_evaluate = True
        if ran_evaluate or not (root / split_dir / "metrics").exists():
            cmd = [sys.executable, "metric.py"]
            if split == "test":
                cmd.append("--test")
            LOGGER.info("running %s in %s", " ".join(cmd), root)
            subprocess.run(cmd, cwd=root, check=True)


def run_mask_process_metrics(root: Path, *, split: str, normalize: bool) -> None:
    cmd = [sys.executable, "process_metrics.py"]
    if normalize:
        cmd.append("--normalize")
    if split == "test":
        cmd.append("--test")
    LOGGER.info("running %s in %s", " ".join(cmd), root)
    subprocess.run(cmd, cwd=root, check=True)


def read_mask_results(root: Path, *, split: str, normalize: bool) -> dict[str, Any]:
    filename = "all_results2.json" if normalize else "all_results.json"
    path = root / SPLIT_DIRS[split] / "metrics" / filename
    LOGGER.info("reading MASK results from %s", path)
    return json.loads(path.read_text())


def flatten_mask_results(
    results: dict[str, dict[str, dict[str, Any]]], *, source_split: str
) -> list[dict[str, Any]]:
    rows = []
    for dataset_split in sorted(results):
        for model in sorted(results[dataset_split]):
            rows.append(
                {
                    "source_split": source_split,
                    "dataset_split": dataset_split,
                    "model": model,
                    **results[dataset_split][model],
                }
            )
    return rows


def write_summary(
    root: Path,
    out_dir: Path,
    *,
    splits: list[str],
    normalize: bool = False,
    prepare_metrics: bool = False,
    concurrency_limit: int = 50,
) -> None:
    if prepare_metrics:
        prepare_mask_metrics(root, splits=splits, concurrency_limit=concurrency_limit)
    rows = collect_summary_rows(root, splits=splits, normalize=normalize)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "summary_by_dataset_model.csv", rows)
    write_csv(out_dir / "summary_by_model.csv", rows_by_model(rows))
    response_rows = collect_response_type_rows(root, splits=splits)
    write_csv(out_dir / "response_type_rows.csv", response_rows)
    write_csv(
        out_dir / "response_types_by_dataset_model.csv",
        summarize_response_types(response_rows),
    )
    (out_dir / "summary_by_split_dataset_model.json").write_text(
        json.dumps(nest_by_split(rows), indent=2, sort_keys=True)
    )
    (out_dir / "summary_by_model.json").write_text(
        json.dumps(nest_by_model(rows), indent=2, sort_keys=True)
    )
    LOGGER.info("wrote summary outputs to %s", out_dir)


def nest_by_split(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    nested: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        nested.setdefault(row["source_split"], {}).setdefault(row["dataset_split"], {})[
            row["model"]
        ] = metric_payload(row)
    return nested


def rows_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (row["model"], row["source_split"], row["dataset_split"]),
    )


def collect_response_type_rows(root: Path, *, splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        metrics_dir = root / SPLIT_DIRS[split] / "metrics"
        if not metrics_dir.exists():
            continue
        for path in sorted(metrics_dir.glob("*.csv")):
            parsed = parse_metric_filename(path.name)
            if parsed is None:
                continue
            dataset_split, model = parsed
            with path.open(newline="") as file:
                for row in csv.DictReader(file):
                    if MAIN_RESPONSE_TYPE_COL not in row:
                        continue
                    rows.append(
                        {
                            "source_split": split,
                            "dataset_split": dataset_split,
                            "model": model,
                            "task_id": row.get("task_id", ""),
                            "response_type": row.get(MAIN_RESPONSE_TYPE_COL, ""),
                            "user_prompt": row.get("user_prompt", ""),
                            "model_response": row.get(MAIN_RESPONSE_COL, ""),
                            "judge_response": row.get(MAIN_RESPONSE_EVAL_COL, ""),
                        }
                    )
    return rows


def parse_metric_filename(filename: str) -> tuple[str, str] | None:
    if not filename.endswith(".csv"):
        return None
    stem = filename[:-4]
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def summarize_response_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str, str], int] = {}
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        group = (row["source_split"], row["dataset_split"], row["model"])
        response_type = row["response_type"] or "missing"
        totals[group] = totals.get(group, 0) + 1
        counts[(*group, response_type)] = counts.get((*group, response_type), 0) + 1

    out = []
    for key in sorted(counts):
        source_split, dataset_split, model, response_type = key
        total = totals[(source_split, dataset_split, model)]
        count = counts[key]
        out.append(
            {
                "source_split": source_split,
                "dataset_split": dataset_split,
                "model": model,
                "response_type": response_type,
                "count": count,
                "total": total,
                "percent": round(count / total * 100, 3) if total else 0.0,
            }
        )
    return out


def nest_by_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    nested: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        nested.setdefault(row["model"], {}).setdefault(row["source_split"], {})[
            row["dataset_split"]
        ] = metric_payload(row)
    return nested


def metric_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"source_split", "dataset_split", "model"}
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = stable_fieldnames(rows)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("wrote %s rows to %s", len(rows), path)


def stable_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["source_split", "dataset_split", "model", "total_responses"]
    seen = [key for key in preferred if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MASK process_metrics.py and collect its JSON summaries."
    )
    parser.add_argument("--root", type=Path, default=Path("mask/mask"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/mask_summary"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_DIRS),
        default=["full", "test"],
        help="Which MASK metrics directories to process.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Pass --normalize through to mask/mask/process_metrics.py.",
    )
    parser.add_argument(
        "--prepare-metrics",
        action="store_true",
        help="Run MASK evaluate.py for stale/missing evaluated files, then metric.py.",
    )
    parser.add_argument(
        "--concurrency-limit",
        type=int,
        default=50,
        help="Concurrency limit passed to mask/mask/evaluate.py when preparing metrics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    write_summary(
        args.root,
        args.out_dir,
        splits=args.splits,
        normalize=args.normalize,
        prepare_metrics=args.prepare_metrics,
        concurrency_limit=args.concurrency_limit,
    )
    print(f"Wrote summary to {args.out_dir}")


if __name__ == "__main__":
    main()
