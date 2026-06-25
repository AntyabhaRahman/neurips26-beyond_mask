"""Aggregate mailbench judgment outcomes into per-model metric summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from beyond_mask.mailenv.scenario import load_scenario
from beyond_mask.results import git_sha, model_slug, write_run

try:
    from mailbench._common import (
        RESULTS_ROOT,
        SCENARIO_DIR,
        run_root as mailbench_run_root,
    )
except ModuleNotFoundError:
    from _common import RESULTS_ROOT, SCENARIO_DIR, run_root as mailbench_run_root

SUMMARY_COLUMNS = [
    "model",
    "model_slug",
    "pressure_kind",
    "outcome",
    "count",
    "total",
    "rate",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _model_ids_by_slug(run_root: Path) -> dict[str, str]:
    run_meta = _read_json(run_root / "run.json")
    return {model_slug(model): model for model in run_meta.get("models", [])}


def _episode_summaries(run_root: Path) -> dict[str, dict]:
    episodes_root = run_root / "episodes"
    if not episodes_root.exists():
        return {}
    return {
        ep_dir.name: _read_json(ep_dir / "episode.json")
        for ep_dir in sorted(episodes_root.iterdir())
        if ep_dir.is_dir() and (ep_dir / "episode.json").exists()
    }


def _scenario_paths_by_id(scenario_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(scenario_dir.glob("*.yaml")):
        scenario = load_scenario(path)
        paths[scenario.id] = path
    return paths


def _pressure_kind(summary: dict, scenario_paths: dict[str, Path]) -> str:
    scenario_id = summary.get("scenario")
    scenario_path = scenario_paths.get(scenario_id)
    if scenario_path is None:
        return "unspecified"
    return load_scenario(scenario_path, variant=summary.get("variant")).pressure_kind


def _empty_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "model": pl.String,
            "model_slug": pl.String,
            "pressure_kind": pl.String,
            "outcome": pl.String,
            "count": pl.UInt32,
            "total": pl.UInt32,
            "rate": pl.Float64,
        }
    )


def _summary_frame(records: list[dict]) -> pl.DataFrame:
    if not records:
        return _empty_summary()

    judgments = pl.DataFrame(records)
    keys = ["model", "model_slug", "pressure_kind"]
    counts = judgments.group_by([*keys, "outcome"]).len("count")
    totals = judgments.group_by(keys).len("total")
    return (
        counts.join(totals, on=keys)
        .with_columns((pl.col("count") / pl.col("total")).alias("rate"))
        .with_columns(
            pl.when(pl.col("pressure_kind") == "all")
            .then(0)
            .otherwise(1)
            .alias("_pressure_order")
        )
        .sort(["model", "model_slug", "_pressure_order", "pressure_kind", "outcome"])
        .select(SUMMARY_COLUMNS)
    )


def write_metrics(run_root: Path, scenario_dir: Path, run_id: str) -> tuple[Path, Path]:
    """Write aggregate mailbench metrics for one run.

    Each proposition-level judgment contributes once to the underlying
    proposition count and appears in two rate buckets: the model-wide ``all``
    bucket and the scenario's pressure-kind bucket.
    """

    run_root.mkdir(parents=True, exist_ok=True)
    model_by_slug = _model_ids_by_slug(run_root)
    episodes = _episode_summaries(run_root)
    scenario_paths = _scenario_paths_by_id(scenario_dir)

    records: list[dict] = []
    proposition_judgments = 0
    judgments_root = run_root / "judgments"
    judgment_paths = (
        sorted(judgments_root.glob("*.json")) if judgments_root.exists() else []
    )
    for judgment_path in judgment_paths:
        slug = judgment_path.stem
        judgments_by_episode = _read_json(judgment_path)
        for episode_name in sorted(judgments_by_episode):
            summary = episodes.get(episode_name, {})
            model = model_by_slug.get(slug) or summary.get("model") or slug
            pressure_kind = _pressure_kind(summary, scenario_paths)
            prop_judgments = judgments_by_episode[episode_name]
            for prop_id in sorted(prop_judgments):
                outcome = str(prop_judgments[prop_id].get("outcome", "unknown"))
                proposition_judgments += 1
                base = {"model": model, "model_slug": slug, "outcome": outcome}
                records.append({**base, "pressure_kind": "all"})
                if pressure_kind != "all":
                    records.append({**base, "pressure_kind": pressure_kind})

    summary = _summary_frame(records)
    parquet_path = run_root / "summary.parquet"
    json_path = run_root / f"{run_id}.json"
    write_run(
        summary,
        {
            "run_id": run_id,
            "git_sha": git_sha(),
            "rows": summary.height,
            "proposition_judgments": proposition_judgments,
        },
        parquet_path,
        json_path,
    )
    return parquet_path, json_path


def main(args: argparse.Namespace) -> None:
    run_id = "test" if args.test else args.run_id
    if not run_id:
        raise SystemExit("--run-id is required unless --test is set")
    parquet_path, json_path = write_metrics(
        mailbench_run_root(RESULTS_ROOT, run_id),
        Path(args.scenarios),
        run_id,
    )
    metadata = _read_json(json_path)
    print(
        json.dumps(
            {
                "parquet": str(parquet_path),
                "json": str(json_path),
                "rows": metadata.get("rows", 0),
                "proposition_judgments": metadata.get("proposition_judgments", 0),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scenarios", default=str(SCENARIO_DIR))
    parser.add_argument(
        "--test", action="store_true", help="summarize the 'test' run_id"
    )
    main(parser.parse_args())
