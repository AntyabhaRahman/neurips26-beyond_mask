from __future__ import annotations

import csv
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts import summarize_mask_results as summary


class MaskSummaryTests(unittest.TestCase):
    def test_runs_mask_process_metrics_for_each_split_and_flattens_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_fake_mask_processor(root)

            rows = summary.collect_summary_rows(root, splits=["full", "test"])

        self.assertEqual(
            rows,
            [
                {
                    "source_split": "full",
                    "dataset_split": "known_facts",
                    "model": "model-a",
                    "total_responses": 2,
                    "honesty_score_1": 50.0,
                },
                {
                    "source_split": "test",
                    "dataset_split": "statistics",
                    "model": "model-b",
                    "total_responses": 1,
                    "honesty_score_1": 100.0,
                },
            ],
        )

    def test_writes_flat_and_nested_outputs_from_mask_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_fake_mask_processor(root)
            write_metric_csv(
                root / "test_csv_data" / "metrics" / "statistics_model-b.csv",
                [
                    {
                        "task_id": "q1",
                        "user_prompt": "Question?",
                        "generation(System Prompt + User Prompt)_run1": "No answer.",
                        "eval(System Prompt + User Prompt)_run1": "Answer: C",
                        "eval(System Prompt + User Prompt)_run1_extracted": "C",
                    }
                ],
            )
            out_dir = root / "summary"

            summary.write_summary(root, out_dir, splits=["test"])

            with (out_dir / "summary_by_dataset_model.csv").open() as file:
                flat_rows = list(csv.DictReader(file))
            with (out_dir / "summary_by_model.csv").open() as file:
                model_rows = list(csv.DictReader(file))
            with (out_dir / "response_type_rows.csv").open() as file:
                response_rows = list(csv.DictReader(file))
            with (out_dir / "response_types_by_dataset_model.csv").open() as file:
                response_summary = list(csv.DictReader(file))
            nested = json.loads((out_dir / "summary_by_split_dataset_model.json").read_text())

        self.assertEqual(flat_rows[0]["source_split"], "test")
        self.assertEqual(flat_rows[0]["dataset_split"], "statistics")
        self.assertEqual(flat_rows[0]["model"], "model-b")
        self.assertEqual(model_rows, flat_rows)
        self.assertEqual(response_rows[0]["response_type"], "C")
        self.assertEqual(response_rows[0]["user_prompt"], "Question?")
        self.assertEqual(response_summary[0]["response_type"], "C")
        self.assertEqual(response_summary[0]["count"], "1")
        self.assertEqual(response_summary[0]["percent"], "100.0")
        self.assertEqual(nested["test"]["statistics"]["model-b"]["total_responses"], 1)

    def test_prepare_metrics_runs_mask_evaluate_then_metric_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "csv_data" / "responses").mkdir(parents=True)
            write_metric_csv(
                root / "csv_data" / "responses" / "known_facts_model-a.csv",
                [{"task_id": "q1", "generation(System Prompt + User Prompt)_run1": "a"}],
            )
            write_fake_mask_pipeline(root)
            out_dir = root / "summary"

            summary.write_summary(root, out_dir, splits=["full"], prepare_metrics=True)

            order = (root / "order.log").read_text().splitlines()

        self.assertEqual(
            order,
            [
                "evaluate csv_data/responses/known_facts_model-a.csv",
                "metric",
                "process",
            ],
        )


def write_fake_mask_processor(root: Path) -> None:
    (root / "csv_data" / "metrics").mkdir(parents=True)
    (root / "test_csv_data" / "metrics").mkdir(parents=True)
    (root / "process_metrics.py").write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--normalize", action="store_true")
            parser.add_argument("--test", action="store_true")
            args = parser.parse_args()

            if args.test:
                path = Path("test_csv_data/metrics/all_results.json")
                data = {"statistics": {"model-b": {"total_responses": 1, "honesty_score_1": 100.0}}}
            else:
                path = Path("csv_data/metrics/all_results.json")
                data = {"known_facts": {"model-a": {"total_responses": 2, "honesty_score_1": 50.0}}}

            path.write_text(json.dumps(data))
            """
        )
    )


def write_fake_mask_pipeline(root: Path) -> None:
    (root / "evaluate.py").write_text(
        textwrap.dedent(
            """
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--input_file")
            parser.add_argument("--concurrency_limit")
            args = parser.parse_args()
            Path("order.log").open("a").write(f"evaluate {args.input_file}\\n")
            out = Path(args.input_file.replace("responses", "evaluated"))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("task_id,eval(System Prompt + User Prompt)_run1_extracted\\nq1,C\\n")
            """
        )
    )
    (root / "metric.py").write_text(
        textwrap.dedent(
            """
            from pathlib import Path

            Path("order.log").open("a").write("metric\\n")
            metrics = Path("csv_data/metrics")
            metrics.mkdir(parents=True, exist_ok=True)
            (metrics / "known_facts_model-a.csv").write_text(
                "task_id,eval(System Prompt + User Prompt)_run1_extracted\\nq1,C\\n"
            )
            """
        )
    )
    (root / "process_metrics.py").write_text(
        textwrap.dedent(
            """
            import json
            from pathlib import Path

            Path("order.log").open("a").write("process\\n")
            Path("csv_data/metrics/all_results.json").write_text(
                json.dumps({"known_facts": {"model-a": {"total_responses": 1}}})
            )
            """
        )
    )


def write_metric_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
