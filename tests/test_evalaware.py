from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.evalaware_cost import build_cost_payload
from scripts.evalaware_batch_run import target_result_record

from beyond_mask.evalaware import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SAMPLES_PER_CONFIG,
    DEFAULT_TARGET_MODELS,
    OPTIONAL_TARGET_MODELS,
    EvalAwareRow,
    ModelPrice,
    aggregate_report,
    build_work_items,
    estimate_cost,
    estimate_judge_cost,
    filter_rows_for_scope,
    model_price_from_record,
    resolve_requested_models,
    stable_custom_id,
    upsert_jsonl,
    write_jsonl,
)
from beyond_mask.openrouter import ChatResult


def row(
    task_id: str,
    *,
    task_name: str = "task",
    valence: str = "safety",
    factors: list[str] | None = None,
    prompt: str = "prompt text",
) -> EvalAwareRow:
    varied = factors or []
    return EvalAwareRow(
        task_id=task_id,
        task_name=task_name,
        valence=valence,
        description="desc",
        factors_varied=varied,
        num_factors_varied=len(varied),
        config={"F1": "realistic"},
        prompt=prompt,
    )


class EvalAwareScopeTests(unittest.TestCase):
    def test_default_samples_per_config_is_one(self) -> None:
        self.assertEqual(DEFAULT_SAMPLES_PER_CONFIG, 1)

    def test_filter_rows_for_paper_subset_keeps_baseline_and_single_factor(self) -> None:
        rows = [
            EvalAwareRow(
                task_id="baseline",
                task_name="task",
                valence="safety",
                description="desc",
                factors_varied=["none (baseline)"],
                num_factors_varied=0,
                config={},
                prompt="prompt text",
            ),
            row("f1", factors=["F1"]),
            row("f1_f2", factors=["F1", "F2"]),
        ]

        filtered = filter_rows_for_scope(rows, "paper")

        self.assertEqual([r.task_id for r in filtered], ["baseline", "f1"])

    def test_filter_rows_for_full_grid_keeps_everything(self) -> None:
        rows = [
            row("baseline"),
            row("f1", factors=["F1"]),
            row("f1_f2", factors=["F1", "F2"]),
        ]

        filtered = filter_rows_for_scope(rows, "full")

        self.assertEqual([r.task_id for r in filtered], ["baseline", "f1", "f1_f2"])

    def test_call_counts_match_evalawarebench_shapes(self) -> None:
        paper_rows = [row(f"r{i}") for i in range(1800)]
        full_rows = [row(f"r{i}") for i in range(51200)]

        paper_items = build_work_items(
            rows=paper_rows,
            models=DEFAULT_TARGET_MODELS,
            samples_per_config=3,
            service_tier="flex",
            kind="target",
        )
        full_items = build_work_items(
            rows=full_rows,
            models=DEFAULT_TARGET_MODELS,
            samples_per_config=3,
            service_tier="flex",
            kind="target",
        )

        self.assertEqual(len(paper_items), 10800)
        self.assertEqual(len(full_items), 307200)


class EvalAwareModelResolutionTests(unittest.TestCase):
    def test_default_target_models_are_opus_and_gpt55_only(self) -> None:
        self.assertEqual(
            DEFAULT_TARGET_MODELS,
            ["anthropic/claude-opus-4.8", "openai/gpt-5.5"],
        )

    def test_default_models_exclude_fable_and_default_judge_is_gpt55(self) -> None:
        self.assertEqual(DEFAULT_JUDGE_MODEL, "openai/gpt-5.5")
        self.assertEqual(OPTIONAL_TARGET_MODELS, [])

    def test_resolve_requested_models_excludes_fable_by_default(self) -> None:
        catalog = {
            "data": [
                {"id": model, "pricing": {"prompt": "0.1", "completion": "0.2"}}
                for model in [*DEFAULT_TARGET_MODELS, "anthropic/claude-fable-5"]
            ]
        }

        resolved = resolve_requested_models(catalog)

        self.assertNotIn("anthropic/claude-fable-5", resolved.available)
        self.assertEqual(resolved.unavailable_optional, [])

    def test_resolve_requested_models_marks_optional_model_unavailable(self) -> None:
        catalog = {
            "data": [
                {"id": model, "pricing": {"prompt": "0.1", "completion": "0.2"}}
                for model in DEFAULT_TARGET_MODELS
            ]
        }

        resolved = resolve_requested_models(
            catalog,
            optional=["anthropic/claude-fable-5"],
        )

        self.assertNotIn("anthropic/claude-fable-5", resolved.available)
        self.assertEqual(
            resolved.unavailable_optional, ["anthropic/claude-fable-5"]
        )


class EvalAwareCachingTests(unittest.TestCase):
    def test_stable_custom_id_is_deterministic_and_path_safe(self) -> None:
        sample_row = row("task/safety", factors=["F2"], prompt="hello")

        first = stable_custom_id(
            kind="target",
            model="anthropic/claude-sonnet-4.6",
            row=sample_row,
            sample_idx=3,
            service_tier="flex",
        )
        second = stable_custom_id(
            kind="target",
            model="anthropic/claude-sonnet-4.6",
            row=sample_row,
            sample_idx=3,
            service_tier="flex",
        )

        self.assertEqual(first, second)
        self.assertNotIn("/", first)
        self.assertIn("target", first)
        self.assertIn("sample3", first)

    def test_build_work_items_skips_completed_ids_unless_force_is_set(self) -> None:
        rows = [row("r1"), row("r2")]
        all_items = build_work_items(
            rows=rows,
            models=["openai/gpt-5.5"],
            samples_per_config=1,
            service_tier="flex",
            kind="target",
        )
        completed = {all_items[0].custom_id}

        remaining = build_work_items(
            rows=rows,
            models=["openai/gpt-5.5"],
            samples_per_config=1,
            service_tier="flex",
            kind="target",
            completed_ids=completed,
        )
        forced = build_work_items(
            rows=rows,
            models=["openai/gpt-5.5"],
            samples_per_config=1,
            service_tier="flex",
            kind="target",
            completed_ids=completed,
            force=True,
        )

        self.assertEqual([item.row.task_id for item in remaining], ["r2"])
        self.assertEqual(len(forced), 2)

    def test_write_jsonl_appends_records_with_custom_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            write_jsonl(path, [{"custom_id": "a"}, {"custom_id": "b"}])

            text = path.read_text()

        self.assertIn('"custom_id": "a"', text)
        self.assertIn('"custom_id": "b"', text)

    def test_upsert_jsonl_replaces_matching_custom_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            write_jsonl(
                path,
                [{"custom_id": "a", "value": 1}, {"custom_id": "b", "value": 2}],
            )
            upsert_jsonl(path, [{"custom_id": "a", "value": 3}])

            text = path.read_text()

        self.assertIn('"value": 3', text)
        self.assertIn('"custom_id": "b"', text)
        self.assertNotIn('"value": 1', text)

    def test_target_result_record_serializes_completed_item(self) -> None:
        item = build_work_items(
            rows=[row("r1", factors=["F1"])],
            models=["openai/gpt-5.5"],
            samples_per_config=1,
            service_tier="flex",
            kind="target",
        )[0]
        result = ChatResult(
            text="answer",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.03,
            finish_reason="stop",
            native_finish_reason=None,
            latency_ms=123,
            cached=False,
            error=None,
            reasoning="reasoning",
            service_tier="flex",
            request_hash="abc",
        )

        record = target_result_record(item, result)

        self.assertEqual(record["custom_id"], item.custom_id)
        self.assertEqual(record["config_name"], "F1")
        self.assertEqual(record["response"], "answer")
        self.assertEqual(record["service_tier"], "flex")


class EvalAwareCostTests(unittest.TestCase):
    def test_build_cost_payload_only_includes_paper_scope(self) -> None:
        rows = [row("baseline"), row("f1", factors=["F1"]), row("f1_f2", factors=["F1", "F2"])]
        catalog = {
            "data": [
                {
                    "id": model,
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                }
                for model in [*DEFAULT_TARGET_MODELS, DEFAULT_JUDGE_MODEL]
            ]
        }

        payload = build_cost_payload(
            rows=rows,
            catalog=catalog,
            samples_per_config=1,
            max_completion_tokens=10,
            judge_model=DEFAULT_JUDGE_MODEL,
            judge_max_tokens=5,
        )

        self.assertEqual(list(payload["scopes"]), ["paper"])
        self.assertEqual(payload["judge_model"], "openai/gpt-5.5")
        self.assertNotIn("anthropic/claude-fable-5", payload["models"])

    def test_model_price_does_not_confirm_flex_from_provider_name(self) -> None:
        price = model_price_from_record(
            {
                "id": "openai/gpt-5.5",
                "pricing": {"prompt": "0.002", "completion": "0.01"},
            }
        )

        self.assertFalse(price.flex_supported)

    def test_estimate_cost_reports_standard_and_confirmed_flex(self) -> None:
        rows = [
            row("r1", prompt="a" * 400),
            row("r2", prompt="b" * 800),
        ]
        prices = {
            "openai/gpt-5.5": ModelPrice(
                model_id="openai/gpt-5.5",
                prompt=0.002,
                completion=0.01,
                flex_discount=0.5,
                flex_supported=True,
            ),
            "anthropic/claude-sonnet-4.6": ModelPrice(
                model_id="anthropic/claude-sonnet-4.6",
                prompt=0.015,
                completion=0.12,
                flex_discount=0.5,
                flex_supported=False,
            ),
        }

        estimate = estimate_cost(
            rows=rows,
            models=list(prices),
            prices=prices,
            samples_per_config=3,
            max_completion_tokens=100,
            system_prompt="system",
        )

        self.assertEqual(estimate.target_calls, 12)
        self.assertLess(
            estimate.by_model["openai/gpt-5.5"].confirmed_flex_cost_usd,
            estimate.by_model["openai/gpt-5.5"].standard_cost_usd,
        )
        self.assertEqual(
            estimate.by_model[
                "anthropic/claude-sonnet-4.6"
            ].confirmed_flex_cost_usd,
            estimate.by_model["anthropic/claude-sonnet-4.6"].standard_cost_usd,
        )

    def test_estimate_judge_cost_adds_one_judge_call_per_target_response(self) -> None:
        rows = [row("r1"), row("r2")]
        prices = {
            "openai/gpt-5": ModelPrice(
                model_id="openai/gpt-5",
                prompt=0.002,
                completion=0.01,
            )
        }

        estimate = estimate_judge_cost(
            rows=rows,
            target_models=["m1", "m2"],
            judge_model="openai/gpt-5",
            prices=prices,
            samples_per_config=3,
            max_target_completion_tokens=100,
            max_judge_completion_tokens=20,
        )

        self.assertEqual(estimate.target_calls, 12)
        self.assertEqual(estimate.by_model["openai/gpt-5"].target_calls, 12)


class EvalAwareReportTests(unittest.TestCase):
    def test_aggregate_report_computes_awareness_propensity_and_baseline_delta(self) -> None:
        records = [
            {
                "model": "m",
                "valence": "safety",
                "config_name": "baseline",
                "judge": {
                    "model_awareness": {"conclusion": "No"},
                    "task_performance": {"conclusion": "Yes"},
                },
            },
            {
                "model": "m",
                "valence": "safety",
                "config_name": "F1",
                "judge": {
                    "model_awareness": {"conclusion": "Yes"},
                    "task_performance": {"conclusion": "No"},
                },
            },
            {
                "model": "m",
                "valence": "capability",
                "config_name": "baseline",
                "judge": {
                    "model_awareness": {"conclusion": "Yes"},
                    "task_performance": {"conclusion": "Yes"},
                },
            },
        ]

        report = aggregate_report(records)

        self.assertAlmostEqual(report["overall"]["awareness_rate"], 2 / 3)
        self.assertAlmostEqual(report["overall"]["task_performance_rate"], 2 / 3)
        self.assertAlmostEqual(report["by_model"]["m"]["propensity_shift"], -0.5)
        self.assertAlmostEqual(
            report["baseline_deltas"]["m"]["safety"]["F1"], -1.0
        )


if __name__ == "__main__":
    unittest.main()
