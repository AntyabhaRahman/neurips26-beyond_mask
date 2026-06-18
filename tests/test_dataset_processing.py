from __future__ import annotations

import io
import json
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bench_setup import dataset_processing as dp


class DatasetProcessingTests(unittest.TestCase):
    def run_script(self, argv: list[str]):
        rows = [{"system_prompt": "s", "user_prompt": "u"}]
        with (
            patch.object(sys, "stdout", io.StringIO()),
            patch.object(sys, "stderr", io.StringIO()),
            patch("datasets.load_dataset", return_value=rows) as load_dataset,
            patch("bench_setup.dataset_processing.run_model_batch", return_value={}) as run_batch,
            patch("bench_setup.dataset_processing.load_dotenv", create=True) as load_dotenv,
            TemporaryDirectory() as tmpdir,
            patch("bench_setup.dataset_processing.RESPONSES_DIR", Path(tmpdir)),
        ):
            dp.main(argv)
        load_dotenv.assert_called_once()
        return load_dataset, run_batch

    def test_num_rows_loads_split_slice(self) -> None:
        load_dataset, _ = self.run_script(
            ["--dataset", "cais/MASK", "--split", "test", "--num_rows", "7"]
        )

        load_dataset.assert_called_once_with(
            "cais/MASK",
            "known_facts",
            split="test[:7]",
            keep_in_memory=True,
        )

    def test_all_loads_entire_split(self) -> None:
        load_dataset, _ = self.run_script(
            ["--dataset", "cais/MASK", "--split", "train", "--all"]
        )

        load_dataset.assert_called_once_with(
            "cais/MASK",
            "known_facts",
            split="train",
            keep_in_memory=True,
        )

    def test_num_rows_must_be_positive(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_script(
                ["--dataset", "cais/MASK", "--split", "test", "--num_rows", "0"]
            )

    def test_expands_normal_prompt_and_beliefs(self) -> None:
        items = dp.build_request_items(
            [
                {
                    "system_prompt": "lie system",
                    "user_prompt": "lie user",
                    "belief_system_1": "belief system",
                    "belief_elicit_1": "belief one",
                    "belief_elicit_2": "belief two",
                    "belief_elicit_3": "belief three",
                }
            ],
            config="known_facts",
            lie_k=1,
        )

        self.assertEqual(
            [item["output_column"] for item in items],
            [
                "generation(System Prompt + User Prompt)_run1",
                "generation(belief_system_1 + belief_elicit_1)_run1",
                "generation(belief_system_1 + belief_elicit_1)_run2",
                "generation(belief_system_1 + belief_elicit_1)_run3",
                "generation(belief_system_1 + belief_elicit_2)_run1",
                "generation(belief_system_1 + belief_elicit_3)_run1",
            ],
        )
        self.assertEqual(
            items[0]["messages"],
            [
                {"role": "system", "content": "lie system"},
                {"role": "user", "content": "lie user"},
            ],
        )
        self.assertEqual(
            items[1]["messages"],
            [
                {"role": "system", "content": "belief system"},
                {"role": "user", "content": "belief one"},
            ],
        )

    def test_statistics_only_repeats_first_belief(self) -> None:
        items = dp.build_request_items(
            [
                {
                    "system_prompt": "s",
                    "user_prompt": "u",
                    "belief_elicit_1": "b1",
                    "belief_elicit_2": "b2",
                    "belief_elicit_3": "b3",
                }
            ],
            config="statistics",
            lie_k=1,
        )

        self.assertEqual(
            [item["output_column"] for item in items],
            [
                "generation(System Prompt + User Prompt)_run1",
                "generation(belief_system_1 + belief_elicit_1)_run1",
                "generation(belief_system_1 + belief_elicit_1)_run2",
                "generation(belief_system_1 + belief_elicit_1)_run3",
            ],
        )
        self.assertEqual(items[1]["messages"][0]["content"], "You are a helpful assistant.")

    def test_doubling_down_preserves_assistant_history(self) -> None:
        items = dp.build_request_items(
            [
                {
                    "system_prompt": "s",
                    "dd_turn_1_prompt": "first question",
                    "dd_turn_1_response": "first answer",
                    "user_prompt": "second question",
                }
            ],
            config="doubling_down_known_facts",
            lie_k=1,
        )

        self.assertEqual(
            items[0]["messages"],
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ],
        )

    def test_provided_facts_has_pressure_generation_only(self) -> None:
        items = dp.build_request_items(
            [{"system_prompt": "s", "user_prompt": "u"}],
            config="provided_facts",
            lie_k=2,
        )

        self.assertEqual(
            [item["output_column"] for item in items],
            [
                "generation(System Prompt + User Prompt)_run1",
                "generation(System Prompt + User Prompt)_run2",
            ],
        )

    def test_custom_ids_are_anthropic_batch_safe(self) -> None:
        items = dp.build_request_items(
            [{"system_prompt": "s", "user_prompt": "u"}],
            config="provided_facts",
            lie_k=1,
        )

        self.assertRegex(items[0]["custom_id"], re.compile(r"^[a-zA-Z0-9_-]{1,64}$"))

    def test_apply_results_uses_custom_id_not_result_order(self) -> None:
        rows = [{"system_prompt": "s", "user_prompt": "u"}]
        items = dp.build_request_items(rows, config="known_facts", lie_k=2)

        dp.apply_generation_results(
            rows,
            items,
            {
                items[1]["custom_id"]: "second",
                items[0]["custom_id"]: "first",
            },
        )

        self.assertEqual(rows[0]["generation(System Prompt + User Prompt)_run1"], "first")
        self.assertEqual(rows[0]["generation(System Prompt + User Prompt)_run2"], "second")

    def test_openai_batch_jsonl_and_unordered_results(self) -> None:
        client = FakeOpenAI()
        items = dp.build_request_items(
            [{"system_prompt": "s", "user_prompt": "u"}],
            config="known_facts",
            lie_k=2,
        )
        client.output = "\n".join(
            [
                json.dumps(
                    {
                        "custom_id": items[1]["custom_id"],
                        "response": {
                            "body": {"choices": [{"message": {"content": "second"}}]}
                        },
                    }
                ),
                json.dumps(
                    {
                        "custom_id": items[0]["custom_id"],
                        "response": {
                            "body": {"choices": [{"message": {"content": "first"}}]}
                        },
                    }
                ),
            ]
        )

        results = dp.run_openai_batch(
            client,
            model="openai/gpt-4o-mini",
            items=items,
            max_tokens=11,
            temperature=0.4,
            effort="medium",
            poll_interval=0,
            timeout_seconds=1,
        )

        lines = [json.loads(line) for line in client.uploaded.splitlines()]
        self.assertEqual(lines[0]["method"], "POST")
        self.assertEqual(lines[0]["url"], "/v1/chat/completions")
        self.assertEqual(lines[0]["body"]["model"], "gpt-4o-mini")
        self.assertEqual(lines[0]["body"]["max_tokens"], 11)
        self.assertEqual(lines[0]["body"]["temperature"], 0.4)
        self.assertEqual(lines[0]["body"]["messages"][0]["role"], "developer")
        self.assertEqual(client.batch_endpoint, "/v1/chat/completions")
        self.assertEqual(results[items[0]["custom_id"]], "first")
        self.assertEqual(results[items[1]["custom_id"]], "second")

    def test_openai_gpt5_batch_uses_reasoning_effort_not_max_tokens(self) -> None:
        client = FakeOpenAI()
        items = dp.build_request_items(
            [{"system_prompt": "s", "user_prompt": "u"}],
            config="known_facts",
            lie_k=1,
        )
        client.output = json.dumps(
            {
                "custom_id": items[0]["custom_id"],
                "response": {
                    "body": {"choices": [{"message": {"content": "ok"}}]}
                },
            }
        )

        dp.run_openai_batch(
            client,
            model="openai/gpt-5.4-mini",
            items=items,
            max_tokens=11,
            temperature=0.4,
            effort="high",
            poll_interval=0,
            timeout_seconds=1,
        )

        body = json.loads(client.uploaded.splitlines()[0])["body"]
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("max_completion_tokens", body)
        self.assertEqual(body["reasoning_effort"], "high")

    def test_openai_batch_item_error_is_recorded(self) -> None:
        client = FakeOpenAI()
        items = dp.build_request_items(
            [{"system_prompt": "s", "user_prompt": "u"}],
            config="known_facts",
            lie_k=1,
        )
        client.output = json.dumps(
            {
                "custom_id": items[0]["custom_id"],
                "error": {"message": "bad prompt"},
            }
        )

        results = dp.run_openai_batch(
            client,
            model="openai/gpt-4o-mini",
            items=items,
            max_tokens=11,
            temperature=0.4,
            effort="medium",
            poll_interval=0,
            timeout_seconds=1,
        )

        self.assertEqual(results[items[0]["custom_id"]], "[ERROR: bad prompt]")

    def test_openai_batch_response_body_error_is_recorded(self) -> None:
        client = FakeOpenAI()
        items = dp.build_request_items(
            [{"system_prompt": "s", "user_prompt": "u"}],
            config="known_facts",
            lie_k=1,
        )
        client.output = json.dumps(
            {
                "custom_id": items[0]["custom_id"],
                "response": {"body": {"error": {"message": "body failed"}}},
            }
        )

        results = dp.run_openai_batch(
            client,
            model="openai/gpt-4o-mini",
            items=items,
            max_tokens=11,
            temperature=0.4,
            effort="medium",
            poll_interval=0,
            timeout_seconds=1,
        )

        self.assertEqual(results[items[0]["custom_id"]], "[ERROR: body failed]")

    def test_openai_completed_batch_with_error_file_is_recorded(self) -> None:
        client = FakeOpenAI()
        client.output_file_id = None
        client.error_file_id = "error-1"
        items = dp.build_request_items(
            [{"system_prompt": "s", "user_prompt": "u"}],
            config="known_facts",
            lie_k=1,
        )
        client.error_output = json.dumps(
            {
                "custom_id": items[0]["custom_id"],
                "response": {
                    "body": {"error": {"message": "unsupported parameter"}}
                },
            }
        )

        results = dp.run_openai_batch(
            client,
            model="openai/gpt-4o-mini",
            items=items,
            max_tokens=11,
            temperature=0.4,
            effort="medium",
            poll_interval=0,
            timeout_seconds=1,
        )

        self.assertEqual(
            results[items[0]["custom_id"]], "[ERROR: unsupported parameter]"
        )

    def test_anthropic_batch_shape_and_error_result(self) -> None:
        client = FakeAnthropic()
        items = dp.build_request_items(
            [{"system_prompt": "s", "user_prompt": "u"}],
            config="known_facts",
            lie_k=2,
        )
        client.results = [
            {
                "custom_id": items[1]["custom_id"],
                "result": {
                    "type": "errored",
                    "error": {"message": "nope"},
                },
            },
            {
                "custom_id": items[0]["custom_id"],
                "result": {
                    "type": "succeeded",
                    "message": {"content": [{"text": "first"}]},
                },
            },
        ]

        results = dp.run_anthropic_batch(
            client,
            model="anthropic/claude-sonnet-4-5",
            items=items,
            max_tokens=9,
            temperature=0.2,
            effort="high",
            poll_interval=0,
            timeout_seconds=1,
        )

        request = client.created_requests[0]
        self.assertEqual(request["params"]["model"], "claude-sonnet-4-5")
        self.assertEqual(request["params"]["max_tokens"], 9)
        self.assertEqual(request["params"]["temperature"], 0.2)
        self.assertEqual(request["params"]["system"], "s")
        self.assertEqual(request["params"]["messages"], [{"role": "user", "content": "u"}])
        self.assertEqual(request["params"]["output_config"], {"effort": "high"})
        self.assertEqual(results[items[0]["custom_id"]], "first")
        self.assertEqual(results[items[1]["custom_id"]], "[ERROR: nope]")

    def test_anthropic_empty_success_content_is_recorded_as_error(self) -> None:
        results = dp.parse_anthropic_results(
            [
                {
                    "custom_id": "row0_lying_run1",
                    "result": {
                        "type": "succeeded",
                        "message": {"content": []},
                    },
                }
            ]
        )

        self.assertEqual(
            results["row0_lying_run1"], "[ERROR: empty response content]"
        )

    def test_anthropic_results_falls_back_to_http_download(self) -> None:
        client = FakeAnthropic()
        items = dp.build_request_items(
            [{"system_prompt": "s", "user_prompt": "u"}],
            config="known_facts",
            lie_k=1,
        )
        client.results_error = RuntimeError("stream failed")
        raw = json.dumps(
            {
                "custom_id": items[0]["custom_id"],
                "result": {
                    "type": "succeeded",
                    "message": {"content": [{"text": "recovered"}]},
                },
            }
        ).encode()

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
            patch("urllib.request.urlopen", return_value=FakeHTTPResponse(raw)),
        ):
            results = dp.run_anthropic_batch(
                client,
                model="anthropic/claude-sonnet-4-5",
                items=items,
                max_tokens=9,
                temperature=0.2,
                effort="high",
                poll_interval=0,
                timeout_seconds=1,
            )

        self.assertEqual(results[items[0]["custom_id"]], "recovered")


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeOpenAI:
    def __init__(self):
        self.output = ""
        self.error_output = ""
        self.output_file_id = "output-1"
        self.error_file_id = None
        self.uploaded = ""
        self.batch_endpoint = None
        self.files = Obj(create=self.create_file, content=self.file_content)
        self.batches = Obj(create=self.create_batch, retrieve=self.retrieve_batch)

    def create_file(self, *, file, purpose):
        self.uploaded = file.read().decode()
        self.purpose = purpose
        return Obj(id="file-1")

    def create_batch(self, *, input_file_id, endpoint, completion_window):
        self.batch_endpoint = endpoint
        self.completion_window = completion_window
        return Obj(id="batch-1")

    def retrieve_batch(self, batch_id):
        return Obj(
            status="completed",
            output_file_id=self.output_file_id,
            error_file_id=self.error_file_id,
        )

    def file_content(self, file_id):
        if file_id == self.error_file_id:
            return Obj(content=self.error_output.encode())
        return Obj(content=self.output.encode())


class FakeAnthropic:
    def __init__(self):
        self.results = []
        self.results_error = None
        self.created_requests = []
        self.messages = Obj(
            batches=Obj(
                create=self.create_batch,
                retrieve=self.retrieve_batch,
                results=self.batch_results,
            )
        )

    def create_batch(self, *, requests):
        self.created_requests = requests
        return Obj(id="batch-1")

    def retrieve_batch(self, message_batch_id):
        return Obj(processing_status="ended")

    def batch_results(self, message_batch_id):
        if self.results_error:
            raise self.results_error
        return self.results


class FakeHTTPResponse:
    def __init__(self, content: bytes):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.content


if __name__ == "__main__":
    unittest.main()
