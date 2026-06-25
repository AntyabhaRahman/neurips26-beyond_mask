from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import run_mask_batch_manifest as manifest


class MaskBatchManifestTests(unittest.TestCase):
    def test_submit_writes_manifest_without_polling(self) -> None:
        rows = [{"system_prompt": "s", "user_prompt": "u"}]
        with (
            TemporaryDirectory() as tmpdir,
            patch("datasets.load_dataset", return_value=rows) as load_dataset,
            patch("scripts.run_mask_batch_manifest.load_dotenv", create=True),
            patch("scripts.run_mask_batch_manifest.dp.create_openai_batch", return_value="batch-o") as openai,
            patch("scripts.run_mask_batch_manifest.dp.create_anthropic_batch", return_value="batch-a") as anthropic,
        ):
            path = manifest.submit(
                [
                    "--models",
                    "openai/gpt-5.4-mini",
                    "anthropic/claude-opus-4-8",
                    "--configs",
                    "known_facts",
                    "--num_rows",
                    "1",
                    "--out_dir",
                    tmpdir,
                ]
            )

            data = json.loads(Path(path).read_text())

        load_dataset.assert_called_with(
            "cais/MASK", "known_facts", split="test[:1]", keep_in_memory=True
        )
        self.assertEqual(openai.call_count, 1)
        self.assertEqual(anthropic.call_count, 1)
        self.assertEqual(
            [(entry["provider"], entry["batch_id"], entry["status"]) for entry in data["entries"]],
            [("openai", "batch-o", "submitted"), ("anthropic", "batch-a", "submitted")],
        )

    def test_poll_writes_finished_csv_and_updates_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            output_path = Path(tmpdir) / "out.csv"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "cais/MASK",
                        "split": "test",
                        "all": False,
                        "num_rows": 1,
                        "lie_k": 1,
                        "entries": [
                            {
                                "provider": "openai",
                                "model": "openai/gpt-5.4-mini",
                                "config": "known_facts",
                                "batch_id": "batch-1",
                                "status": "submitted",
                                "output_path": str(output_path),
                            }
                        ],
                    }
                )
            )
            rows = [{"system_prompt": "s", "user_prompt": "u"}]
            client = Obj(batches=Obj(retrieve=lambda batch_id: Obj(status="completed")))
            with (
                patch("scripts.run_mask_batch_manifest.make_client", return_value=client),
                patch("datasets.load_dataset", return_value=rows),
                patch(
                    "scripts.run_mask_batch_manifest.dp.openai_batch_results",
                    return_value=("completed", {"row0_lying_run1": "answer"}),
                ),
            ):
                manifest.poll([str(manifest_path)])

            self.assertIn("answer", output_path.read_text())
            data = json.loads(manifest_path.read_text())
            self.assertEqual(data["entries"][0]["status"], "completed")

    def test_poll_active_batch_does_not_reload_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "cais/MASK",
                        "split": "test",
                        "all": True,
                        "num_rows": 1,
                        "lie_k": 1,
                        "entries": [
                            {
                                "provider": "openai",
                                "model": "openai/gpt-5.4-mini",
                                "config": "known_facts",
                                "batch_id": "batch-1",
                                "status": "submitted",
                                "output_path": str(Path(tmpdir) / "out.csv"),
                            }
                        ],
                    }
                )
            )
            client = Obj(batches=Obj(retrieve=lambda batch_id: Obj(status="in_progress")))
            with (
                patch("scripts.run_mask_batch_manifest.make_client", return_value=client),
                patch("datasets.load_dataset") as load_dataset,
                patch("scripts.run_mask_batch_manifest.dp.openai_batch_results") as results,
            ):
                manifest.poll([str(manifest_path)])

            load_dataset.assert_not_called()
            results.assert_not_called()
            data = json.loads(manifest_path.read_text())
            self.assertEqual(data["entries"][0]["status"], "in_progress")


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    unittest.main()
