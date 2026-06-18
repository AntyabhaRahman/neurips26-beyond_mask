from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunMaskBatchScriptTests(unittest.TestCase):
    def run_script(self, *args: str, **extra_env: str) -> list[str]:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            log = tmp / "uv.log"
            uv = tmp / "uv"
            uv.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$*\" >> {log}\n"
            )
            uv.chmod(0o755)
            env = os.environ | {"PATH": f"{tmp}:{os.environ['PATH']}", **extra_env}

            subprocess.run(
                ["bash", "scripts/run_mask_batch_generation.sh", *args],
                cwd=root,
                env=env,
                check=True,
            )

            return log.read_text().splitlines()

    def test_test_run_uses_one_row_one_model_one_config(self) -> None:
        lines = self.run_script(TEST_RUN="1")
        generation_lines = lines[:-1]

        self.assertEqual(len(generation_lines), 1)
        self.assertIn("--num_rows 1", generation_lines[0])
        self.assertNotIn("--all", generation_lines[0])
        self.assertIn("--config known_facts", generation_lines[0])
        self.assertIn("--models anthropic/claude-opus-4-8", generation_lines[0])
        self.assertEqual(
            lines[-1],
            "run python scripts/summarize_mask_results.py --splits full --prepare-metrics --concurrency-limit 50",
        )

    def test_num_rows_arg_keeps_all_configs(self) -> None:
        lines = self.run_script("--num-rows", "3")
        generation_lines = lines[:-1]

        self.assertEqual(len(generation_lines), 6)
        self.assertTrue(all("--num_rows 3" in line for line in generation_lines))
        self.assertTrue(all("--all" not in line for line in generation_lines))
        self.assertIn("--config known_facts", generation_lines[0])
        self.assertIn("--config statistics", generation_lines[-1])
        self.assertEqual(
            lines[-1],
            "run python scripts/summarize_mask_results.py --splits full --prepare-metrics --concurrency-limit 50",
        )

    def test_test_arg_can_pair_with_num_rows(self) -> None:
        lines = self.run_script("--test", "-n", "2")
        generation_lines = lines[:-1]

        self.assertEqual(len(generation_lines), 1)
        self.assertIn("--num_rows 2", generation_lines[0])
        self.assertIn("--config known_facts", generation_lines[0])
        self.assertEqual(
            lines[-1],
            "run python scripts/summarize_mask_results.py --splits full --prepare-metrics --concurrency-limit 50",
        )

    def test_summary_can_be_skipped(self) -> None:
        lines = self.run_script(TEST_RUN="1", RUN_SUMMARY="0")

        self.assertEqual(len(lines), 1)
        self.assertNotIn("summarize_mask_results.py", lines[-1])

    def test_prepare_metrics_can_be_skipped(self) -> None:
        lines = self.run_script(TEST_RUN="1", PREPARE_METRICS="0")

        self.assertEqual(
            lines[-1],
            "run python scripts/summarize_mask_results.py --splits full",
        )

    def test_summary_only_skips_generation(self) -> None:
        lines = self.run_script(SUMMARY_ONLY="1")

        self.assertEqual(
            lines,
            [
                "run python scripts/summarize_mask_results.py --splits full --prepare-metrics --concurrency-limit 50"
            ],
        )


if __name__ == "__main__":
    unittest.main()
