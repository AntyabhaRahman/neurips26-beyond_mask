import io
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


class MaskJudgeBatchCliTests(unittest.TestCase):
    def test_help_accepts_model_subset_and_state_directory(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        output = io.StringIO()
        with (
            patch("os.getcwd", return_value=str(repo / "mask/mask")),
            patch("tiktoken.get_encoding"),
            patch.object(
                sys,
                "argv",
                ["run_mask_judge_batch.py", "--help"],
            ),
            redirect_stdout(output),
            self.assertRaises(SystemExit) as exit_info,
        ):
            runpy.run_path(
                repo / "scripts/run_mask_judge_batch.py",
                run_name="__main__",
            )

        self.assertEqual(exit_info.exception.code, 0)
        self.assertIn("--model-slug", output.getvalue())
        self.assertIn("--archetypes", output.getvalue())
        self.assertIn("--batch-dir", output.getvalue())


if __name__ == "__main__":
    unittest.main()
