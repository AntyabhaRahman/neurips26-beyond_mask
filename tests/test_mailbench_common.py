import unittest
from pathlib import Path
from mailbench._common import episode_dir_name, run_root


class CommonTests(unittest.TestCase):
    def test_episode_dir_name(self):
        self.assertEqual(
            episode_dir_name("q3_spin", None, "openai/gpt-5"), "q3_spin__openai__gpt-5"
        )
        self.assertEqual(
            episode_dir_name("q3_spin", "soft", "openai/gpt-5"),
            "q3_spin@soft__openai__gpt-5",
        )

    def test_run_root(self):
        self.assertEqual(run_root(Path("results"), "r1"), Path("results/mailbench/r1"))
