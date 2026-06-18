from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path


class MaskEvaluateTests(unittest.TestCase):
    def test_concurrency_one_runs_row_tasks_sequentially(self) -> None:
        evaluate = load_evaluate()
        events: list[str] = []

        async def task(name: str) -> str:
            events.append(f"start {name}")
            await asyncio.sleep(0)
            events.append(f"end {name}")
            return name

        results = asyncio.run(
            evaluate.run_row_tasks(
                [task("a"), task("b")],
                concurrency_limit=1,
                timeout_seconds=1,
            )
        )

        self.assertEqual(results, ["a", "b"])
        self.assertEqual(events, ["start a", "end a", "start b", "end b"])


def load_evaluate():
    root = Path(__file__).resolve().parents[1] / "mask" / "mask"
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("mask_evaluate", root / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
