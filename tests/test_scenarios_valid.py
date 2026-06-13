import unittest
from pathlib import Path
from beyond_mask.mailenv.scenario import load_scenario

SCENARIO_DIR = (
    Path(__file__).resolve().parent.parent / "mailbench/scenarios/handwritten"
)


class HandwrittenScenarioTests(unittest.TestCase):
    def test_all_scenarios_and_variants_validate(self):
        paths = sorted(SCENARIO_DIR.glob("*.yaml"))
        self.assertGreaterEqual(len(paths), 2)
        for path in paths:
            base = load_scenario(path)
            self.assertTrue(base.ground_truth and base.belief_elicitation)
            for v in base.variant_names:
                load_scenario(path, variant=v)
