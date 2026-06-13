import unittest
from beyond_mask.mailenv.personas import ScriptEngine
from beyond_mask.mailenv.scenario import ScenarioEmail, ScriptStep, Trigger
from beyond_mask.mailenv.store import Address


def step(**trig) -> ScriptStep:
    return ScriptStep(Trigger(**trig), ScenarioEmail(None, Address("a@b.c"), "s", "b"))


class ScriptEngineTests(unittest.TestCase):
    def test_on_start_fires_once_at_turn_zero(self):
        eng = ScriptEngine([step(on_start=True)])
        self.assertEqual(len(eng.due(0, set())), 1)
        self.assertEqual(eng.due(0, set()), [])
        self.assertTrue(eng.exhausted)

    def test_at_turn_fires_at_or_after_turn(self):
        eng = ScriptEngine([step(at_turn=2)])
        self.assertEqual(eng.due(1, set()), [])
        self.assertEqual(len(eng.due(3, set())), 1)  # late is fine, fires once

    def test_after_agent_email_to(self):
        eng = ScriptEngine([step(after_agent_email_to="boss@x.y")])
        self.assertEqual(eng.due(5, set()), [])
        self.assertEqual(len(eng.due(5, {"boss@x.y"})), 1)
        self.assertEqual(eng.due(6, {"boss@x.y"}), [])

    def test_partial_multi_step_engine_not_exhausted_until_all_fire(self):
        """exhausted must be False until every step has fired."""
        eng = ScriptEngine([step(at_turn=1), step(at_turn=3)])
        # Only first step fires at turn 1
        fired = eng.due(1, set())
        self.assertEqual(len(fired), 1)
        self.assertFalse(eng.exhausted)
        # Second step fires at turn 3
        fired = eng.due(3, set())
        self.assertEqual(len(fired), 1)
        self.assertTrue(eng.exhausted)
