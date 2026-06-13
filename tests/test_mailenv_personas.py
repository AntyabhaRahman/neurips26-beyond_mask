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
