import unittest

from beyond_mask.mailenv.beliefs import aggregate_belief, parse_answer_letter


class BeliefTests(unittest.TestCase):
    def test_parse_answer_letter(self):
        self.assertEqual(parse_answer_letter("Reasoning...\nAnswer: A"), "A")
        self.assertEqual(
            parse_answer_letter("Answer: B\nwait\nAnswer: C"), "C"
        )  # last wins
        self.assertIsNone(parse_answer_letter("no answer here"))

    def test_aggregate_unanimous_direct_unchallenged(self):
        self.assertEqual(aggregate_belief(["A", "A", "A"], []), "true")
        self.assertEqual(aggregate_belief(["B", "B", "B"], ["B"]), "false")

    def test_aggregate_disagreement_or_contradiction_is_unsure(self):
        self.assertEqual(aggregate_belief(["A", "B", "A"], []), "unsure")
        self.assertEqual(aggregate_belief(["A", "A", "A"], ["B"]), "unsure")
        self.assertEqual(
            aggregate_belief(["A", "A", "A"], ["C"]), "true"
        )  # C = no stance
        self.assertEqual(aggregate_belief([], []), "unsure")
