import json
import tempfile
import unittest
from pathlib import Path

import httpx

from beyond_mask.mailenv.episode import run_episode
from beyond_mask.mailenv.scenario import parse_scenario
from beyond_mask.openrouter import OpenRouterClient
from tests.test_mailenv_scenario import VALID


def tool_call_response(name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning": f"I will call {name}.",
                    "tool_calls": [
                        {
                            "id": f"call_{name}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001},
        "provider": "TestProv",
        "model": "test/model",
    }


SCRIPTED = [
    tool_call_response("list_emails", {}),
    tool_call_response("get_email", {"email_id": "em-0001"}),
    tool_call_response(
        "send_email",
        {
            "to": ["chair@oakboard.org"],
            "subject": "Re: Q3?",
            "body": "Q3 revenue fell 12% QoQ.",
        },
    ),
]


def make_handler(responses: list[dict]):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body.get("parallel_tool_calls") is False
        assert "tools" in body
        resp = responses[calls["n"]]
        calls["n"] += 1
        return httpx.Response(200, json=resp)

    return handler


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def run_once(self, tmp: Path, cache: Path, responses=SCRIPTED):
        scenario = parse_scenario(VALID)
        async with OpenRouterClient(
            api_key="k", transport=httpx.MockTransport(make_handler(responses))
        ) as client:
            return await run_episode(
                client, "test/model", scenario, tmp, cache_dir=cache
            )

    async def test_full_episode(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            result = await self.run_once(tmp / "ep", tmp / "cache")
            self.assertEqual(result.end_reason, "receiver_email")
            events = [
                json.loads(line)
                for line in (tmp / "ep/events.jsonl").read_text().splitlines()
            ]
            kinds = [e["event"] for e in events]
            self.assertEqual(kinds[0], "episode_start")
            self.assertIn("exposure", kinds)  # facts email opened before final send
            self.assertIn("agent_email", kinds)
            self.assertEqual(kinds[-1], "episode_end")
            exposure = next(e for e in events if e["event"] == "exposure")
            self.assertEqual(exposure["proposition"], "p1")
            for e in events:  # determinism contract
                self.assertNotIn("latency_ms", e)
                self.assertNotIn("cached", e)
            assistant_events = [e for e in events if e["event"] == "assistant_message"]
            self.assertGreaterEqual(len(assistant_events), 1)
            for event in assistant_events:
                self.assertRegex(event["request_hash"], r"^[0-9a-f]{64}$")
                self.assertIn("cache_hit", event)
                self.assertEqual(event["provider"], "TestProv")
                self.assertEqual(event["served_model"], "test/model")
                self.assertIn("prompt_tokens", event)
                self.assertIn("completion_tokens", event)
            self.assertTrue((tmp / "ep/mailbox/Sent").exists())
            summary = json.loads((tmp / "ep/episode.json").read_text())
            self.assertEqual(summary["end_reason"], "receiver_email")
            self.assertAlmostEqual(summary["cost_usd"], 0.0003)
            self.assertEqual(len(summary["calls"]), len(assistant_events))
            self.assertEqual(
                summary["calls"][0]["request_hash"],
                assistant_events[0]["request_hash"],
            )

    async def test_replay_is_byte_identical_via_cache(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            await self.run_once(tmp / "fill-cache", tmp / "cache")

            def explode(request):  # second run must never hit the network
                raise AssertionError("network hit on replay")

            scenario = parse_scenario(VALID)
            async with OpenRouterClient(
                api_key="k", transport=httpx.MockTransport(explode)
            ) as client:
                await run_episode(
                    client, "test/model", scenario, tmp / "ep1", cache_dir=tmp / "cache"
                )
                await run_episode(
                    client, "test/model", scenario, tmp / "ep2", cache_dir=tmp / "cache"
                )
            self.assertEqual(
                (tmp / "ep1/events.jsonl").read_bytes(),
                (tmp / "ep2/events.jsonl").read_bytes(),
            )

    async def test_malformed_tool_call_envelope_continues(self):
        """A tool_calls entry missing id/function.name logs tool_error and episode ends cleanly."""
        malformed = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"type": "function"}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            "provider": "TestProv",
            "model": "test/model",
        }
        # After the malformed turn, continue with get_email then send_email
        responses = [malformed, SCRIPTED[1], SCRIPTED[2]]
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            result = await self.run_once(tmp / "ep", tmp / "cache", responses=responses)
            events = [
                json.loads(line)
                for line in (tmp / "ep/events.jsonl").read_text().splitlines()
            ]
            kinds = [e["event"] for e in events]
            # Must have logged a tool_error for the malformed envelope
            self.assertIn("tool_error", kinds)
            tool_err = next(e for e in events if e["event"] == "tool_error")
            self.assertIn("malformed", tool_err.get("message", "").lower())
            # Episode must have finished cleanly
            self.assertEqual(result.end_reason, "receiver_email")
            self.assertTrue((tmp / "ep/episode.json").exists())
            self.assertEqual(kinds[-1], "episode_end")

    async def test_exposure_not_logged_after_final_send_same_turn(self):
        """exposure must not be logged after send_email in the same turn."""
        # Single response with TWO tool calls: send_email first, then get_email
        two_call_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {
                                    "name": "send_email",
                                    "arguments": json.dumps(
                                        {
                                            "to": ["chair@oakboard.org"],
                                            "subject": "Re: Q3?",
                                            "body": "Q3 revenue fell 12% QoQ.",
                                        }
                                    ),
                                },
                            },
                            {
                                "id": "call_b",
                                "type": "function",
                                "function": {
                                    "name": "get_email",
                                    "arguments": json.dumps({"email_id": "em-0001"}),
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0001},
            "provider": "TestProv",
            "model": "test/model",
        }
        responses = [two_call_response]
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            result = await self.run_once(tmp / "ep", tmp / "cache", responses=responses)
            events = [
                json.loads(line)
                for line in (tmp / "ep/events.jsonl").read_text().splitlines()
            ]
            kinds = [e["event"] for e in events]
            # No exposure should be logged since final_email was already set by send_email
            self.assertNotIn("exposure", kinds)
            self.assertEqual(result.end_reason, "receiver_email")

    async def test_idle_agent_ends_incomplete(self):
        idle = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Okay."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
        }
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            result = await self.run_once(
                tmp / "ep", tmp / "cache", responses=[idle] * 12
            )
            self.assertEqual(result.end_reason, "episode_incomplete")
