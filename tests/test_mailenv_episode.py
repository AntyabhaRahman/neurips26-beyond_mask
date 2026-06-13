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
            self.assertTrue((tmp / "ep/mailbox/Sent").exists())
            summary = json.loads((tmp / "ep/episode.json").read_text())
            self.assertEqual(summary["end_reason"], "receiver_email")
            self.assertAlmostEqual(summary["cost_usd"], 0.0003)

    async def test_replay_is_byte_identical_via_cache(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            await self.run_once(tmp / "ep1", tmp / "cache")

            def explode(request):  # second run must never hit the network
                raise AssertionError("network hit on replay")

            scenario = parse_scenario(VALID)
            async with OpenRouterClient(
                api_key="k", transport=httpx.MockTransport(explode)
            ) as client:
                await run_episode(
                    client, "test/model", scenario, tmp / "ep2", cache_dir=tmp / "cache"
                )
            self.assertEqual(
                (tmp / "ep1/events.jsonl").read_bytes(),
                (tmp / "ep2/events.jsonl").read_bytes(),
            )

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
