from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from beyond_mask.openrouter import OpenRouterClient


class OpenRouterClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_sends_seed_without_deprecated_usage_flag(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "ok",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 1,
                        "total_tokens": 5,
                        "cost": 0.00001,
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        async with OpenRouterClient(
            "test-key",
            base_url="https://openrouter.test/api/v1",
            transport=transport,
        ) as client:
            result = await client.chat(
                "openai/test",
                [{"role": "user", "content": "hello"}],
                temperature=0.0,
                max_tokens=8,
                cache_dir=None,
                seed=42,
            )

        self.assertEqual(result.text, "ok")
        self.assertEqual(result.prompt_tokens, 4)
        self.assertEqual(result.completion_tokens, 1)
        self.assertEqual(result.cost_usd, 0.00001)
        self.assertEqual(captured["body"]["seed"], 42)
        self.assertNotIn("usage", captured["body"])

    async def test_chat_sends_multiturn_messages_and_session_id(self) -> None:
        captured: dict = {}
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "Continue"},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Done.",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 2,
                        "total_tokens": 11,
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        async with OpenRouterClient(
            "test-key",
            base_url="https://openrouter.test/api/v1",
            transport=transport,
        ) as client:
            await client.chat(
                "openai/test",
                messages,
                temperature=0.0,
                max_tokens=8,
                cache_dir=None,
                session_id="mask-row-0",
            )

        self.assertEqual(captured["body"]["messages"], messages)
        self.assertEqual(captured["body"]["session_id"], "mask-row-0")

    async def test_chat_sends_reasoning_config_and_parses_trace_metadata(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "native_finish_reason": "native-stop",
                            "message": {
                                "role": "assistant",
                                "content": "final answer",
                                "reasoning": "visible reasoning",
                                "reasoning_details": [
                                    {
                                        "type": "reasoning.text",
                                        "text": "visible reasoning",
                                        "id": "reasoning-1",
                                        "format": "openai-responses-v1",
                                        "index": 0,
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 6,
                        "total_tokens": 10,
                        "completion_tokens_details": {
                            "reasoning_tokens": 3,
                        },
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        async with OpenRouterClient(
            "test-key",
            base_url="https://openrouter.test/api/v1",
            transport=transport,
        ) as client:
            result = await client.chat(
                "openai/test",
                [{"role": "user", "content": "hello"}],
                temperature=0.0,
                max_tokens=8,
                cache_dir=None,
                reasoning={"effort": "medium", "exclude": False},
            )

        self.assertEqual(
            captured["body"]["reasoning"], {"effort": "medium", "exclude": False}
        )
        self.assertEqual(result.text, "final answer")
        self.assertEqual(result.message["reasoning"], "visible reasoning")
        self.assertEqual(result.reasoning, "visible reasoning")
        self.assertEqual(
            result.reasoning_details,
            [
                {
                    "type": "reasoning.text",
                    "text": "visible reasoning",
                    "id": "reasoning-1",
                    "format": "openai-responses-v1",
                    "index": 0,
                }
            ],
        )
        self.assertEqual(result.native_finish_reason, "native-stop")
        self.assertEqual(result.reasoning_tokens, 3)

    async def test_reasoning_config_is_part_of_cache_key(self) -> None:
        calls: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": body["reasoning"]["effort"],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        with tempfile.TemporaryDirectory() as cache_dir:
            async with OpenRouterClient(
                "test-key",
                base_url="https://openrouter.test/api/v1",
                transport=transport,
            ) as client:
                low = await client.chat(
                    "openai/test",
                    [{"role": "user", "content": "hello"}],
                    temperature=0.0,
                    max_tokens=8,
                    cache_dir=Path(cache_dir),
                    reasoning={"effort": "low", "exclude": False},
                )
                high = await client.chat(
                    "openai/test",
                    [{"role": "user", "content": "hello"}],
                    temperature=0.0,
                    max_tokens=8,
                    cache_dir=Path(cache_dir),
                    reasoning={"effort": "high", "exclude": False},
                )

        self.assertEqual(low.text, "low")
        self.assertEqual(high.text, "high")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
