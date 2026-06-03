from __future__ import annotations

import json
import unittest

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


if __name__ == "__main__":
    unittest.main()
