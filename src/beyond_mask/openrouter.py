from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_DELAYS = (1.0, 2.0, 4.0)


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    finish_reason: str | None
    native_finish_reason: str | None
    latency_ms: int
    cached: bool
    error: str | None
    message: dict | None = None
    reasoning: str | None = None
    reasoning_details: list[dict] | None = None
    reasoning_tokens: int | None = None
    raw: dict | None = None


def _cache_key(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    seed: int | None,
    session_id: str | None,
    reasoning: dict | None,
    response_format: dict | None,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "session_id": session_id,
            "reasoning": reasoning,
            "response_format": response_format,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cached(path: Path) -> ChatResult | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return ChatResult(
        text=data.get("text", ""),
        prompt_tokens=data.get("prompt_tokens"),
        completion_tokens=data.get("completion_tokens"),
        cost_usd=data.get("cost_usd"),
        finish_reason=data.get("finish_reason"),
        native_finish_reason=data.get("native_finish_reason"),
        latency_ms=0,
        cached=True,
        error=None,
        message=data.get("message"),
        reasoning=data.get("reasoning"),
        reasoning_details=data.get("reasoning_details"),
        reasoning_tokens=data.get("reasoning_tokens"),
        raw=data.get("raw"),
    )


def _store_cached(path: Path, result: ChatResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(result)
    # cached flag and latency are runtime-only
    payload.pop("cached", None)
    payload.pop("latency_ms", None)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_usage(usage: dict | None) -> tuple[int | None, int | None, float | None]:
    if not usage:
        return None, None, None
    return (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("cost"),
    )


def _parse_reasoning_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        return None
    return completion_details.get("reasoning_tokens")


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        concurrency: int = 8,
        timeout: float = 60.0,
        referer: str = "https://github.com/anthropics/neurips26-beyond_mask",
        title: str = "neurips26-beyond_mask",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": referer,
                "X-Title": title,
            },
            timeout=timeout,
            transport=transport,
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        cache_dir: Path | None,
        seed: int | None = None,
        session_id: str | None = None,
        reasoning: dict | None = None,
        response_format: dict | None = None,
    ) -> ChatResult:
        cache_path: Path | None = None
        if cache_dir is not None:
            cache_path = cache_dir / f"{_cache_key(model, messages, temperature, max_tokens, seed, session_id, reasoning, response_format)}.json"
            hit = _load_cached(cache_path)
            if hit is not None:
                return hit

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        if session_id is not None:
            body["session_id"] = session_id
        if reasoning is not None:
            body["reasoning"] = reasoning
        if response_format is not None:
            body["response_format"] = response_format

        start = time.perf_counter()
        async with self._sem:
            result = await self._post_with_retries(body)
        result.latency_ms = int((time.perf_counter() - start) * 1000)

        if cache_path is not None and result.error is None:
            _store_cached(cache_path, result)
        return result

    async def _post_with_retries(self, body: dict) -> ChatResult:
        last_error: str | None = None
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                response = await self._client.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < len(RETRY_DELAYS):
                    await asyncio.sleep(RETRY_DELAYS[attempt])
                    continue
                break

            if response.status_code in RETRY_STATUSES and attempt < len(RETRY_DELAYS):
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else RETRY_DELAYS[attempt]
                except ValueError:
                    delay = RETRY_DELAYS[attempt]
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 400:
                snippet = response.text[:500]
                return ChatResult(
                    text="",
                    prompt_tokens=None,
                    completion_tokens=None,
                    cost_usd=None,
                    finish_reason=None,
                    native_finish_reason=None,
                    latency_ms=0,
                    cached=False,
                    error=f"HTTP {response.status_code}: {snippet}",
                )

            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            prompt_tokens, completion_tokens, cost_usd = _parse_usage(data.get("usage"))
            return ChatResult(
                text=message.get("content") or "",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                finish_reason=choice.get("finish_reason"),
                native_finish_reason=choice.get("native_finish_reason"),
                latency_ms=0,
                cached=False,
                error=None,
                message=message,
                reasoning=message.get("reasoning"),
                reasoning_details=message.get("reasoning_details"),
                reasoning_tokens=_parse_reasoning_tokens(data.get("usage")),
                raw=data,
            )

        return ChatResult(
            text="",
            prompt_tokens=None,
            completion_tokens=None,
            cost_usd=None,
            finish_reason=None,
            native_finish_reason=None,
            latency_ms=0,
            cached=False,
            error=last_error or "exceeded retries",
        )
