from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import aiosqlite
import orjson
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential

from subs2anki.db.core import create_cache_schema, create_engine_for_path, resolve_db_path
from subs2anki.llm.types import CachedResult

ModelT = TypeVar("ModelT", bound=BaseModel)


def canonical_json_bytes(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def extract_reasoning(message: object) -> str | None:
    for key in ("reasoning", "reasoning_content"):
        value = getattr(message, key, None)
        if value:
            if isinstance(value, str):
                return value
            return json.dumps(value, ensure_ascii=False)

    model_extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_content"):
        value = model_extra.get(key)
        if value:
            if isinstance(value, str):
                return value
            return json.dumps(value, ensure_ascii=False)

    return None


async def pick_model(client: AsyncOpenAI, requested_model: str | None) -> str:
    if requested_model:
        return requested_model

    models = await client.models.list()
    data = list(models.data)
    if not data:
        raise RuntimeError("No models returned by the configured endpoint.")
    return data[0].id


def response_schema_hash(response_model: type[BaseModel]) -> str:
    return hash_bytes(canonical_json_bytes(response_model.model_json_schema()))


def build_parameters_payload(
    *,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None,
    thinking: bool,
) -> dict[str, Any]:
    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": thinking},
            "reasoning": {"effort": reasoning_effort},
            "cache_prompt": True,
        },
    }


def build_cache_key(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    parameters_payload: dict[str, Any],
    schema_hash: str,
) -> tuple[str, str, str, str, str]:
    system_prompt_hash = hash_bytes(system_prompt.encode("utf-8"))
    user_prompt_hash = hash_bytes(user_prompt.encode("utf-8"))
    parameters_hash = hash_bytes(canonical_json_bytes(parameters_payload))
    cache_key = hash_bytes(
        canonical_json_bytes(
            {
                "model": model,
                "system_prompt_hash": system_prompt_hash,
                "user_prompt_hash": user_prompt_hash,
                "parameters_hash": parameters_hash,
                "schema_hash": schema_hash,
            }
        )
    )
    return cache_key, system_prompt_hash, user_prompt_hash, parameters_hash, schema_hash


class PromptCacheStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def open(self) -> None:
        resolved_path = resolve_db_path(self.path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine_for_path(resolved_path)
        try:
            create_cache_schema(engine)
        finally:
            engine.dispose()

        self._connection = await aiosqlite.connect(resolved_path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._ensure_schema_hash_column()
        await self._connection.execute(
            """
            UPDATE prompt_cache
            SET status = 'pending',
                updated_at = CURRENT_TIMESTAMP
            WHERE status = 'running'
            """
        )
        await self._connection.commit()

    async def _ensure_schema_hash_column(self) -> None:
        cursor = await self.connection.execute("PRAGMA table_info(prompt_cache)")
        rows = await cursor.fetchall()
        await cursor.close()
        columns = {row["name"] for row in rows}
        if "schema_hash" not in columns:
            await self.connection.execute(
                "ALTER TABLE prompt_cache ADD COLUMN schema_hash TEXT NOT NULL DEFAULT ''"
            )
            await self.connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("PromptCacheStore is not open.")
        return self._connection

    async def ensure_row(
        self,
        *,
        cache_key: str,
        model: str,
        system_prompt_hash: str,
        user_prompt_hash: str,
        parameters_hash: str,
        schema_hash: str,
        system_prompt: str,
        user_prompt: str,
        parameters_json: str,
        input_json: str,
    ) -> None:
        async with self._write_lock:
            await self.connection.execute(
                """
                INSERT INTO prompt_cache (
                    cache_key,
                    model,
                    system_prompt_hash,
                    user_prompt_hash,
                    parameters_hash,
                    schema_hash,
                    system_prompt,
                    user_prompt,
                    parameters_json,
                    input_json,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(cache_key) DO UPDATE SET
                    model = excluded.model,
                    system_prompt_hash = excluded.system_prompt_hash,
                    user_prompt_hash = excluded.user_prompt_hash,
                    parameters_hash = excluded.parameters_hash,
                    schema_hash = excluded.schema_hash,
                    system_prompt = excluded.system_prompt,
                    user_prompt = excluded.user_prompt,
                    parameters_json = excluded.parameters_json,
                    input_json = excluded.input_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    cache_key,
                    model,
                    system_prompt_hash,
                    user_prompt_hash,
                    parameters_hash,
                    schema_hash,
                    system_prompt,
                    user_prompt,
                    parameters_json,
                    input_json,
                ),
            )
            await self.connection.commit()

    async def get(self, cache_key: str) -> CachedResult | None:
        cursor = await self.connection.execute(
            """
            SELECT status, response_json, reasoning_content, elapsed_ms, error_text
            FROM prompt_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None

        return CachedResult(
            cache_key=cache_key,
            status=row["status"],
            response_json=row["response_json"],
            reasoning_content=row["reasoning_content"],
            elapsed_ms=row["elapsed_ms"],
            error_text=row["error_text"],
            cache_hit=row["status"] == "completed" and row["response_json"] is not None,
        )

    async def mark_running(self, cache_key: str) -> None:
        async with self._write_lock:
            await self.connection.execute(
                """
                UPDATE prompt_cache
                SET status = 'running',
                    error_text = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE cache_key = ?
                """,
                (cache_key,),
            )
            await self.connection.commit()

    async def mark_completed(
        self,
        *,
        cache_key: str,
        response: BaseModel,
        reasoning_content: str | None,
        elapsed_ms: float,
    ) -> None:
        async with self._write_lock:
            await self.connection.execute(
                """
                UPDATE prompt_cache
                SET status = 'completed',
                    response_json = ?,
                    reasoning_content = ?,
                    elapsed_ms = ?,
                    error_text = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE cache_key = ?
                """,
                (
                    response.model_dump_json(),
                    reasoning_content,
                    elapsed_ms,
                    cache_key,
                ),
            )
            await self.connection.commit()

    async def mark_failed(self, *, cache_key: str, error_text: str) -> None:
        async with self._write_lock:
            await self.connection.execute(
                """
                UPDATE prompt_cache
                SET status = 'failed',
                    error_text = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE cache_key = ?
                """,
                (error_text, cache_key),
            )
            await self.connection.commit()


async def run_cached_structured_completion(
    *,
    client: AsyncOpenAI | None,
    cache: PromptCacheStore,
    model: str,
    system_prompt: str,
    user_prompt: str,
    input_payload: dict[str, Any],
    response_model: type[ModelT],
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None,
    max_retries: int,
    thinking: bool,
    sanitizer: Callable[[ModelT], ModelT] | None = None,
    fallback_response: ModelT | None = None,
) -> tuple[ModelT | None, dict[str, Any]]:
    schema_hash = response_schema_hash(response_model)
    parameters_payload = build_parameters_payload(
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
    )
    cache_key, system_prompt_hash, user_prompt_hash, parameters_hash, schema_hash = build_cache_key(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        parameters_payload=parameters_payload,
        schema_hash=schema_hash,
    )
    await cache.ensure_row(
        cache_key=cache_key,
        model=model,
        system_prompt_hash=system_prompt_hash,
        user_prompt_hash=user_prompt_hash,
        parameters_hash=parameters_hash,
        schema_hash=schema_hash,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        parameters_json=canonical_json_bytes(parameters_payload).decode("utf-8"),
        input_json=orjson.dumps(input_payload).decode("utf-8"),
    )

    cached = await cache.get(cache_key)
    if cached is not None and cached.status == "completed" and cached.response_json is not None:
        try:
            parsed = response_model.model_validate_json(cached.response_json)
            if sanitizer is not None:
                parsed = sanitizer(parsed)
            return (
                parsed,
                {
                    "cache_hit": True,
                    "elapsed_ms": cached.elapsed_ms,
                    "status": "completed",
                    "error_text": None,
                },
            )
        except Exception as exc:
            await cache.mark_failed(cache_key=cache_key, error_text=f"Cached response invalid: {exc}")

    if client is None:
        raise RuntimeError(
            f"Cache miss for {cache_key} and no OpenAI-compatible client is configured."
        )

    await cache.mark_running(cache_key)

    started = time.perf_counter()
    try:
        completion = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_random_exponential(multiplier=2, max=30),
            retry=(
                lambda retry_state: isinstance(
                    retry_state.outcome.exception(),
                    (RateLimitError, APITimeoutError, APIConnectionError, APIError),
                )
                if retry_state.outcome and retry_state.outcome.failed
                else False
            ),
            reraise=True,
        ):
            with attempt:
                completion = await client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    response_format=response_model,
                    max_tokens=max_tokens,
                    extra_body=parameters_payload["extra_body"],
                )

        elapsed_ms = (time.perf_counter() - started) * 1000
        if completion is None:
            raise RuntimeError("Model request did not produce a completion.")

        message = completion.choices[0].message
        parsed = message.parsed
        if parsed is None:
            refusal = getattr(message, "refusal", None)
            raise RuntimeError(
                f"Model did not return parsed structured output. Refusal: {refusal}"
            )

        if sanitizer is not None:
            parsed = sanitizer(parsed)

        reasoning_content = extract_reasoning(message)
        await cache.mark_completed(
            cache_key=cache_key,
            response=parsed,
            reasoning_content=reasoning_content,
            elapsed_ms=elapsed_ms,
        )
        return (
            parsed,
            {
                "cache_hit": False,
                "elapsed_ms": elapsed_ms,
                "status": "completed",
                "error_text": None,
            },
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        if fallback_response is not None:
            await cache.mark_completed(
                cache_key=cache_key,
                response=fallback_response,
                reasoning_content=f"fallback due to error: {exc}",
                elapsed_ms=elapsed_ms,
            )
            return (
                fallback_response,
                {
                    "cache_hit": False,
                    "elapsed_ms": elapsed_ms,
                    "status": "completed",
                    "error_text": str(exc),
                },
            )

        await cache.mark_failed(cache_key=cache_key, error_text=str(exc))
        return (
            None,
            {
                "cache_hit": False,
                "elapsed_ms": None,
                "status": "failed",
                "error_text": str(exc),
            },
        )
