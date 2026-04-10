#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "aiosqlite>=0.20.0",
#   "openai>=1.75.0",
#   "orjson>=3.10.16",
#   "pydantic>=2.11.0",
#   "python-dotenv>=1.0.1",
#   "tenacity>=9.0.0",
#   "typer>=0.16.0",
# ]
# ///

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Literal

import aiosqlite
import orjson
import typer
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential

app = typer.Typer(no_args_is_help=True, add_completion=False)

ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ROOT_ENV_PATH)

DEFAULT_INPUT_PATH = Path("data/lemma_normalisation_preprocess.jsonl")
DEFAULT_ENTRIES_INPUT_PATH = Path("data/lemma_entries.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/translation_pass.head25.jsonl")
DEFAULT_CACHE_PATH = Path("data/translation_pass_cache.sqlite3")
DEFAULT_BASE_URL = os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL")
DEFAULT_TOKEN = os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY")
DEFAULT_MODEL = "qwen/qwen3.5-122b-a10b"
DEFAULT_REASONING_EFFORT = "none"
TEMPERATURE = 0.1
MAX_TOKENS = 256
MAX_RETRIES = 5

SYSTEM_PROMPT = """You are a Modern Greek to English learner-facing translator.

Task:
Given one normalized Modern Greek lexeme candidate, translate only `normalized_form` into English.

Rules:
- Translate the normalized form, not the example sentence.
- Use `original_forms`, `source_candidate_lemmas`, and `sentences` for disambiguation.
- Prefer English dictionary/base-form meanings, not every inflected English form.
- For verbs, return base verb meanings only; do not include conjugated English forms unless they express a distinct meaning.
- Return all common learner-relevant meanings as concise English strings.
- Do not artificially narrow the answer to only one subtitle sense.
- Function words must be translated or explained as learner-facing English meanings.
- Include articles, particles, prepositions, conjunctions, and pronouns when relevant.
- Do not include Greek transliteration.
- Do not include justifications.
- Do not translate the full sentence.
- Example: `ο` should return `the`, not `a` or `an`.
- Example: `είμαι` should return `be`, not separate forms like `am`, `is`, and `are`.
- Example: `έχω` should return `have`, `possess`, or `own`, not `has` or `had`.

Output format:
Return JSON with exactly this shape:
{"translations":["..."]}
"""


class SentenceContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    sentence_text: str
    source_srt: str


class NormalizedRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    original_forms: list[str] = Field(..., min_length=1)
    normalized_form: str
    word_class: str


class LemmaEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    lemmatised_form: str
    count: int
    original_forms: list[str] = Field(..., min_length=1)
    sentences: list[SentenceContext] = Field(default_factory=list)


class TranslationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_form: str
    word_class: str
    original_forms: list[str]
    source_candidate_lemmas: list[str]
    occurrence_count: int
    sentences: list[SentenceContext]
    example_sentence: SentenceContext | None


class TranslationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translations: list[str] = Field(..., min_length=1)


class CachedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_key: str
    status: Literal["completed", "failed", "pending", "running"]
    response_json: str | None = None
    reasoning_content: str | None = None
    elapsed_ms: float | None = None
    error_text: str | None = None
    cache_hit: bool = False


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


def build_parameters_payload(*, thinking: bool, reasoning_effort: str) -> dict[str, Any]:
    return {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
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


def load_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    rows: list[BaseModel] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            rows.append(model.model_validate(orjson.loads(raw_line)))
        except Exception as exc:
            raise RuntimeError(f"Invalid JSONL row at {path}:{line_number}: {exc}") from exc
    return rows


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def unique_sentences(sentences: list[SentenceContext]) -> list[SentenceContext]:
    seen: set[tuple[str, str]] = set()
    unique: list[SentenceContext] = []
    for sentence in sentences:
        key = (sentence.source_srt, sentence.sentence_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sentence)
    return unique


def form_pattern(form: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(form)}(?!\w)", re.IGNORECASE)


def sentence_matches_form(sentence_text: str, form: str) -> bool:
    return bool(form_pattern(form).search(sentence_text))


def pick_example_sentence(candidate: TranslationCandidate) -> SentenceContext | None:
    if not candidate.sentences:
        return None
    seed_payload = canonical_json_bytes(
        {
            "normalized_form": candidate.normalized_form,
            "word_class": candidate.word_class,
            "original_forms": candidate.original_forms,
        }
    )
    seed = int(hash_bytes(seed_payload)[:16], 16)
    return random.Random(seed).choice(candidate.sentences)


def build_candidates(
    normalized_rows: list[NormalizedRow],
    lemma_entries: list[LemmaEntry],
    limit: int | None,
    context_sentence_limit: int,
) -> list[TranslationCandidate]:
    candidates: list[TranslationCandidate] = []
    for row in normalized_rows:
        row_forms = set(row.original_forms)
        matching_entries = [entry for entry in lemma_entries if row_forms.intersection(entry.original_forms)]
        source_candidate_lemmas = unique_strings(
            [entry.lemmatised_form for entry in matching_entries if entry.lemmatised_form]
        )
        occurrence_count = sum(entry.count for entry in matching_entries)
        matching_sentences = unique_sentences(
            [
                sentence
                for entry in matching_entries
                for sentence in entry.sentences
                if any(sentence_matches_form(sentence.sentence_text, form) for form in row.original_forms)
            ]
        )
        candidate = TranslationCandidate(
            normalized_form=row.normalized_form,
            word_class=row.word_class,
            original_forms=unique_strings(row.original_forms),
            source_candidate_lemmas=source_candidate_lemmas,
            occurrence_count=occurrence_count,
            sentences=matching_sentences,
            example_sentence=None,
        )
        candidate.example_sentence = pick_example_sentence(candidate)
        candidate.sentences = candidate.sentences[:context_sentence_limit]
        candidates.append(candidate)
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def build_translation_prompt_payload(candidate: TranslationCandidate) -> dict[str, Any]:
    payload = candidate.model_dump(mode="json")
    payload.pop("word_class", None)
    return payload


def build_user_prompt(candidate: TranslationCandidate) -> str:
    return json.dumps(build_translation_prompt_payload(candidate), ensure_ascii=False)


def sanitize_translation_response(response: TranslationResponse) -> TranslationResponse:
    translations = unique_strings(
        [translation.strip() for translation in response.translations if translation.strip()]
    )
    if not translations:
        raise RuntimeError("Translation response was empty after sanitization.")
    return TranslationResponse(translations=translations)


class CacheStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                system_prompt_hash TEXT NOT NULL,
                user_prompt_hash TEXT NOT NULL,
                parameters_hash TEXT NOT NULL,
                schema_hash TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                input_json TEXT NOT NULL,
                status TEXT NOT NULL,
                response_json TEXT,
                reasoning_content TEXT,
                elapsed_ms REAL,
                error_text TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._connection.execute(
            """
            UPDATE prompt_cache
            SET status = 'pending',
                updated_at = CURRENT_TIMESTAMP
            WHERE status = 'running'
            """
        )
        await self._connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("CacheStore is not open.")
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
                (response.model_dump_json(), reasoning_content, elapsed_ms, cache_key),
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
    client: AsyncOpenAI,
    cache: CacheStore,
    model: str,
    system_prompt: str,
    user_prompt: str,
    input_payload: dict[str, Any],
    response_model: type[BaseModel],
    thinking: bool,
    reasoning_effort: str,
) -> tuple[BaseModel | None, dict[str, Any]]:
    schema_hash = response_schema_hash(response_model)
    parameters_payload = build_parameters_payload(thinking=thinking, reasoning_effort=reasoning_effort)
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
            if isinstance(parsed, TranslationResponse):
                parsed = sanitize_translation_response(parsed)
            return parsed, {
                "cache_hit": True,
                "elapsed_ms": cached.elapsed_ms,
                "status": "completed",
                "error_text": None,
            }
        except Exception as exc:
            await cache.mark_failed(cache_key=cache_key, error_text=f"Cached response invalid: {exc}")

    await cache.mark_running(cache_key)
    started = time.perf_counter()
    try:
        completion = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRIES),
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
                    temperature=TEMPERATURE,
                    response_format=response_model,
                    max_tokens=MAX_TOKENS,
                    extra_body=parameters_payload["extra_body"],
                )

        elapsed_ms = (time.perf_counter() - started) * 1000
        if completion is None:
            raise RuntimeError("Model request did not produce a completion.")

        message = completion.choices[0].message
        parsed = message.parsed
        if parsed is None:
            refusal = getattr(message, "refusal", None)
            raise RuntimeError(f"Model did not return parsed structured output. Refusal: {refusal}")

        if isinstance(parsed, TranslationResponse):
            parsed = sanitize_translation_response(parsed)

        reasoning_content = extract_reasoning(message)
        await cache.mark_completed(
            cache_key=cache_key,
            response=parsed,
            reasoning_content=reasoning_content,
            elapsed_ms=elapsed_ms,
        )
        return parsed, {
            "cache_hit": False,
            "elapsed_ms": elapsed_ms,
            "status": "completed",
            "error_text": None,
        }
    except Exception as exc:
        await cache.mark_failed(cache_key=cache_key, error_text=str(exc))
        return None, {
            "cache_hit": False,
            "elapsed_ms": None,
            "status": "failed",
            "error_text": str(exc),
        }


async def translate_candidate(
    *,
    client: AsyncOpenAI,
    cache: CacheStore,
    model: str,
    candidate: TranslationCandidate,
    thinking: bool,
    reasoning_effort: str,
) -> dict[str, Any]:
    translation, cache_entry = await run_cached_structured_completion(
        client=client,
        cache=cache,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(candidate),
        input_payload=build_translation_prompt_payload(candidate),
        response_model=TranslationResponse,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
    if translation is None:
        return {"row": None, "cache": cache_entry}
    if not isinstance(translation, TranslationResponse):
        raise RuntimeError("Translation response did not match the expected response model.")

    return {
        "row": {
            "normalized_form": candidate.normalized_form,
            "word_class": candidate.word_class,
            "translations": translation.translations,
            "example_sentence": (
                candidate.example_sentence.model_dump(mode="json")
                if candidate.example_sentence is not None
                else None
            ),
            "original_forms": candidate.original_forms,
            "source_candidate_lemmas": candidate.source_candidate_lemmas,
            "occurrence_count": candidate.occurrence_count,
        },
        "cache": cache_entry,
    }


def write_jsonl(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for result in results:
            row = result.get("row")
            if row is None:
                continue
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


async def run_translation_pass(
    *,
    input_path: Path,
    entries_input_path: Path,
    output_path: Path,
    cache_path: Path,
    base_url: str,
    api_key: str,
    model_name: str | None,
    concurrency: int,
    thinking: bool,
    reasoning_effort: str,
    limit: int | None,
    context_sentence_limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    normalized_rows = [row for row in load_jsonl(input_path, NormalizedRow) if isinstance(row, NormalizedRow)]
    lemma_entries = [entry for entry in load_jsonl(entries_input_path, LemmaEntry) if isinstance(entry, LemmaEntry)]
    candidates = build_candidates(normalized_rows, lemma_entries, limit, context_sentence_limit)

    cache = CacheStore(cache_path)
    await cache.open()
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    model = await pick_model(client, model_name)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(candidate: TranslationCandidate) -> dict[str, Any]:
        async with semaphore:
            return await translate_candidate(
                client=client,
                cache=cache,
                model=model,
                candidate=candidate,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
            )

    try:
        results = await asyncio.gather(*(worker(candidate) for candidate in candidates))
    finally:
        await client.close()
        await cache.close()

    write_jsonl(results, output_path)
    return model, results


@app.command()
def main(
    input_path: Path = typer.Option(
        DEFAULT_INPUT_PATH,
        "--input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the normalized lemma JSONL file.",
    ),
    entries_input_path: Path = typer.Option(
        DEFAULT_ENTRIES_INPUT_PATH,
        "--entries-input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the source lemma entries JSONL file.",
    ),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--output",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save the translation JSONL output.",
    ),
    cache_path: Path = typer.Option(
        DEFAULT_CACHE_PATH,
        "--cache-db",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="SQLite cache path for prompt/response persistence.",
    ),
    concurrency: int = typer.Option(
        10,
        "--concurrency",
        min=1,
        help="Maximum number of concurrent model requests.",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL or "",
        "--base-url",
        help="OpenAI-compatible base URL.",
    ),
    token: str = typer.Option(
        DEFAULT_TOKEN or "",
        "--token",
        help="API token for the OpenAI-compatible endpoint.",
    ),
    model: str | None = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Model name. If omitted, the first model from /v1/models is used.",
    ),
    reasoning_effort: str = typer.Option(
        DEFAULT_REASONING_EFFORT,
        "--reasoning-effort",
        help="Reasoning effort passed through the provider extra_body.",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable or disable model thinking mode.",
    ),
    limit: int | None = typer.Option(
        25,
        "--limit",
        min=1,
        help="Maximum number of raw-order normalized rows to translate.",
    ),
    context_sentence_limit: int = typer.Option(
        25,
        "--context-sentences",
        min=1,
        help="Maximum matching sentence contexts to send per translation prompt.",
    ),
) -> None:
    if not base_url:
        typer.echo(f"Missing base URL. Set BASE_URL in {ROOT_ENV_PATH} or pass --base-url.", err=True)
        raise typer.Exit(code=1)
    if not token:
        typer.echo(f"Missing API token. Set OPENAI_TOKEN in {ROOT_ENV_PATH} or pass --token.", err=True)
        raise typer.Exit(code=1)

    try:
        resolved_model, results = asyncio.run(
            run_translation_pass(
                input_path=input_path,
                entries_input_path=entries_input_path,
                output_path=output_path,
                cache_path=cache_path,
                base_url=base_url,
                api_key=token,
                model_name=model,
                concurrency=concurrency,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                limit=limit,
                context_sentence_limit=context_sentence_limit,
            )
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"model={resolved_model}")
    typer.echo(f"output={output_path}")
    typer.echo(f"cache_db={cache_path}")
    typer.echo(f"candidate_count={len(results)}")
    typer.echo(f"completed={sum(1 for result in results if result['cache']['status'] == 'completed')}")
    typer.echo(f"failed={sum(1 for result in results if result['cache']['status'] == 'failed')}")
    typer.echo(f"cache_hits={sum(1 for result in results if result['cache']['cache_hit'])}")


if __name__ == "__main__":
    app()
