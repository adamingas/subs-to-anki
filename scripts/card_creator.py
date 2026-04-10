#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "aiosqlite>=0.20.0",
#   "genanki>=0.13.1",
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
import html
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Literal

import aiosqlite
import genanki
import orjson
import typer
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ConfigDict, Field
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential

app = typer.Typer(no_args_is_help=True, add_completion=False)

ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ROOT_ENV_PATH)

DEFAULT_REVIEW_INPUT_PATH = Path("data/lemma_review_results.json")
DEFAULT_ENTRIES_INPUT_PATH = Path("data/lemma_entries.jsonl")
DEFAULT_OUTPUT_JSON_PATH = Path("data/card_creator_results.json")
DEFAULT_OUTPUT_TSV_PATH = Path("data/anki_cards.tsv")
DEFAULT_OUTPUT_APKG_PATH = Path("data/anki_cards.apkg")
DEFAULT_CACHE_PATH = Path("data/card_creator_cache.sqlite3")
DEFAULT_BASE_URL = os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL")
DEFAULT_TOKEN = os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY")
DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_DECK_NAME = "Greek Lemmas"
TEMPERATURE = 0.2
MAX_TOKENS = 256
MAX_RETRIES = 5

SYSTEM_PROMPT = """You are creating concise learner-facing English glosses for Modern Greek Anki cards.

Task:
Given a reviewed learner-facing normalized form, plus the observed original forms and Greek sentence
context, return the best short English translation for the target word.

Rules:
- Translate `normalized_form`, not the example sentence.
- Use `original_forms` and `sentences` to disambiguate the sense.
- Return a short learner-facing gloss or a short semicolon-separated gloss set.
- Prefer lowercase unless the word is a proper noun.
- Keep the answer compact enough to fit naturally on the back of a flashcard.
- Do not explain, justify, transliterate, or add usage notes.
- Do not include quotation marks.

Input JSON:
{"normalized_form":"ηρεμώ","source_candidate_lemmas":["ηρέμησε","ηρεμήσω"],"original_forms":["ηρέμησε","Ηρέμησε","ηρεμήσω"],"sentences":[{"sentence_text":"Εντάξει, ηρέμησε.","source_srt":"data/raw/example.srt"},{"sentence_text":"Θέλω να ηρεμήσω και δεν θέλω να πάρω χάπια.","source_srt":"data/raw/example.srt"}],"occurrence_count":7}
Output JSON:
{"english_translation":"calm down; relax"}

Input JSON:
{"normalized_form":"σπίτι","source_candidate_lemmas":["σπίτι"],"original_forms":["σπίτι"],"sentences":[{"sentence_text":"Χωρίς συγγνώμη, σπίτι δεν γυρίζεις.","source_srt":"data/raw/example.srt"}],"occurrence_count":14}
Output JSON:
{"english_translation":"home; house"}

Output format:
Return JSON with exactly this shape:
{"english_translation":"..."}
"""

SRT_TAG_PATTERN = re.compile(r"\{\\[^}]+\}")
ANKI_MODEL_ID = 1387426501


class SentenceContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    sentence_text: str
    source_srt: str


class ReviewExample(BaseModel):
    model_config = ConfigDict(extra="allow")

    file: str
    text: str


class ReviewCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_lemma: str
    examples: list[ReviewExample] = Field(default_factory=list)
    occurrence_count: int


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_lemma: str
    citation_lemma: str
    confidence: str | None = None


class ReviewItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    cache: dict[str, Any] | None = None
    candidate: ReviewCandidate
    review: ReviewDecision


class ReviewResultsFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    items: list[ReviewItem]


class ReviewJsonlRowV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    original_forms: list[str]
    citation_form: str
    confidence: str | None = None


class ReviewJsonlRowV2(BaseModel):
    model_config = ConfigDict(extra="allow")

    original_forms: list[str]
    normalized_form: str
    word_class: str | None = None


class UnifiedReviewRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_form: str
    source_candidate_lemma: str | None = None
    original_forms: list[str] = Field(default_factory=list)
    occurrence_count: int | None = None


class LemmaEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    lemmatised_form: str
    count: int
    original_forms: list[str]
    sentences: list[SentenceContext]


class TranslationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_form: str
    source_candidate_lemmas: list[str]
    original_forms: list[str]
    sentences: list[SentenceContext]
    occurrence_count: int


class TranslationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    english_translation: str = Field(
        ...,
        min_length=1,
        description="Short learner-facing English translation for the normalized Modern Greek form.",
    )


class CardOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_form: str
    english_translation: str
    sentence: SentenceContext
    front: str
    back_html: str
    source_candidate_lemmas: list[str]
    original_forms: list[str]
    occurrence_count: int


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


def clean_sentence_text(text: str) -> str:
    text = SRT_TAG_PATTERN.sub(" ", text)
    return " ".join(text.split())


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


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


def stable_int(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def pick_sentence(candidate: TranslationCandidate, seed: int) -> SentenceContext:
    if not candidate.sentences:
        raise RuntimeError(f"No sentence context available for normalized form {candidate.normalized_form!r}.")

    chooser = random.Random(stable_int(str(seed), candidate.normalized_form))
    return chooser.choice(candidate.sentences)


def build_card_output(candidate: TranslationCandidate, translation: TranslationResponse, seed: int) -> CardOutput:
    sentence = pick_sentence(candidate, seed)
    english_translation = translation.english_translation.strip()
    front = candidate.normalized_form
    back_html = f"{html.escape(english_translation)}<br><br>{html.escape(sentence.sentence_text)}"
    return CardOutput(
        normalized_form=candidate.normalized_form,
        english_translation=english_translation,
        sentence=sentence,
        front=front,
        back_html=back_html,
        source_candidate_lemmas=candidate.source_candidate_lemmas,
        original_forms=candidate.original_forms,
        occurrence_count=candidate.occurrence_count,
    )


async def pick_model(client: AsyncOpenAI, requested_model: str | None) -> str:
    if requested_model:
        return requested_model

    models = await client.models.list()
    data = list(models.data)
    if not data:
        raise RuntimeError("No models returned by the configured endpoint.")
    return data[0].id


def build_user_prompt(candidate: TranslationCandidate) -> str:
    return json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False)


def build_parameters_payload(
    *,
    response_model: type[BaseModel],
    schema_name: str,
    thinking: bool,
) -> dict[str, Any]:
    return {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": response_model.model_json_schema(),
            },
        },
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": thinking},
            "cache_prompt": True,
        },
    }


def build_cache_key(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    parameters_payload: dict[str, Any],
) -> tuple[str, str, str, str]:
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
            }
        )
    )
    return cache_key, system_prompt_hash, user_prompt_hash, parameters_hash


def load_review_results(input_path: Path) -> list[UnifiedReviewRow]:
    if input_path.suffix == ".jsonl":
        rows: list[UnifiedReviewRow] = []
        for line_number, raw_line in enumerate(input_path.read_bytes().splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                payload = orjson.loads(raw_line)
                if "normalized_form" in payload:
                    row = ReviewJsonlRowV2.model_validate(payload)
                    normalized_form = row.normalized_form
                else:
                    row = ReviewJsonlRowV1.model_validate(payload)
                    normalized_form = row.citation_form
                rows.append(
                    UnifiedReviewRow(
                        normalized_form=normalized_form.strip(),
                        original_forms=unique_preserving_order(row.original_forms),
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid review JSONL row at {input_path}:{line_number}: {exc}"
                ) from exc
        return rows

    try:
        payload = orjson.loads(input_path.read_bytes())
        review_file = ReviewResultsFile.model_validate(payload)
    except Exception as exc:
        raise RuntimeError(f"Invalid review JSON at {input_path}: {exc}") from exc

    return [
        UnifiedReviewRow(
            normalized_form=item.review.citation_lemma.strip(),
            source_candidate_lemma=item.candidate.candidate_lemma.strip(),
            original_forms=[],
            occurrence_count=item.candidate.occurrence_count,
        )
        for item in review_file.items
    ]


def load_entries(input_path: Path) -> dict[str, LemmaEntry]:
    entries: dict[str, LemmaEntry] = {}
    for line_number, raw_line in enumerate(input_path.read_bytes().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = orjson.loads(raw_line)
            entry = LemmaEntry.model_validate(payload)
        except Exception as exc:
            raise RuntimeError(f"Invalid JSONL lemma entry at {input_path}:{line_number}: {exc}") from exc

        cleaned_entry = LemmaEntry(
            lemmatised_form=entry.lemmatised_form.strip(),
            count=entry.count,
            original_forms=unique_preserving_order(entry.original_forms),
            sentences=[
                SentenceContext(
                    sentence_text=clean_sentence_text(sentence.sentence_text),
                    source_srt=sentence.source_srt,
                )
                for sentence in entry.sentences
                if clean_sentence_text(sentence.sentence_text)
            ],
        )

        existing = entries.get(cleaned_entry.lemmatised_form)
        if existing is None:
            entries[cleaned_entry.lemmatised_form] = cleaned_entry
            continue

        entries[cleaned_entry.lemmatised_form] = LemmaEntry(
            lemmatised_form=cleaned_entry.lemmatised_form,
            count=existing.count + cleaned_entry.count,
            original_forms=unique_preserving_order(existing.original_forms + cleaned_entry.original_forms),
            sentences=dedupe_sentences(existing.sentences + cleaned_entry.sentences),
        )
    return entries


def dedupe_sentences(sentences: list[SentenceContext]) -> list[SentenceContext]:
    seen: set[tuple[str, str]] = set()
    ordered: list[SentenceContext] = []
    for sentence in sentences:
        key = (sentence.sentence_text, sentence.source_srt)
        if not sentence.sentence_text or key in seen:
            continue
        seen.add(key)
        ordered.append(sentence)
    return ordered


def find_matching_entry(original_forms: list[str], entries: dict[str, LemmaEntry]) -> LemmaEntry:
    row_forms = set(original_forms)
    if not row_forms:
        raise RuntimeError("Review row did not contain any original_forms.")

    matches: list[tuple[int, int, LemmaEntry]] = []
    for entry in entries.values():
        entry_forms = set(entry.original_forms)
        if row_forms.issubset(entry_forms):
            extra_forms = len(entry_forms) - len(row_forms)
            matches.append((extra_forms, -entry.count, entry))

    if not matches:
        raise RuntimeError(
            "Could not match review row original_forms to any lemma entry: "
            f"{sorted(row_forms)}"
        )

    matches.sort(key=lambda item: (item[0], item[1], item[2].lemmatised_form))
    return matches[0][2]


def build_translation_candidates(
    *,
    review_rows: list[UnifiedReviewRow],
    entries: dict[str, LemmaEntry],
) -> tuple[list[TranslationCandidate], list[str]]:
    grouped: dict[str, dict[str, Any]] = {}
    missing_entries: list[str] = []

    for item in review_rows:
        normalized_form = item.normalized_form.strip()
        if not normalized_form:
            raise RuntimeError("Found review item with an empty citation_lemma.")

        bucket = grouped.setdefault(
            normalized_form,
            {
                "source_candidate_lemmas": [],
                "original_forms": [],
                "sentences": [],
                "occurrence_count": 0,
            },
        )
        if item.source_candidate_lemma is not None:
            source_candidate_lemma = item.source_candidate_lemma.strip()
            if not source_candidate_lemma:
                raise RuntimeError("Found review item with an empty candidate lemma.")
            bucket["source_candidate_lemmas"].append(source_candidate_lemma)
            bucket["occurrence_count"] += item.occurrence_count or 0

            entry = entries.get(source_candidate_lemma)
            if entry is None:
                missing_entries.append(source_candidate_lemma)
                continue

            bucket["original_forms"].extend(entry.original_forms)
            bucket["sentences"].extend(entry.sentences)
            continue

        entry = find_matching_entry(item.original_forms, entries)
        bucket["source_candidate_lemmas"].append(entry.lemmatised_form)
        bucket["occurrence_count"] += entry.count
        bucket["original_forms"].extend(item.original_forms or entry.original_forms)
        bucket["sentences"].extend(entry.sentences)

    candidates: list[TranslationCandidate] = []
    for normalized_form, bucket in grouped.items():
        sentences = dedupe_sentences(bucket["sentences"])
        if not sentences:
            raise RuntimeError(
                f"No lemma entry sentences found for normalized form {normalized_form!r}. "
                "Expected matching source lemmas in data/lemma_entries.jsonl."
            )

        original_forms = unique_preserving_order(bucket["original_forms"])
        if not original_forms:
            original_forms = unique_preserving_order(bucket["source_candidate_lemmas"])

        candidates.append(
            TranslationCandidate(
                normalized_form=normalized_form,
                source_candidate_lemmas=unique_preserving_order(bucket["source_candidate_lemmas"]),
                original_forms=original_forms,
                sentences=sentences,
                occurrence_count=bucket["occurrence_count"],
            )
        )

    return candidates, unique_preserving_order(missing_entries)


class CacheStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def open(self) -> None:
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
                    system_prompt,
                    user_prompt,
                    parameters_json,
                    input_json,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(cache_key) DO UPDATE SET
                    model = excluded.model,
                    system_prompt_hash = excluded.system_prompt_hash,
                    user_prompt_hash = excluded.user_prompt_hash,
                    parameters_hash = excluded.parameters_hash,
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

        response_json = row["response_json"]
        return CachedResult(
            cache_key=cache_key,
            status=row["status"],
            response_json=response_json,
            reasoning_content=row["reasoning_content"],
            elapsed_ms=row["elapsed_ms"],
            error_text=row["error_text"],
            cache_hit=row["status"] == "completed" and response_json is not None,
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
    client: AsyncOpenAI,
    cache: CacheStore,
    model: str,
    system_prompt: str,
    user_prompt: str,
    input_payload: dict[str, Any],
    response_model: type[BaseModel],
    schema_name: str,
    thinking: bool,
) -> tuple[BaseModel | None, dict[str, Any]]:
    parameters_payload = build_parameters_payload(
        response_model=response_model,
        schema_name=schema_name,
        thinking=thinking,
    )
    cache_key, system_prompt_hash, user_prompt_hash, parameters_hash = build_cache_key(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        parameters_payload=parameters_payload,
    )
    await cache.ensure_row(
        cache_key=cache_key,
        model=model,
        system_prompt_hash=system_prompt_hash,
        user_prompt_hash=user_prompt_hash,
        parameters_hash=parameters_hash,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        parameters_json=canonical_json_bytes(parameters_payload).decode("utf-8"),
        input_json=orjson.dumps(input_payload).decode("utf-8"),
    )

    cached = await cache.get(cache_key)
    if cached is not None and cached.status == "completed" and cached.response_json is not None:
        try:
            parsed = response_model.model_validate_json(cached.response_json)
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
            raise RuntimeError(
                f"Model did not return parsed structured output. Refusal: {refusal}"
            )

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


async def translate_candidate(
    *,
    client: AsyncOpenAI,
    cache: CacheStore,
    model: str,
    candidate: TranslationCandidate,
    thinking: bool,
    seed: int,
) -> dict[str, Any]:
    translation, cache_info = await run_cached_structured_completion(
        client=client,
        cache=cache,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(candidate),
        input_payload=candidate.model_dump(mode="json"),
        response_model=TranslationResponse,
        schema_name="translation_card",
        thinking=thinking,
    )
    if translation is None:
        return {"card": None, "cache": cache_info}

    if not isinstance(translation, TranslationResponse):
        raise RuntimeError("Translation response did not match the expected response model.")

    card = build_card_output(candidate, translation, seed)
    return {"card": card.model_dump(mode="json"), "cache": cache_info}


def write_output_json(
    *,
    output_path: Path,
    model: str,
    review_input_path: Path,
    entries_input_path: Path,
    tsv_output_path: Path,
    apkg_output_path: Path,
    deck_name: str,
    seed: int,
    concurrency: int,
    thinking: bool,
    results: list[dict[str, Any]],
    missing_entries: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "items": results,
        "model": model,
        "source_files": {
            "review_results": str(review_input_path),
            "lemma_entries": str(entries_input_path),
        },
        "outputs": {
            "json": str(output_path),
            "tsv": str(tsv_output_path),
            "apkg": str(apkg_output_path),
        },
        "settings": {
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "concurrency": concurrency,
            "thinking": thinking,
            "deck_name": deck_name,
            "seed": seed,
            "prompt_version": "card-translation-v1",
        },
        "summary": {
            "candidate_count": len(results),
            "completed": sum(1 for row in results if row["cache"]["status"] == "completed"),
            "failed": sum(1 for row in results if row["cache"]["status"] == "failed"),
            "cache_hits": sum(1 for row in results if row["cache"]["cache_hit"]),
            "missing_entry_candidates": missing_entries,
        },
    }
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def write_anki_tsv(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for row in results:
            card = row.get("card")
            if row["cache"]["status"] != "completed" or card is None:
                continue
            front = str(card["front"]).replace("\t", " ").replace("\n", " ").strip()
            back_html = str(card["back_html"]).replace("\t", " ").replace("\n", " ").strip()
            handle.write(f"{front}\t{back_html}\n")


def build_genanki_model() -> genanki.Model:
    return genanki.Model(
        ANKI_MODEL_ID,
        "Greek Lemma Card",
        fields=[
            {"name": "Front"},
            {"name": "Back"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": "{{FrontSide}}<hr id=\"answer\">{{Back}}",
            }
        ],
    )


def write_anki_package(results: list[dict[str, Any]], output_path: Path, deck_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deck = genanki.Deck(stable_int("deck", deck_name), deck_name)
    model = build_genanki_model()

    for row in results:
        card = row.get("card")
        if row["cache"]["status"] != "completed" or card is None:
            continue

        note = genanki.Note(
            model=model,
            fields=[card["front"], card["back_html"]],
            guid=genanki.guid_for(card["front"]),
        )
        deck.add_note(note)

    genanki.Package(deck).write_to_file(output_path)


async def run_card_creation(
    *,
    review_input_path: Path,
    entries_input_path: Path,
    output_json_path: Path,
    output_tsv_path: Path,
    output_apkg_path: Path,
    cache_path: Path,
    deck_name: str,
    base_url: str,
    api_key: str,
    model_name: str | None,
    concurrency: int,
    thinking: bool,
    seed: int,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    review_rows = load_review_results(review_input_path)
    entries = load_entries(entries_input_path)
    candidates, missing_entries = build_translation_candidates(
        review_rows=review_rows,
        entries=entries,
    )

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
                seed=seed,
            )

    try:
        results = await asyncio.gather(*(worker(candidate) for candidate in candidates))
    finally:
        await client.close()
        await cache.close()

    write_output_json(
        output_path=output_json_path,
        model=model,
        review_input_path=review_input_path,
        entries_input_path=entries_input_path,
        tsv_output_path=output_tsv_path,
        apkg_output_path=output_apkg_path,
        deck_name=deck_name,
        seed=seed,
        concurrency=concurrency,
        thinking=thinking,
        results=results,
        missing_entries=missing_entries,
    )
    write_anki_tsv(results, output_tsv_path)
    write_anki_package(results, output_apkg_path, deck_name)
    return model, results, missing_entries


@app.command()
def main(
    review_input_path: Path = typer.Option(
        DEFAULT_REVIEW_INPUT_PATH,
        "--review-input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the reviewed lemma JSON file.",
    ),
    entries_input_path: Path = typer.Option(
        DEFAULT_ENTRIES_INPUT_PATH,
        "--entries-input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the lemma entries JSONL file with original forms and sentences.",
    ),
    output_json_path: Path = typer.Option(
        DEFAULT_OUTPUT_JSON_PATH,
        "--output-json",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save the detailed card creation JSON output.",
    ),
    output_tsv_path: Path = typer.Option(
        DEFAULT_OUTPUT_TSV_PATH,
        "--output-tsv",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save a two-column Anki import TSV.",
    ),
    output_apkg_path: Path = typer.Option(
        DEFAULT_OUTPUT_APKG_PATH,
        "--output-apkg",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save an Anki package file.",
    ),
    cache_path: Path = typer.Option(
        DEFAULT_CACHE_PATH,
        "--cache-db",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="SQLite cache path for prompt/response persistence.",
    ),
    deck_name: str = typer.Option(
        DEFAULT_DECK_NAME,
        "--deck-name",
        help="Deck name to use for the generated .apkg file.",
    ),
    concurrency: int = typer.Option(
        10,
        "--concurrency",
        min=1,
        help="Maximum number of concurrent model requests.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Seed used for deterministic random sentence selection.",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL or "",
        "--base-url",
        help="OpenAI-compatible base URL.",
        show_default=False,
    ),
    token: str = typer.Option(
        DEFAULT_TOKEN or "",
        "--token",
        help="API token for the OpenAI-compatible endpoint.",
        show_default=False,
    ),
    model: str | None = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Model name. If omitted, the first model from /v1/models is used.",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable or disable model thinking mode.",
    ),
) -> None:
    if not base_url:
        typer.echo(
            f"Missing base URL. Set BASE_URL in {ROOT_ENV_PATH} or pass --base-url.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not token:
        typer.echo(
            f"Missing API token. Set OPENAI_TOKEN in {ROOT_ENV_PATH} or pass --token.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        resolved_model, results, missing_entries = asyncio.run(
            run_card_creation(
                review_input_path=review_input_path,
                entries_input_path=entries_input_path,
                output_json_path=output_json_path,
                output_tsv_path=output_tsv_path,
                output_apkg_path=output_apkg_path,
                cache_path=cache_path,
                deck_name=deck_name,
                base_url=base_url,
                api_key=token,
                model_name=model,
                concurrency=concurrency,
                thinking=thinking,
                seed=seed,
            )
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"model={resolved_model}")
    typer.echo(f"output_json={output_json_path}")
    typer.echo(f"output_tsv={output_tsv_path}")
    typer.echo(f"output_apkg={output_apkg_path}")
    typer.echo(f"cache_db={cache_path}")
    typer.echo(f"candidate_count={len(results)}")
    typer.echo(f"completed={sum(1 for row in results if row['cache']['status'] == 'completed')}")
    typer.echo(f"failed={sum(1 for row in results if row['cache']['status'] == 'failed')}")
    typer.echo(f"cache_hits={sum(1 for row in results if row['cache']['cache_hit'])}")
    typer.echo(f"missing_entry_candidates={len(missing_entries)}")


if __name__ == "__main__":
    app()
