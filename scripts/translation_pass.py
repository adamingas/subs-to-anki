#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from subs2anki.db import (
    load_ordered_translation_candidates,
    rebuild_normalized_lexeme_ordering,
    resolve_scope_id,
)
from subs2anki.db.export import write_jsonl
from subs2anki.llm import (
    PromptCacheStore,
    SentenceContext,
    TranslationCandidate,
    TranslationResponse,
    pick_model,
    run_cached_structured_completion,
)
from subs2anki.translations import (
    TRANSLATION_SYSTEM_PROMPT,
    build_translation_prompt_input,
    build_translation_user_prompt,
    sanitize_translation_response,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)

ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ROOT_ENV_PATH)

DEFAULT_DB_PATH = Path("data/subs2anki.sqlite3")
DEFAULT_OUTPUT_PATH = Path("data/translation_pass.head25.jsonl")
DEFAULT_BASE_URL = os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL")
DEFAULT_TOKEN = os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY")
DEFAULT_MODEL = "qwen/qwen3.5-122b-a10b"
DEFAULT_REASONING_EFFORT: str | None = "none"
DEFAULT_ORDERING_ALGORITHM = "greedy_unlock"
TEMPERATURE = 0.1
MAX_TOKENS = 256
MAX_RETRIES = 5


class TranslationPassRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_lexeme_id: int
    normalized_form: str
    word_class: str
    source_lemmas: list[str]
    original_forms: list[str]
    occurrence_count: int
    example_sentence: SentenceContext
    english_translation: str | None = None
    example_sentence_translation: str | None = None
    cache: dict[str, Any]


def ensure_unique_normalized_forms(rows: list[TranslationPassRow]) -> None:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for row in rows:
        normalized_form = row.normalized_form.strip().casefold()
        if normalized_form in seen:
            duplicates.append(row.normalized_form)
            continue
        seen[normalized_form] = row.normalized_lexeme_id
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise RuntimeError(f"Duplicate normalized forms in translation output: {preview}")


def build_result_row(
    candidate: TranslationCandidate,
    prompt_input: dict[str, Any],
    translation: TranslationResponse | None,
    cache_info: dict[str, Any],
) -> TranslationPassRow:
    return TranslationPassRow(
        normalized_lexeme_id=candidate.normalized_lexeme_id,
        normalized_form=candidate.normalized_form,
        word_class=candidate.word_class.value,
        source_lemmas=list(candidate.source_lemmas),
        original_forms=list(candidate.original_forms),
        occurrence_count=candidate.occurrence_count,
        example_sentence=SentenceContext.model_validate(prompt_input["example_sentence"]),
        english_translation=translation.english_translation if translation is not None else None,
        example_sentence_translation=(
            translation.example_sentence_translation if translation is not None else None
        ),
        cache=cache_info,
    )


async def translate_candidate(
    *,
    client: AsyncOpenAI | None,
    cache: PromptCacheStore,
    model: str,
    candidate: TranslationCandidate,
    thinking: bool,
    reasoning_effort: str | None,
) -> TranslationPassRow:
    prompt_input_model = build_translation_prompt_input(candidate)
    prompt_input = prompt_input_model.model_dump(mode="json")
    translation, cache_info = await run_cached_structured_completion(
        client=client,
        cache=cache,
        model=model,
        system_prompt=TRANSLATION_SYSTEM_PROMPT,
        user_prompt=build_translation_user_prompt(prompt_input_model),
        input_payload=prompt_input,
        response_model=TranslationResponse,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        reasoning_effort=reasoning_effort,
        max_retries=MAX_RETRIES,
        thinking=thinking,
        sanitizer=sanitize_translation_response,
    )
    if translation is not None and not isinstance(translation, TranslationResponse):
        raise RuntimeError("Translation response did not match the expected response model.")
    return build_result_row(candidate, prompt_input, translation, cache_info)


async def run_translation_pass(
    *,
    db_path: Path,
    scope_id: int,
    output_path: Path,
    base_url: str,
    api_key: str,
    model_name: str | None,
    concurrency: int,
    thinking: bool,
    reasoning_effort: str | None,
    ordering_algorithm: str,
    limit: int | None,
) -> tuple[str, dict[str, Any], list[TranslationPassRow]]:
    ordering_metrics = rebuild_normalized_lexeme_ordering(
        db_path,
        scope_id=scope_id,
        algorithm=ordering_algorithm,
    )
    candidates = load_ordered_translation_candidates(db_path, scope_id=scope_id)
    if limit is not None:
        candidates = candidates[:limit]
    total_candidates = len(candidates)

    cache = PromptCacheStore(db_path)
    await cache.open()
    client: AsyncOpenAI | None = None
    if base_url and api_key:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    if client is None:
        model = model_name or DEFAULT_MODEL
    else:
        model = await pick_model(client, model_name)

    semaphore = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()
    started_at = time.perf_counter()
    completed_count = 0
    failed_count = 0
    cache_hit_count = 0

    async def worker(candidate: TranslationCandidate) -> TranslationPassRow:
        nonlocal completed_count, failed_count, cache_hit_count
        async with semaphore:
            result = await translate_candidate(
                client=client,
                cache=cache,
                model=model,
                candidate=candidate,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
            )
        async with progress_lock:
            completed_count += 1
            if result.cache["status"] == "failed":
                failed_count += 1
            if result.cache["cache_hit"]:
                cache_hit_count += 1
            if (
                completed_count <= 5
                or completed_count % 25 == 0
                or completed_count == total_candidates
            ):
                elapsed_seconds = time.perf_counter() - started_at
                typer.echo(
                    "progress="
                    f"{completed_count}/{total_candidates} "
                    f"failed={failed_count} "
                    f"cache_hits={cache_hit_count} "
                    f"elapsed_s={elapsed_seconds:.1f}"
                )
        return result

    try:
        results = await asyncio.gather(*(worker(candidate) for candidate in candidates))
    finally:
        if client is not None:
            await client.close()
        await cache.close()

    ensure_unique_normalized_forms(results)
    write_jsonl([row.model_dump(mode="json") for row in results], output_path)
    return model, ordering_metrics, results


@app.command()
def main(
    db_path: Path = typer.Option(
        DEFAULT_DB_PATH,
        "--db",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the subs2anki SQLite database.",
    ),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--output",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save the translation JSONL output.",
    ),
    concurrency: int = typer.Option(
        10,
        "--concurrency",
        min=1,
        help="Maximum number of concurrent model requests.",
    ),
    limit: int | None = typer.Option(
        25,
        "--limit",
        min=1,
        help="Maximum number of reviewed lexemes to translate.",
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
    reasoning_effort: str = typer.Option(
        DEFAULT_REASONING_EFFORT or "none",
        "--reasoning-effort",
        help="Reasoning effort passed through the provider extra_body.",
    ),
    ordering_algorithm: str = typer.Option(
        DEFAULT_ORDERING_ALGORITHM,
        "--ordering-algorithm",
        help="Sentence-ordering algorithm to use before translation.",
    ),
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable or disable model thinking mode.",
    ),
    scope_name: str | None = typer.Option(
        None,
        "--scope",
        help="Scope name to translate. If omitted, the only scope in the DB is used.",
    ),
) -> None:
    if (base_url and not token) or (token and not base_url):
        typer.echo(
            "Provide both --base-url and --token, or neither to run against the existing cache only.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        scope_id = resolve_scope_id(db_path, scope_name=scope_name)
        resolved_model, ordering_metrics, results = asyncio.run(
            run_translation_pass(
                db_path=db_path,
                scope_id=scope_id,
                output_path=output_path,
                base_url=base_url,
                api_key=token,
                model_name=model,
                concurrency=concurrency,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                ordering_algorithm=ordering_algorithm,
                limit=limit,
            )
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"model={resolved_model}")
    typer.echo(f"db={db_path}")
    typer.echo(f"scope_id={scope_id}")
    if scope_name is not None:
        typer.echo(f"scope_name={scope_name}")
    typer.echo(f"output={output_path}")
    typer.echo("cache_db=same as --db")
    typer.echo(f"ordering_algorithm={ordering_metrics['algorithm']}")
    typer.echo(f"ordering_cycle_breaks={ordering_metrics['cycle_breaks']}")
    typer.echo(f"ordering_no_sentence={ordering_metrics['no_sentence']}")
    typer.echo(f"ordering_first_no_sentence={ordering_metrics['first_no_sentence']}")
    typer.echo(f"candidate_count={len(results)}")
    typer.echo(f"completed={sum(1 for row in results if row.cache['status'] == 'completed')}")
    typer.echo(f"failed={sum(1 for row in results if row.cache['status'] == 'failed')}")
    typer.echo(f"cache_hits={sum(1 for row in results if row.cache['cache_hit'])}")


if __name__ == "__main__":
    app()
