#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import html
import json
import os
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from subs2anki.db import load_translation_candidates
from subs2anki.db.export import write_jsonl
from subs2anki.llm import (
    PromptCacheStore,
    SentenceContext,
    TranslationCandidate,
    TranslationPromptInput,
    TranslationResponse,
    pick_model,
    run_cached_structured_completion,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)

ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ROOT_ENV_PATH)

DEFAULT_DB_PATH = Path("data/subs2anki.sqlite3")
DEFAULT_OUTPUT_JSONL_PATH = Path("data/card_creator_results.jsonl")
DEFAULT_OUTPUT_TSV_PATH = Path("data/anki_cards.tsv")
DEFAULT_OUTPUT_APKG_PATH = Path("data/anki_cards.apkg")
DEFAULT_BASE_URL = os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL")
DEFAULT_TOKEN = os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY")
DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_DECK_NAME = "Greek Lemmas"
TEMPERATURE = 0.2
MAX_TOKENS = 384
REASONING_EFFORT: str | None = "none"
MAX_RETRIES = 5
ANKI_MODEL_ID = 1387426501

SYSTEM_PROMPT = """You are creating concise learner-facing English translations for Modern Greek study cards.

Task:
Given one reviewed learner-facing normalized form and one selected Greek example sentence, return:
- a short learner-facing English translation for the target word
- a natural English translation of the selected example sentence

Rules:
- Translate `normalized_form`, not `source_lemma`.
- Use `word_class`, `original_forms`, and `example_sentence` to disambiguate the sense.
- `english_translation` should be short and compact, suitable for the back of a flashcard.
- `example_sentence_translation` should be a natural full-sentence English translation of `example_sentence.sentence_text`.
- Prefer lowercase for `english_translation` unless the target is a proper noun.
- Do not explain, justify, transliterate, or add notes.
- Do not include quotation marks.

Input JSON:
{"reviewed_lexeme_id":7,"normalized_form":"ηρεμώ","word_class":"verb","source_lemma":"ηρέμησε","original_forms":["ηρέμησε","Ηρέμησε","ηρεμήσω"],"example_sentence":{"sentence_text":"Θέλω να ηρεμήσω και δεν θέλω να πάρω χάπια.","source_srt":"data/raw/example.srt"},"occurrence_count":7}
Output JSON:
{"english_translation":"calm down; relax","example_sentence_translation":"I want to calm down and I don't want to take pills."}

Input JSON:
{"reviewed_lexeme_id":12,"normalized_form":"σπίτι","word_class":"noun","source_lemma":"σπίτι","original_forms":["σπίτι"],"example_sentence":{"sentence_text":"Χωρίς συγγνώμη, σπίτι δεν γυρίζεις.","source_srt":"data/raw/example.srt"},"occurrence_count":14}
Output JSON:
{"english_translation":"home; house","example_sentence_translation":"Without an apology, you're not going back home."}

Output format:
Return JSON with exactly this shape:
{"english_translation":"...","example_sentence_translation":"..."}
"""


class CardResultRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewed_lexeme_id: int
    normalized_form: str
    word_class: str
    source_lemma: str
    original_forms: list[str]
    occurrence_count: int
    example_sentence: SentenceContext
    english_translation: str | None = None
    example_sentence_translation: str | None = None
    front: str | None = None
    back_html: str | None = None
    cache: dict[str, Any]


def pick_example_sentence(candidate: TranslationCandidate) -> SentenceContext:
    if not candidate.sentences:
        raise RuntimeError(
            f"No sentence context available for reviewed lexeme id {candidate.reviewed_lexeme_id}."
        )
    # Placeholder until sentence-selection becomes its own DB-backed step.
    return candidate.sentences[0]


def build_translation_input(candidate: TranslationCandidate) -> TranslationPromptInput:
    return TranslationPromptInput(
        reviewed_lexeme_id=candidate.reviewed_lexeme_id,
        normalized_form=candidate.normalized_form,
        word_class=candidate.word_class,
        source_lemma=candidate.source_lemma,
        original_forms=list(candidate.original_forms),
        example_sentence=pick_example_sentence(candidate),
        occurrence_count=candidate.occurrence_count,
    )


def build_card_result(
    candidate: TranslationCandidate,
    prompt_input: TranslationPromptInput,
    translation: TranslationResponse | None,
    cache_info: dict[str, Any],
) -> CardResultRow:
    english_translation = None
    example_sentence_translation = None
    front = None
    back_html = None

    if translation is not None:
        english_translation = translation.english_translation.strip()
        example_sentence_translation = translation.example_sentence_translation.strip()
        front = candidate.normalized_form
        back_html = (
            f"{html.escape(english_translation)}"
            f"<br><br>{html.escape(prompt_input.example_sentence.sentence_text)}"
            f"<br>{html.escape(example_sentence_translation)}"
        )

    return CardResultRow(
        reviewed_lexeme_id=candidate.reviewed_lexeme_id,
        normalized_form=candidate.normalized_form,
        word_class=candidate.word_class.value,
        source_lemma=candidate.source_lemma,
        original_forms=list(candidate.original_forms),
        occurrence_count=candidate.occurrence_count,
        example_sentence=prompt_input.example_sentence,
        english_translation=english_translation,
        example_sentence_translation=example_sentence_translation,
        front=front,
        back_html=back_html,
        cache=cache_info,
    )


def write_anki_tsv(results: list[CardResultRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for row in results:
            if row.cache["status"] != "completed" or row.front is None or row.back_html is None:
                continue
            front = row.front.replace("\t", " ").replace("\n", " ").strip()
            back_html = row.back_html.replace("\t", " ").replace("\n", " ").strip()
            handle.write(f"{front}\t{back_html}\n")


def build_genanki_model() -> Any:
    import genanki

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


def write_anki_package(results: list[CardResultRow], output_path: Path, deck_name: str) -> None:
    import genanki

    output_path.parent.mkdir(parents=True, exist_ok=True)
    deck = genanki.Deck(1387426501, deck_name)
    model = build_genanki_model()

    for row in results:
        if row.cache["status"] != "completed" or row.front is None or row.back_html is None:
            continue

        note = genanki.Note(
            model=model,
            fields=[row.front, row.back_html],
            guid=genanki.guid_for(f"{row.reviewed_lexeme_id}:{row.front}"),
        )
        deck.add_note(note)

    genanki.Package(deck).write_to_file(output_path)


async def translate_candidate(
    *,
    client: AsyncOpenAI | None,
    cache: PromptCacheStore,
    model: str,
    candidate: TranslationCandidate,
    thinking: bool,
) -> CardResultRow:
    prompt_input = build_translation_input(candidate)
    translation, cache_info = await run_cached_structured_completion(
        client=client,
        cache=cache,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=json.dumps(prompt_input.model_dump(mode="json"), ensure_ascii=False),
        input_payload=prompt_input.model_dump(mode="json"),
        response_model=TranslationResponse,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        max_retries=MAX_RETRIES,
        thinking=thinking,
    )
    if translation is not None and not isinstance(translation, TranslationResponse):
        raise RuntimeError("Translation response did not match the expected response model.")
    return build_card_result(candidate, prompt_input, translation, cache_info)


async def run_card_creation(
    *,
    db_path: Path,
    output_jsonl_path: Path,
    output_tsv_path: Path | None,
    output_apkg_path: Path | None,
    deck_name: str,
    base_url: str,
    api_key: str,
    model_name: str | None,
    concurrency: int,
    thinking: bool,
    write_tsv: bool,
    write_apkg: bool,
    limit: int | None,
) -> tuple[str, list[CardResultRow]]:
    candidates = load_translation_candidates(db_path)
    if limit is not None:
        candidates = candidates[:limit]

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

    async def worker(candidate: TranslationCandidate) -> CardResultRow:
        async with semaphore:
            return await translate_candidate(
                client=client,
                cache=cache,
                model=model,
                candidate=candidate,
                thinking=thinking,
            )

    try:
        results = await asyncio.gather(*(worker(candidate) for candidate in candidates))
    finally:
        if client is not None:
            await client.close()
        await cache.close()

    write_jsonl([row.model_dump(mode="json") for row in results], output_jsonl_path)
    if write_tsv:
        if output_tsv_path is None:
            raise RuntimeError("TSV output requested but no --output-tsv path was provided.")
        write_anki_tsv(results, output_tsv_path)
    if write_apkg:
        if output_apkg_path is None:
            raise RuntimeError("APKG output requested but no --output-apkg path was provided.")
        write_anki_package(results, output_apkg_path, deck_name)
    return model, results


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
    output_jsonl_path: Path = typer.Option(
        DEFAULT_OUTPUT_JSONL_PATH,
        "--output-jsonl",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save the detailed card creation JSONL output.",
    ),
    output_tsv_path: Path | None = typer.Option(
        DEFAULT_OUTPUT_TSV_PATH,
        "--output-tsv",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save a two-column Anki import TSV.",
    ),
    output_apkg_path: Path | None = typer.Option(
        DEFAULT_OUTPUT_APKG_PATH,
        "--output-apkg",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save an Anki package file.",
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
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Optional limit on the number of reviewed lexemes to translate.",
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
    write_tsv: bool = typer.Option(
        False,
        "--write-tsv/--no-write-tsv",
        help="Write the TSV artifact.",
    ),
    write_apkg: bool = typer.Option(
        False,
        "--write-apkg/--no-write-apkg",
        help="Write the APKG artifact.",
    ),
) -> None:
    if (base_url and not token) or (token and not base_url):
        typer.echo(
            "Provide both --base-url and --token, or neither to run against the existing cache only.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        resolved_model, results = asyncio.run(
            run_card_creation(
                db_path=db_path,
                output_jsonl_path=output_jsonl_path,
                output_tsv_path=output_tsv_path,
                output_apkg_path=output_apkg_path,
                deck_name=deck_name,
                base_url=base_url,
                api_key=token,
                model_name=model,
                concurrency=concurrency,
                thinking=thinking,
                write_tsv=write_tsv,
                write_apkg=write_apkg,
                limit=limit,
            )
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"model={resolved_model}")
    typer.echo(f"db={db_path}")
    typer.echo(f"output_jsonl={output_jsonl_path}")
    typer.echo(f"output_tsv={output_tsv_path if write_tsv else 'disabled'}")
    typer.echo(f"output_apkg={output_apkg_path if write_apkg else 'disabled'}")
    typer.echo("cache_db=same as --db")
    typer.echo(f"candidate_count={len(results)}")
    typer.echo(f"completed={sum(1 for row in results if row.cache['status'] == 'completed')}")
    typer.echo(f"failed={sum(1 for row in results if row.cache['status'] == 'failed')}")
    typer.echo(f"cache_hits={sum(1 for row in results if row.cache['cache_hit'])}")


if __name__ == "__main__":
    app()
