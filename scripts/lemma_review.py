#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from openai import AsyncOpenAI

from subs2anki.db import (
    export_reviewed_lexemes_jsonl,
    load_review_candidates,
    replace_reviewed_lexemes,
    resolve_scope_id,
)
from subs2anki.llm import (
    LemmaCandidate,
    LexemeClass,
    LexemeGroup,
    LexemeResolutionResponse,
    PromptCacheStore,
    PronounGroup,
    PronounKind,
    PronounResolutionResponse,
    ReviewedLexemeRow,
    SentenceContext,
    pick_model,
    run_cached_structured_completion,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)

ROOT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ROOT_ENV_PATH)

DEFAULT_DB_PATH = Path("data/subs2anki.sqlite3")
DEFAULT_OUTPUT_PATH = Path("data/lemma_review_results.jsonl")
DEFAULT_BASE_URL = os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL")
DEFAULT_TOKEN = os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY")
DEFAULT_MODEL = "google/gemma-4-31b-it"
TEMPERATURE = 0.2
MAX_TOKENS = 768
REASONING_EFFORT: str | None = "none"
MAX_RETRIES = 5

SYSTEM_PROMPT = """You are a Modern Greek learner-facing lexeme resolver.

Task:
Given one JSON object from an earlier subtitle lemmatization pipeline, resolve it into one or
more learner-facing lexeme groups.

Each output lexeme group must contain:
- `normalized_form`
- `word_class`
- `original_forms`

Rules:
- Focus primarily on `original_forms`. Those are the observed source words from the subtitles.
- Treat `lemmatised_form` as a noisy upstream hint only.
- Use `sentences` only to disambiguate which source forms belong together.
- Return one or more lexeme groups in `lexemes`.
- Every source form from `original_forms` must appear in exactly one output group.
- Split the row into as many groups as necessary. If unsure, split more finely rather than over-merge.
- This is learner-family normalization, not strict lexicographic lemmatization.
- Upper/lowercase variants of the same ordinary word belong to the same lexeme group.
- Only use `proper_noun` when the form is genuinely a name; do not use `other` just because a pronoun,
  article, particle, conjunction, or preposition appears capitalized at the start of a sentence.
- For non-`proper_noun` groups, prefer lowercase normalized forms.

Normalization policy:
- `noun`: prefer noun-family normalization when a noun and a related adjective belong to the same learner family.
  Example: `ανθρώπινος -> άνθρωπος`.
- `verb`: use the citation form of the verb.
  Example: `ηρέμησε -> ηρεμώ`.
- `adjective`: convert adverbs to their adjective family when no noun-family target is better.
  Example: `γρήγορα -> γρήγορος`.
- `adverb`: keep truly adverbial words as adverbs when they should not be normalized further.
  Examples: `εδώ -> εδώ`, `πώς -> πώς`.
- Function lexemes stay as function lexemes:
  - `preposition`: `σε`, `από`
  - `conjunction`: `και`, `αλλά`
  - `particle`: `θα`, `να`
  - `pronoun`: `εγώ`, `εμένα`, `εσύ`, `εσένα`
  - `article`: `ο`, `ένας`
- Use `proper_noun` for names.
- Use `other` for malformed subtitle noise or forms that do not safely fit the taxonomy.

Allowed `word_class` values:
- noun
- verb
- adjective
- adverb
- preposition
- conjunction
- particle
- pronoun
- article
- proper_noun
- other

Allowed `word_class` values with examples:
- `noun`: `σπίτι`, `άνθρωπος`, `γυναίκα`
- `verb`: `είμαι`, `κάνω`, `λέω`
- `adjective`: `καλός`, `μεγάλος`, `μπλε`
- `adverb`: `εδώ`, `πώς`, `μαζί`
- `preposition`: `σε`, `από`, `χωρίς`
- `conjunction`: `και`, `αλλά`, `επειδή`
- `particle`: `θα`, `να`, `δεν`
- `pronoun`: `εγώ`, `εμένα`, `αυτός`
- `article`: `ο`, `η`, `ένας`
- `proper_noun`: `Μαρία`, `Αθήνα`, `Γιάννης`
- `other`: `}δεν`, `λες...`, `-`

Examples:

Input JSON:
{"lemmatised_form":"άνθρωπος","count":3,"original_forms":["άνθρωποι","άνθρωπος"],"sentences":[{"sentence_text":"Οι άνθρωποι αλλάζουν.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"άνθρωπος","word_class":"noun","original_forms":["άνθρωποι","άνθρωπος"]}]}

Input JSON:
{"lemmatised_form":"ηρέμησε","count":3,"original_forms":["ηρέμησε","ηρεμήσεις","ηρεμήσω"],"sentences":[{"sentence_text":"Ηρέμησε.","source_srt":"data/raw/example.srt"},{"sentence_text":"Μπορείς να ηρεμήσεις;","source_srt":"data/raw/example.srt"},{"sentence_text":"Θα ηρεμήσω.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"ηρεμώ","word_class":"verb","original_forms":["ηρέμησε","ηρεμήσεις","ηρεμήσω"]}]}

Input JSON:
{"lemmatised_form":"γρήγορα","count":2,"original_forms":["γρήγορα"],"sentences":[{"sentence_text":"Τρέχει γρήγορα.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"γρήγορος","word_class":"adjective","original_forms":["γρήγορα"]}]}

Input JSON:
{"lemmatised_form":"εδώ","count":2,"original_forms":["εδώ","Εδώ"],"sentences":[{"sentence_text":"Εδώ είμαι.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"εδώ","word_class":"adverb","original_forms":["εδώ","Εδώ"]}]}

Input JSON:
{"lemmatised_form":"πώς","count":2,"original_forms":["πώς","Πώς"],"sentences":[{"sentence_text":"Πώς είσαι;","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"πώς","word_class":"adverb","original_forms":["πώς","Πώς"]}]}

Input JSON:
{"lemmatised_form":"τα","count":4,"original_forms":["Εγώ","εγώ","Εμένα","εμένα"],"sentences":[{"sentence_text":"Εγώ είμαι εδώ.","source_srt":"data/raw/example.srt"},{"sentence_text":"Εμένα ποιος θα με ακούσει;","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"εγώ","word_class":"pronoun","original_forms":["Εγώ","εγώ"]},{"normalized_form":"εμένα","word_class":"pronoun","original_forms":["Εμένα","εμένα"]}]}

Input JSON:
{"lemmatised_form":"τα","count":4,"original_forms":["Εσύ","εσύ","εσένα"],"sentences":[{"sentence_text":"Εσύ τι λες;","source_srt":"data/raw/example.srt"},{"sentence_text":"Εσένα περίμενα.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"εσύ","word_class":"pronoun","original_forms":["Εσύ","εσύ"]},{"normalized_form":"εσένα","word_class":"pronoun","original_forms":["εσένα"]}]}

Input JSON:
{"lemmatised_form":"σε","count":2,"original_forms":["σε"],"sentences":[{"sentence_text":"Ήρθα σε σένα.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"σε","word_class":"preposition","original_forms":["σε"]}]}

Input JSON:
{"lemmatised_form":"και","count":2,"original_forms":["και"],"sentences":[{"sentence_text":"Και τώρα;","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"και","word_class":"conjunction","original_forms":["και"]}]}

Input JSON:
{"lemmatised_form":"θα","count":2,"original_forms":["θα"],"sentences":[{"sentence_text":"Θα έρθω.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"θα","word_class":"particle","original_forms":["θα"]}]}

Input JSON:
{"lemmatised_form":"ο","count":2,"original_forms":["ο","ένας"],"sentences":[{"sentence_text":"Ο άνθρωπος ήρθε.","source_srt":"data/raw/example.srt"},{"sentence_text":"Ένας φίλος με πήρε.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"ο","word_class":"article","original_forms":["ο"]},{"normalized_form":"ένας","word_class":"article","original_forms":["ένας"]}]}

Input JSON:
{"lemmatised_form":"ο","count":3,"original_forms":["Του","του","των"],"sentences":[{"sentence_text":"Του μίλησα χθες.","source_srt":"data/raw/example.srt"},{"sentence_text":"Το σπίτι του είναι μακριά.","source_srt":"data/raw/example.srt"},{"sentence_text":"Η γνώμη των άλλων.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"ο","word_class":"article","original_forms":["Του","του","των"]}]}

Input JSON:
{"lemmatised_form":"μαρία","count":2,"original_forms":["Μαρίας","Μαρία"],"sentences":[{"sentence_text":"Το βιβλίο της Μαρίας.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"Μαρία","word_class":"proper_noun","original_forms":["Μαρίας","Μαρία"]}]}

Input JSON:
{"lemmatised_form":"}δεν","count":1,"original_forms":["}δεν"],"sentences":[{"sentence_text":"}δεν ξέρω","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"δεν","word_class":"particle","original_forms":["}δεν"]}]}

Output format:
Return JSON with exactly this shape:
{"lexemes":[{"normalized_form":"...","word_class":"noun|verb|adjective|adverb|preposition|conjunction|particle|pronoun|article|proper_noun|other","original_forms":["..."]}]}
"""

PRONOUN_PROMPT = """You are a Modern Greek learner-facing pronoun resolver.

Task:
Given one pronoun lexeme group plus the subtitle sentences that contain its source forms, split it
into learner-facing pronoun forms.

Rules:
- Focus on learner-facing pronoun forms, not dictionary lemma paradigms.
- Keep weak/clitic pronouns as their own normalized forms.
- Do not normalize `μου`, `με` to `εγώ`.
- Do not normalize `σου`, `σε` to `εσύ`.
- Keep uppercase and lowercase variants of the same ordinary pronoun together.
- Use one output group per learner-facing pronoun form.
- Every input source form must appear in exactly one output group.
- If unsure, split more finely rather than merging forms with different learner-facing use.
- Return only pronoun groups.
- For non-name pronouns, prefer lowercase normalized forms.
- Do not explain.

Examples:

Input JSON:
{"normalized_form":"εγώ","original_forms":["Εγώ","εγώ","Εμένα","εμένα","Μου","μου","Με","με"],"sentences":[{"sentence_text":"Εγώ είμαι εδώ.","source_srt":"data/raw/example.srt"},{"sentence_text":"Εμένα περίμεναν.","source_srt":"data/raw/example.srt"},{"sentence_text":"Μου είπε να φύγω.","source_srt":"data/raw/example.srt"},{"sentence_text":"Με είδε στον δρόμο.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"pronouns":[{"normalized_form":"εγώ","pronoun_kind":"strong","original_forms":["Εγώ","εγώ"]},{"normalized_form":"εμένα","pronoun_kind":"oblique","original_forms":["Εμένα","εμένα"]},{"normalized_form":"μου","pronoun_kind":"clitic","original_forms":["Μου","μου"]},{"normalized_form":"με","pronoun_kind":"clitic","original_forms":["Με","με"]}]}

Input JSON:
{"normalized_form":"εσύ","original_forms":["Εσύ","εσύ","εσένα","Σου","σου","Σε","σε"],"sentences":[{"sentence_text":"Εσύ τι λες;","source_srt":"data/raw/example.srt"},{"sentence_text":"Εσένα φώναζα.","source_srt":"data/raw/example.srt"},{"sentence_text":"Σου το είπα ήδη.","source_srt":"data/raw/example.srt"},{"sentence_text":"Σε είδα χθες.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"pronouns":[{"normalized_form":"εσύ","pronoun_kind":"strong","original_forms":["Εσύ","εσύ"]},{"normalized_form":"εσένα","pronoun_kind":"oblique","original_forms":["εσένα"]},{"normalized_form":"σου","pronoun_kind":"clitic","original_forms":["Σου","σου"]},{"normalized_form":"σε","pronoun_kind":"clitic","original_forms":["Σε","σε"]}]}

Input JSON:
{"normalized_form":"τι","original_forms":["Τι","τι"],"sentences":[{"sentence_text":"Τι κάνεις;","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"pronouns":[{"normalized_form":"τι","pronoun_kind":"strong","original_forms":["Τι","τι"]}]}

Output format:
Return JSON with exactly this shape:
{"pronouns":[{"normalized_form":"...","pronoun_kind":"strong|oblique|clitic","original_forms":["..."]}]}
"""


def build_user_prompt(candidate: LemmaCandidate) -> str:
    return json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False)


def build_singleton_fallback_response(candidate: LemmaCandidate) -> LexemeResolutionResponse:
    return LexemeResolutionResponse(
        lexemes=[
            LexemeGroup(
                normalized_form=form,
                word_class=LexemeClass.other,
                original_forms=[form],
            )
            for form in candidate.original_forms
        ]
    )


def form_pattern(form: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(form)}(?!\w)", re.IGNORECASE)


def sentence_matches_form(sentence_text: str, form: str) -> bool:
    return bool(form_pattern(form).search(sentence_text))


def unique_sentence_contexts(sentences: list[SentenceContext]) -> list[SentenceContext]:
    seen: set[tuple[str, str]] = set()
    unique: list[SentenceContext] = []
    for sentence in sentences:
        key = (sentence.source_srt, sentence.sentence_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sentence)
    return unique


def build_pronoun_input(candidate: LemmaCandidate, lexeme: LexemeGroup) -> dict[str, Any]:
    matched_sentences = unique_sentence_contexts(
        [
            sentence
            for sentence in candidate.sentences
            if any(sentence_matches_form(sentence.sentence_text, form) for form in lexeme.original_forms)
        ]
    )
    return {
        "normalized_form": lexeme.normalized_form,
        "original_forms": list(lexeme.original_forms),
        "sentences": [sentence.model_dump(mode="json") for sentence in matched_sentences],
    }


def sanitize_lexeme_resolution(
    candidate: LemmaCandidate,
    response: LexemeResolutionResponse,
) -> LexemeResolutionResponse:
    parent_order = {form: index for index, form in enumerate(candidate.original_forms)}
    parent_forms = set(candidate.original_forms)
    seen_forms: set[str] = set()
    sanitized: list[LexemeGroup] = []

    for lexeme in response.lexemes:
        normalized_form = lexeme.normalized_form.strip()
        if not normalized_form:
            continue

        ordered_unique_forms: list[str] = []
        child_seen: set[str] = set()
        for form in lexeme.original_forms:
            if form in child_seen or form not in parent_forms or form in seen_forms:
                continue
            child_seen.add(form)
            ordered_unique_forms.append(form)

        if not ordered_unique_forms:
            continue

        ordered_unique_forms.sort(key=parent_order.__getitem__)
        seen_forms.update(ordered_unique_forms)
        sanitized.append(
            LexemeGroup(
                normalized_form=normalized_form,
                word_class=lexeme.word_class,
                original_forms=ordered_unique_forms,
            )
        )

    for form in candidate.original_forms:
        if form not in seen_forms:
            sanitized.append(
                LexemeGroup(
                    normalized_form=form,
                    word_class=LexemeClass.other,
                    original_forms=[form],
                )
            )

    return LexemeResolutionResponse(lexemes=sanitized)


def sanitize_pronoun_resolution(
    lexeme: LexemeGroup,
    response: PronounResolutionResponse,
) -> PronounResolutionResponse:
    parent_order = {form: index for index, form in enumerate(lexeme.original_forms)}
    parent_forms = set(lexeme.original_forms)
    seen_forms: set[str] = set()
    sanitized: list[PronounGroup] = []

    for pronoun in response.pronouns:
        normalized_form = pronoun.normalized_form.strip()
        if not normalized_form:
            continue

        ordered_unique_forms: list[str] = []
        child_seen: set[str] = set()
        for form in pronoun.original_forms:
            if form in child_seen or form not in parent_forms or form in seen_forms:
                continue
            child_seen.add(form)
            ordered_unique_forms.append(form)

        if not ordered_unique_forms:
            continue

        ordered_unique_forms.sort(key=parent_order.__getitem__)
        seen_forms.update(ordered_unique_forms)
        sanitized.append(
            PronounGroup(
                normalized_form=normalized_form,
                pronoun_kind=pronoun.pronoun_kind,
                original_forms=ordered_unique_forms,
            )
        )

    for form in lexeme.original_forms:
        if form not in seen_forms:
            sanitized.append(
                PronounGroup(
                    normalized_form=form.lower(),
                    pronoun_kind=PronounKind.clitic,
                    original_forms=[form],
                )
            )

    return PronounResolutionResponse(pronouns=sanitized)


def build_pronoun_fallback_response(lexeme: LexemeGroup) -> PronounResolutionResponse:
    return PronounResolutionResponse(
        pronouns=[
            PronounGroup(
                normalized_form=form.lower(),
                pronoun_kind=PronounKind.clitic,
                original_forms=[form],
            )
            for form in lexeme.original_forms
        ]
    )


async def review_candidate(
    *,
    client: AsyncOpenAI | None,
    cache: PromptCacheStore,
    model: str,
    candidate: LemmaCandidate,
    thinking: bool,
) -> tuple[str, list[ReviewedLexemeRow], dict[str, Any]]:
    fallback_response = build_singleton_fallback_response(candidate)
    review, cache_entry = await run_cached_structured_completion(
        client=client,
        cache=cache,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(candidate),
        input_payload=candidate.model_dump(mode="json"),
        response_model=LexemeResolutionResponse,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        max_retries=MAX_RETRIES,
        thinking=thinking,
        sanitizer=lambda response: sanitize_lexeme_resolution(candidate, response),
        fallback_response=fallback_response,
    )
    if review is None:
        return candidate.lemmatised_form, [], cache_entry

    rows: list[ReviewedLexemeRow] = []
    cache_entries: list[dict[str, Any]] = [cache_entry]

    for lexeme in review.lexemes:
        if lexeme.word_class != LexemeClass.pronoun:
            rows.append(
                ReviewedLexemeRow(
                    original_forms=lexeme.original_forms,
                    normalized_form=lexeme.normalized_form,
                    word_class=lexeme.word_class,
                )
            )
            continue

        pronoun_input = build_pronoun_input(candidate, lexeme)
        pronoun_review, pronoun_cache = await run_cached_structured_completion(
            client=client,
            cache=cache,
            model=model,
            system_prompt=PRONOUN_PROMPT,
            user_prompt=json.dumps(pronoun_input, ensure_ascii=False),
            input_payload=pronoun_input,
            response_model=PronounResolutionResponse,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            reasoning_effort=REASONING_EFFORT,
            max_retries=MAX_RETRIES,
            thinking=thinking,
            sanitizer=lambda response, lexeme=lexeme: sanitize_pronoun_resolution(lexeme, response),
            fallback_response=build_pronoun_fallback_response(lexeme),
        )
        cache_entries.append(pronoun_cache)
        if pronoun_review is None:
            continue

        rows.extend(
            ReviewedLexemeRow(
                original_forms=pronoun.original_forms,
                normalized_form=pronoun.normalized_form,
                word_class=LexemeClass.pronoun,
            )
            for pronoun in pronoun_review.pronouns
        )

    return (
        candidate.lemmatised_form,
        rows,
        {
            "cache_hit": all(entry["cache_hit"] for entry in cache_entries),
            "elapsed_ms": sum(entry["elapsed_ms"] or 0.0 for entry in cache_entries),
            "status": "completed"
            if all(entry["status"] == "completed" for entry in cache_entries)
            else "failed",
            "error_text": next(
                (entry["error_text"] for entry in reversed(cache_entries) if entry["error_text"]),
                None,
            ),
        },
    )


async def run_reviews(
    *,
    db_path: Path,
    scope_id: int,
    output_path: Path,
    base_url: str,
    api_key: str,
    model_name: str | None,
    concurrency: int,
    thinking: bool,
) -> tuple[str, dict[str, list[ReviewedLexemeRow]], list[dict[str, Any]]]:
    candidates = load_review_candidates(db_path, scope_id=scope_id)
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

    async def worker(candidate: LemmaCandidate) -> tuple[str, list[ReviewedLexemeRow], dict[str, Any]]:
        async with semaphore:
            return await review_candidate(
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

    review_rows_by_lemma = {lemma_text: rows for lemma_text, rows, _ in results}
    replace_reviewed_lexemes(db_path, scope_id=scope_id, review_rows_by_lemma=review_rows_by_lemma)
    export_reviewed_lexemes_jsonl(db_path, output_path, scope_id=scope_id)
    return model, review_rows_by_lemma, [cache_entry for _, _, cache_entry in results]


@app.command()
def main(
    db_path: Path = typer.Option(
        DEFAULT_DB_PATH,
        "--db",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="SQLite database produced by the subtitle importer.",
    ),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--output",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save the review JSONL compatibility export.",
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
    thinking: bool = typer.Option(
        False,
        "--thinking/--no-thinking",
        help="Enable or disable model thinking mode.",
    ),
    scope_name: str | None = typer.Option(
        None,
        "--scope",
        help="Scope name to review. If omitted, the only scope in the DB is used.",
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
        resolved_model, review_rows_by_lemma, cache_entries = asyncio.run(
            run_reviews(
                db_path=db_path,
                scope_id=scope_id,
                output_path=output_path,
                base_url=base_url,
                api_key=token,
                model_name=model,
                concurrency=concurrency,
                thinking=thinking,
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
    typer.echo(f"reviewed_lemmas={len(review_rows_by_lemma)}")
    typer.echo(f"reviewed_rows={sum(len(rows) for rows in review_rows_by_lemma.values())}")
    typer.echo(f"cache_hits={sum(1 for entry in cache_entries if entry['cache_hit'])}")


if __name__ == "__main__":
    app()
