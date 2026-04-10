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
import re
import time
from enum import Enum
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

DEFAULT_INPUT_PATH = Path("data/lemma_entries.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/lemma_review_results.jsonl")
DEFAULT_CACHE_PATH = Path("data/lemma_review_cache.sqlite3")
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
{"lemmatised_form":"τα","count":7,"original_forms":["Εγώ","εγώ","Εμένα","εμένα","Εσύ","εσύ","εσένα"],"sentences":[{"sentence_text":"Εγώ μιλάω τώρα.","source_srt":"data/raw/example.srt"},{"sentence_text":"Εμένα ποιος θα με ακούσει;","source_srt":"data/raw/example.srt"},{"sentence_text":"Εσύ τι λες;","source_srt":"data/raw/example.srt"},{"sentence_text":"Εσένα περίμενα.","source_srt":"data/raw/example.srt"}]}
Output JSON:
{"lexemes":[{"normalized_form":"εγώ","word_class":"pronoun","original_forms":["Εγώ","εγώ"]},{"normalized_form":"εμένα","word_class":"pronoun","original_forms":["Εμένα","εμένα"]},{"normalized_form":"εσύ","word_class":"pronoun","original_forms":["Εσύ","εσύ"]},{"normalized_form":"εσένα","word_class":"pronoun","original_forms":["εσένα"]}]}

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
{"lexemes":[{"normalized_form":"}δεν","word_class":"other","original_forms":["}δεν"]}]}

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


class SentenceContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    sentence_text: str
    source_srt: str


class LemmaCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    lemmatised_form: str
    count: int
    original_forms: list[str]
    sentences: list[SentenceContext]
    english_lemmas: list[str] | None = None


class LexemeClass(str, Enum):
    noun = "noun"
    verb = "verb"
    adjective = "adjective"
    adverb = "adverb"
    preposition = "preposition"
    conjunction = "conjunction"
    particle = "particle"
    pronoun = "pronoun"
    article = "article"
    proper_noun = "proper_noun"
    other = "other"


class LexemeGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_form: str = Field(
        ...,
        description=(
            "The learner-facing normalized form for this lexeme group. "
            "Use noun-preferred family normalization, verb citation forms, "
            "adjective-family normalization for adverbs, and concrete function-lexeme forms."
        ),
    )
    word_class: LexemeClass
    original_forms: list[str] = Field(
        ...,
        min_length=1,
        description="Only the source forms that belong to this lexeme group.",
    )


class LexemeResolutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lexemes: list[LexemeGroup] = Field(
        ...,
        min_length=1,
        description="The resolved learner-facing lexeme groups for the input row.",
    )


class PronounKind(str, Enum):
    strong = "strong"
    oblique = "oblique"
    clitic = "clitic"


class PronounGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_form: str
    pronoun_kind: PronounKind
    original_forms: list[str] = Field(..., min_length=1)


class PronounResolutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pronouns: list[PronounGroup] = Field(..., min_length=1)


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


def build_user_prompt(candidate: LemmaCandidate) -> str:
    return json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False)


def response_schema_hash(response_model: type[BaseModel]) -> str:
    return hash_bytes(canonical_json_bytes(response_model.model_json_schema()))


def build_parameters_payload(*, thinking: bool) -> dict[str, Any]:
    return {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": thinking},
            "reasoning": {"effort": REASONING_EFFORT},
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


def load_candidates(input_path: Path) -> list[LemmaCandidate]:
    candidates: list[LemmaCandidate] = []
    for line_number, raw_line in enumerate(input_path.read_bytes().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = orjson.loads(raw_line)
            candidates.append(LemmaCandidate.model_validate(payload))
        except Exception as exc:
            raise RuntimeError(
                f"Invalid JSONL candidate at {input_path}:{line_number}: {exc}"
            ) from exc
    return candidates


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


def build_pronoun_input(
    candidate: LemmaCandidate,
    lexeme: LexemeGroup,
) -> dict[str, Any]:
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
        if not lexeme.normalized_form.strip():
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
                normalized_form=lexeme.normalized_form.strip(),
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
                schema_hash TEXT NOT NULL DEFAULT '',
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
    thinking: bool,
    sanitizer: Any = None,
    fallback_response: BaseModel | None = None,
) -> tuple[BaseModel | None, dict[str, Any]]:
    schema_hash = response_schema_hash(response_model)
    parameters_payload = build_parameters_payload(thinking=thinking)
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


async def review_candidate(
    *,
    client: AsyncOpenAI,
    cache: CacheStore,
    model: str,
    candidate: LemmaCandidate,
    thinking: bool,
) -> dict[str, Any]:
    fallback_response = build_singleton_fallback_response(candidate)
    review, cache_entry = await run_cached_structured_completion(
        client=client,
        cache=cache,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(candidate),
        input_payload=candidate.model_dump(mode="json"),
        response_model=LexemeResolutionResponse,
        thinking=thinking,
        sanitizer=lambda response: sanitize_lexeme_resolution(candidate, response),
        fallback_response=fallback_response,
    )
    if review is None:
        return {"rows": [], "cache": cache_entry}
    if not isinstance(review, LexemeResolutionResponse):
        raise RuntimeError("Lexeme review did not match the expected response model.")

    rows: list[dict[str, Any]] = []
    cache_entries: list[dict[str, Any]] = [cache_entry]

    for lexeme in review.lexemes:
        if lexeme.word_class != LexemeClass.pronoun:
            rows.append(
                {
                    "original_forms": lexeme.original_forms,
                    "normalized_form": lexeme.normalized_form,
                    "word_class": lexeme.word_class.value,
                }
            )
            continue

        pronoun_input = build_pronoun_input(candidate, lexeme)
        pronoun_fallback = build_pronoun_fallback_response(lexeme)
        pronoun_review, pronoun_cache = await run_cached_structured_completion(
            client=client,
            cache=cache,
            model=model,
            system_prompt=PRONOUN_PROMPT,
            user_prompt=json.dumps(pronoun_input, ensure_ascii=False),
            input_payload=pronoun_input,
            response_model=PronounResolutionResponse,
            thinking=thinking,
            sanitizer=lambda response, lexeme=lexeme: sanitize_pronoun_resolution(lexeme, response),
            fallback_response=pronoun_fallback,
        )
        cache_entries.append(pronoun_cache)

        if pronoun_review is None:
            continue
        if not isinstance(pronoun_review, PronounResolutionResponse):
            raise RuntimeError("Pronoun review did not match the expected response model.")

        rows.extend(
            {
                "original_forms": pronoun.original_forms,
                "normalized_form": pronoun.normalized_form,
                "word_class": LexemeClass.pronoun.value,
            }
            for pronoun in pronoun_review.pronouns
        )

    return {
        "rows": rows,
        "cache": {
            "cache_hit": all(entry["cache_hit"] for entry in cache_entries),
            "elapsed_ms": sum(entry["elapsed_ms"] or 0.0 for entry in cache_entries),
            "status": "completed" if all(entry["status"] == "completed" for entry in cache_entries) else "failed",
            "error_text": next((entry["error_text"] for entry in reversed(cache_entries) if entry["error_text"]), None),
        },
    }


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for row in rows:
            for output_row in row.get("rows", []):
                handle.write(orjson.dumps(output_row))
                handle.write(b"\n")


async def run_reviews(
    *,
    input_path: Path,
    output_path: Path,
    cache_path: Path,
    base_url: str,
    api_key: str,
    model_name: str | None,
    concurrency: int,
    thinking: bool,
) -> tuple[str, list[dict[str, Any]]]:
    candidates = load_candidates(input_path)

    cache = CacheStore(cache_path)
    await cache.open()
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    model = await pick_model(client, model_name)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(candidate: LemmaCandidate) -> dict[str, Any]:
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
        help="Path to the upstream lemma JSONL file.",
    ),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--output",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save the review JSONL output.",
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
        resolved_model, results = asyncio.run(
            run_reviews(
                input_path=input_path,
                output_path=output_path,
                cache_path=cache_path,
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
    typer.echo(f"output={output_path}")
    typer.echo(f"cache_db={cache_path}")
    typer.echo(f"candidate_count={len(results)}")
    typer.echo(f"completed={sum(1 for row in results if row['cache']['status'] == 'completed')}")
    typer.echo(f"failed={sum(1 for row in results if row['cache']['status'] == 'failed')}")
    typer.echo(f"cache_hits={sum(1 for row in results if row['cache']['cache_hit'])}")


if __name__ == "__main__":
    app()
