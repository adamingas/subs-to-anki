from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class ReviewedLexemeRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_forms: list[str] = Field(..., min_length=1)
    normalized_form: str
    word_class: LexemeClass


class TranslationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewed_lexeme_id: int
    normalized_form: str
    word_class: LexemeClass
    source_lemma: str
    original_forms: list[str] = Field(..., min_length=1)
    sentences: list[SentenceContext] = Field(..., min_length=1)
    occurrence_count: int


class TranslationPromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewed_lexeme_id: int
    normalized_form: str
    word_class: LexemeClass
    source_lemma: str
    original_forms: list[str] = Field(..., min_length=1)
    example_sentence: SentenceContext
    occurrence_count: int


class TranslationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    english_translation: str = Field(..., min_length=1)
    example_sentence_translation: str = Field(..., min_length=1)


class CachedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_key: str
    status: Literal["completed", "failed", "pending", "running"]
    response_json: str | None = None
    reasoning_content: str | None = None
    elapsed_ms: float | None = None
    error_text: str | None = None
    cache_hit: bool = False
