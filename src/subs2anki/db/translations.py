from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from subs2anki.db.core import create_engine_for_path, create_schema
from subs2anki.db.models import (
    Lemma,
    NormalizedLexeme,
    NormalizedLexemeReviewedLexeme,
    OriginalForm,
    ReviewedLexeme,
    ReviewedLexemeOriginalForm,
    Sentence,
    SubtitleFile,
    TokenOccurrence,
)
from subs2anki.llm.types import LexemeClass, SentenceContext, TranslationCandidate
from subs2anki.subtitles import clean_subtitle_text

SRT_TAG_PATTERN = re.compile(r"\{\\[^}]+\}")


def _clean_sentence_text(text: str) -> str:
    return clean_subtitle_text(SRT_TAG_PATTERN.sub(" ", text))


def clean_translation_sentence_text(text: str) -> str:
    return _clean_sentence_text(text)


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def load_translation_candidates(db_path: Path, scope_id: int) -> list[TranslationCandidate]:
    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
        with Session(engine) as session:
            rows = session.execute(
                select(
                    NormalizedLexeme.id.label("normalized_lexeme_id"),
                    NormalizedLexeme.normalized_form,
                    NormalizedLexeme.word_class,
                    Lemma.text.label("source_lemma"),
                    OriginalForm.id.label("original_form_id"),
                    OriginalForm.text.label("original_form"),
                    TokenOccurrence.id.label("token_occurrence_id"),
                    Sentence.id.label("sentence_id"),
                    Sentence.text.label("sentence_text"),
                    SubtitleFile.path.label("source_srt"),
                )
                .join(
                    NormalizedLexemeReviewedLexeme,
                    NormalizedLexemeReviewedLexeme.normalized_lexeme_id == NormalizedLexeme.id,
                )
                .join(ReviewedLexeme, ReviewedLexeme.id == NormalizedLexemeReviewedLexeme.reviewed_lexeme_id)
                .join(Lemma, Lemma.id == ReviewedLexeme.source_lemma_id)
                .join(
                    ReviewedLexemeOriginalForm,
                    ReviewedLexemeOriginalForm.reviewed_lexeme_id == ReviewedLexeme.id,
                )
                .join(OriginalForm, OriginalForm.id == ReviewedLexemeOriginalForm.original_form_id)
                .join(TokenOccurrence, TokenOccurrence.original_form_id == OriginalForm.id)
                .join(Sentence, Sentence.id == TokenOccurrence.sentence_id)
                .join(SubtitleFile, SubtitleFile.id == Sentence.subtitle_file_id)
                .where(NormalizedLexeme.scope_id == scope_id)
                .order_by(NormalizedLexeme.id, Sentence.id, TokenOccurrence.token_position)
            ).all()
    finally:
        engine.dispose()

    buckets: dict[int, dict[str, object]] = {}
    for row in rows:
        normalized_lexeme_id = int(row.normalized_lexeme_id)
        bucket = buckets.setdefault(
            normalized_lexeme_id,
            {
                "normalized_form": str(row.normalized_form),
                "word_class": str(row.word_class),
                "source_lemmas": [],
                "_source_lemma_keys": set(),
                "original_forms": [],
                "_original_form_ids": set(),
                "sentences": [],
                "_sentence_keys": set(),
                "_token_occurrence_ids": set(),
                "occurrence_count": 0,
            },
        )

        cast_source_lemma_keys = bucket["_source_lemma_keys"]
        cast_source_lemmas = bucket["source_lemmas"]
        assert isinstance(cast_source_lemma_keys, set)
        assert isinstance(cast_source_lemmas, list)
        source_lemma = str(row.source_lemma).strip()
        if source_lemma and source_lemma not in cast_source_lemma_keys:
            cast_source_lemma_keys.add(source_lemma)
            cast_source_lemmas.append(source_lemma)

        cast_token_occurrence_ids = bucket["_token_occurrence_ids"]
        assert isinstance(cast_token_occurrence_ids, set)
        token_occurrence_id = int(row.token_occurrence_id)
        if token_occurrence_id not in cast_token_occurrence_ids:
            cast_token_occurrence_ids.add(token_occurrence_id)
            bucket["occurrence_count"] = int(bucket["occurrence_count"]) + 1

        cast_original_forms = bucket["original_forms"]
        cast_original_form_ids = bucket["_original_form_ids"]
        assert isinstance(cast_original_forms, list)
        assert isinstance(cast_original_form_ids, set)
        original_form_id = int(row.original_form_id)
        if original_form_id not in cast_original_form_ids:
            cast_original_form_ids.add(original_form_id)
            cast_original_forms.append(str(row.original_form))

        sentence_text = _clean_sentence_text(str(row.sentence_text))
        sentence_key = (str(row.source_srt), sentence_text)
        cast_sentence_keys = bucket["_sentence_keys"]
        cast_sentences = bucket["sentences"]
        assert isinstance(cast_sentence_keys, set)
        assert isinstance(cast_sentences, list)
        if sentence_text and sentence_key not in cast_sentence_keys:
            cast_sentence_keys.add(sentence_key)
            cast_sentences.append(
                SentenceContext(
                    sentence_text=sentence_text,
                    source_srt=str(row.source_srt),
                )
            )

    candidates: list[TranslationCandidate] = []
    for normalized_lexeme_id in sorted(buckets):
        bucket = buckets[normalized_lexeme_id]
        sentences = bucket["sentences"]
        if not isinstance(sentences, list) or not sentences:
            raise RuntimeError(
                f"No sentence context found for normalized lexeme id {normalized_lexeme_id}."
            )
        original_forms = bucket["original_forms"]
        if not isinstance(original_forms, list):
            raise RuntimeError(f"Invalid original forms bucket for normalized lexeme id {normalized_lexeme_id}.")
        source_lemmas = bucket["source_lemmas"]
        if not isinstance(source_lemmas, list):
            raise RuntimeError(f"Invalid source lemmas bucket for normalized lexeme id {normalized_lexeme_id}.")

        candidates.append(
            TranslationCandidate(
                normalized_lexeme_id=normalized_lexeme_id,
                normalized_form=str(bucket["normalized_form"]).strip(),
                word_class=LexemeClass(str(bucket["word_class"])),
                source_lemmas=_unique_preserving_order([str(lemma) for lemma in source_lemmas]),
                original_forms=_unique_preserving_order([str(form) for form in original_forms]),
                sentences=sentences,
                occurrence_count=int(bucket["occurrence_count"]),
            )
        )

    return candidates


def load_translation_sentence_id_lookup(
    db_path: Path,
    scope_id: int,
) -> dict[tuple[int, str, str], int]:
    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
        with Session(engine) as session:
            rows = session.execute(
                select(
                    NormalizedLexeme.id,
                    Sentence.id,
                    Sentence.text,
                    SubtitleFile.path,
                )
                .join(
                    NormalizedLexemeReviewedLexeme,
                    NormalizedLexemeReviewedLexeme.normalized_lexeme_id == NormalizedLexeme.id,
                )
                .join(ReviewedLexeme, ReviewedLexeme.id == NormalizedLexemeReviewedLexeme.reviewed_lexeme_id)
                .join(
                    ReviewedLexemeOriginalForm,
                    ReviewedLexemeOriginalForm.reviewed_lexeme_id == ReviewedLexeme.id,
                )
                .join(OriginalForm, OriginalForm.id == ReviewedLexemeOriginalForm.original_form_id)
                .join(TokenOccurrence, TokenOccurrence.original_form_id == OriginalForm.id)
                .join(Sentence, Sentence.id == TokenOccurrence.sentence_id)
                .join(SubtitleFile, SubtitleFile.id == Sentence.subtitle_file_id)
                .where(NormalizedLexeme.scope_id == scope_id)
                .order_by(NormalizedLexeme.id, Sentence.id)
            ).all()
    finally:
        engine.dispose()

    lookup: dict[tuple[int, str, str], int] = {}
    for normalized_lexeme_id, sentence_id, sentence_text, source_srt in rows:
        key = (int(normalized_lexeme_id), str(source_srt), _clean_sentence_text(str(sentence_text)))
        lookup.setdefault(key, int(sentence_id))
    return lookup
