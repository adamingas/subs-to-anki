from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from subs2anki.db.core import create_engine_for_path, create_schema
from subs2anki.db.models import (
    Lemma,
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


def load_translation_candidates(db_path: Path) -> list[TranslationCandidate]:
    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
        with Session(engine) as session:
            rows = session.execute(
                select(
                    ReviewedLexeme.id,
                    ReviewedLexeme.normalized_form,
                    ReviewedLexeme.word_class,
                    Lemma.text.label("source_lemma"),
                    OriginalForm.text.label("original_form"),
                    Sentence.id.label("sentence_id"),
                    Sentence.text.label("sentence_text"),
                    SubtitleFile.path.label("source_srt"),
                )
                .join(Lemma, Lemma.id == ReviewedLexeme.source_lemma_id)
                .join(
                    ReviewedLexemeOriginalForm,
                    ReviewedLexemeOriginalForm.reviewed_lexeme_id == ReviewedLexeme.id,
                )
                .join(OriginalForm, OriginalForm.id == ReviewedLexemeOriginalForm.original_form_id)
                .join(TokenOccurrence, TokenOccurrence.original_form_id == OriginalForm.id)
                .join(Sentence, Sentence.id == TokenOccurrence.sentence_id)
                .join(SubtitleFile, SubtitleFile.id == Sentence.subtitle_file_id)
                .order_by(ReviewedLexeme.id, Sentence.id, TokenOccurrence.token_position)
            ).all()
    finally:
        engine.dispose()

    buckets: dict[int, dict[str, object]] = {}
    for row in rows:
        reviewed_lexeme_id = int(row.id)
        bucket = buckets.setdefault(
            reviewed_lexeme_id,
            {
                "normalized_form": str(row.normalized_form),
                "word_class": str(row.word_class),
                "source_lemma": str(row.source_lemma),
                "original_forms": [],
                "sentences": [],
                "_sentence_keys": set(),
                "occurrence_count": 0,
            },
        )

        bucket["occurrence_count"] = int(bucket["occurrence_count"]) + 1
        cast_original_forms = bucket["original_forms"]
        assert isinstance(cast_original_forms, list)
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
    for reviewed_lexeme_id in sorted(buckets):
        bucket = buckets[reviewed_lexeme_id]
        sentences = bucket["sentences"]
        if not isinstance(sentences, list) or not sentences:
            raise RuntimeError(
                f"No sentence context found for reviewed lexeme id {reviewed_lexeme_id}."
            )
        original_forms = bucket["original_forms"]
        if not isinstance(original_forms, list):
            raise RuntimeError(f"Invalid original forms bucket for reviewed lexeme id {reviewed_lexeme_id}.")

        candidates.append(
            TranslationCandidate(
                reviewed_lexeme_id=reviewed_lexeme_id,
                normalized_form=str(bucket["normalized_form"]).strip(),
                word_class=LexemeClass(str(bucket["word_class"])),
                source_lemma=str(bucket["source_lemma"]).strip(),
                original_forms=_unique_preserving_order([str(form) for form in original_forms]),
                sentences=sentences,
                occurrence_count=int(bucket["occurrence_count"]),
            )
        )

    return candidates
