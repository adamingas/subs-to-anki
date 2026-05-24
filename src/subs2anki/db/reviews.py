from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from subs2anki.db.core import create_engine_for_path, create_schema
from subs2anki.db.export import collect_lemma_entries, write_jsonl
from subs2anki.db.models import (
    Lemma,
    OriginalForm,
    ReviewedLexeme,
    ReviewedLexemeOriginalForm,
    Sentence,
    SubtitleFile,
    TokenOccurrence,
)
from subs2anki.llm.types import LemmaCandidate, ReviewedLexemeRow, SentenceContext


def load_review_candidates(db_path: Path) -> list[LemmaCandidate]:
    engine = create_engine_for_path(db_path)
    try:
        with Session(engine) as session:
            rows = collect_lemma_entries(session)
    finally:
        engine.dispose()
    return [LemmaCandidate.model_validate(row) for row in rows]


def replace_reviewed_lexemes(
    db_path: Path,
    review_rows_by_lemma: dict[str, list[ReviewedLexemeRow]],
) -> None:
    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
        with Session(engine) as session:
            lemma_texts = sorted(review_rows_by_lemma)
            source_lemmas = {
                lemma.text: lemma
                for lemma in session.scalars(
                    select(Lemma)
                    .options(joinedload(Lemma.original_forms))
                    .where(Lemma.text.in_(lemma_texts))
                ).unique()
            }

            missing = [lemma_text for lemma_text in lemma_texts if lemma_text not in source_lemmas]
            if missing:
                raise RuntimeError(f"Could not find source lemmas in DB: {', '.join(missing)}")

            for source_lemma in source_lemmas.values():
                for existing in list(source_lemma.reviewed_lexemes):
                    session.delete(existing)

            session.flush()

            for lemma_text in lemma_texts:
                source_lemma = source_lemmas[lemma_text]
                forms_by_text = {form.text: form for form in source_lemma.original_forms}

                for row in review_rows_by_lemma[lemma_text]:
                    reviewed_lexeme = ReviewedLexeme(
                        source_lemma=source_lemma,
                        normalized_form=row.normalized_form,
                        word_class=row.word_class.value,
                    )
                    session.add(reviewed_lexeme)
                    session.flush()

                    for form_text in row.original_forms:
                        original_form = forms_by_text.get(form_text)
                        if original_form is None:
                            raise RuntimeError(
                                f"Original form {form_text!r} is not attached to source lemma {lemma_text!r}."
                            )
                        session.add(
                            ReviewedLexemeOriginalForm(
                                reviewed_lexeme=reviewed_lexeme,
                                original_form=original_form,
                            )
                        )

            session.commit()
    finally:
        engine.dispose()


def collect_reviewed_lexeme_rows(session: Session) -> list[dict[str, Any]]:
    reviewed_lexemes = session.scalars(
        select(ReviewedLexeme)
        .options(
            joinedload(ReviewedLexeme.source_lemma),
            joinedload(ReviewedLexeme.reviewed_original_forms).joinedload(
                ReviewedLexemeOriginalForm.original_form
            ),
        )
        .order_by(ReviewedLexeme.source_lemma_id, ReviewedLexeme.normalized_form, ReviewedLexeme.word_class)
    ).unique()

    rows: list[dict[str, Any]] = []
    for reviewed_lexeme in reviewed_lexemes:
        original_forms = sorted(
            (link.original_form.text for link in reviewed_lexeme.reviewed_original_forms),
            key=str.casefold,
        )
        if not original_forms:
            continue
        rows.append(
            {
                "original_forms": original_forms,
                "normalized_form": reviewed_lexeme.normalized_form,
                "word_class": reviewed_lexeme.word_class,
            }
        )
    return rows


def export_reviewed_lexemes_jsonl(db_path: Path, output_path: Path) -> Path:
    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
        with Session(engine) as session:
            rows = collect_reviewed_lexeme_rows(session)
    finally:
        engine.dispose()
    return write_jsonl(rows, output_path)


def load_reviewed_lexeme_sentences(db_path: Path, reviewed_lexeme_id: int) -> list[SentenceContext]:
    engine = create_engine_for_path(db_path)
    try:
        with Session(engine) as session:
            sentence_rows = session.execute(
                select(Sentence.text, SubtitleFile.path)
                .join(TokenOccurrence, TokenOccurrence.sentence_id == Sentence.id)
                .join(OriginalForm, OriginalForm.id == TokenOccurrence.original_form_id)
                .join(
                    ReviewedLexemeOriginalForm,
                    ReviewedLexemeOriginalForm.original_form_id == OriginalForm.id,
                )
                .join(SubtitleFile, SubtitleFile.id == Sentence.subtitle_file_id)
                .where(ReviewedLexemeOriginalForm.reviewed_lexeme_id == reviewed_lexeme_id)
                .distinct()
                .order_by(Sentence.id)
            ).all()
    finally:
        engine.dispose()

    return [
        SentenceContext(sentence_text=sentence_text, source_srt=subtitle_path)
        for sentence_text, subtitle_path in sentence_rows
    ]
