from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from subs2anki.db.core import create_engine_for_path, create_schema
from subs2anki.db.models import (
    NormalizedLexeme,
    NormalizedLexemeOrdering,
    NormalizedLexemeReviewedLexeme,
    OriginalForm,
    ReviewedLexeme,
    ReviewedLexemeOriginalForm,
)


def _word_class_priority(word_class: str) -> tuple[int, int, str]:
    return (
        1 if word_class == "other" else 0,
        1 if word_class == "proper_noun" else 0,
        word_class,
    )


def _normalized_form_priority(normalized_form: str) -> tuple[int, str]:
    return (
        0 if normalized_form == normalized_form.casefold() else 1,
        normalized_form,
    )


def rebuild_normalized_lexemes(db_path: Path, scope_id: int) -> dict[str, Any]:
    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
        with Session(engine) as session:
            rows = session.execute(
                select(
                    ReviewedLexeme.id.label("reviewed_lexeme_id"),
                    ReviewedLexeme.normalized_form,
                    ReviewedLexeme.word_class,
                    OriginalForm.total_occurrences.label("form_occurrence_count"),
                )
                .join(
                    ReviewedLexemeOriginalForm,
                    ReviewedLexemeOriginalForm.reviewed_lexeme_id == ReviewedLexeme.id,
                )
                .join(OriginalForm, OriginalForm.id == ReviewedLexemeOriginalForm.original_form_id)
                .where(ReviewedLexeme.scope_id == scope_id)
                .order_by(ReviewedLexeme.normalized_form, ReviewedLexeme.id, OriginalForm.id)
            ).all()

            session.query(NormalizedLexemeOrdering).filter(
                NormalizedLexemeOrdering.scope_id == scope_id
            ).delete()

            scoped_normalized_ids = list(
                session.scalars(
                    select(NormalizedLexeme.id).where(NormalizedLexeme.scope_id == scope_id)
                )
            )
            if scoped_normalized_ids:
                session.query(NormalizedLexemeReviewedLexeme).filter(
                    NormalizedLexemeReviewedLexeme.normalized_lexeme_id.in_(scoped_normalized_ids)
                ).delete(synchronize_session=False)
                session.query(NormalizedLexeme).filter(
                    NormalizedLexeme.scope_id == scope_id
                ).delete(synchronize_session=False)
            session.flush()

            grouped_reviewed_lexeme_ids: dict[str, set[int]] = defaultdict(set)
            grouped_word_class_weights: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            grouped_normalized_form_weights: dict[str, dict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )

            for row in rows:
                normalized_form = str(row.normalized_form).strip()
                if not normalized_form:
                    continue
                normalized_key = normalized_form.casefold()
                reviewed_lexeme_id = int(row.reviewed_lexeme_id)
                weight = int(row.form_occurrence_count or 0)

                grouped_reviewed_lexeme_ids[normalized_key].add(reviewed_lexeme_id)
                grouped_word_class_weights[normalized_key][str(row.word_class)] += weight
                grouped_normalized_form_weights[normalized_key][normalized_form] += weight

            created_count = 0
            merged_reviewed_lexeme_count = 0
            duplicate_group_count = 0
            duplicate_reviewed_lexeme_count = 0

            for normalized_key in sorted(grouped_reviewed_lexeme_ids):
                reviewed_lexeme_ids = sorted(grouped_reviewed_lexeme_ids[normalized_key])
                word_class_weights = grouped_word_class_weights[normalized_key]
                normalized_form_weights = grouped_normalized_form_weights[normalized_key]

                chosen_word_class = min(
                    word_class_weights,
                    key=lambda value: (
                        -word_class_weights[value],
                        _word_class_priority(value),
                    ),
                )
                chosen_normalized_form = min(
                    normalized_form_weights,
                    key=lambda value: (
                        -normalized_form_weights[value],
                        _normalized_form_priority(value),
                    ),
                )

                normalized_lexeme = NormalizedLexeme(
                    scope_id=scope_id,
                    normalized_form=chosen_normalized_form,
                    word_class=chosen_word_class,
                )
                session.add(normalized_lexeme)
                session.flush()

                for reviewed_lexeme_id in reviewed_lexeme_ids:
                    session.add(
                        NormalizedLexemeReviewedLexeme(
                            normalized_lexeme_id=normalized_lexeme.id,
                            reviewed_lexeme_id=reviewed_lexeme_id,
                        )
                    )

                created_count += 1
                merged_reviewed_lexeme_count += len(reviewed_lexeme_ids)
                if len(reviewed_lexeme_ids) > 1:
                    duplicate_group_count += 1
                    duplicate_reviewed_lexeme_count += len(reviewed_lexeme_ids)

            session.commit()
    finally:
        engine.dispose()

    return {
        "normalized_lexeme_count": created_count,
        "merged_reviewed_lexeme_count": merged_reviewed_lexeme_count,
        "duplicate_group_count": duplicate_group_count,
        "duplicate_reviewed_lexeme_count": duplicate_reviewed_lexeme_count,
    }
