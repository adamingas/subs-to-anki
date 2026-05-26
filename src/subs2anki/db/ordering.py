from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from subs2anki.db.core import create_engine_for_path, create_schema, resolve_db_path
from subs2anki.db.models import NormalizedLexemeOrdering, Sentence, SubtitleFile
from subs2anki.db.normalized import rebuild_normalized_lexemes
from subs2anki.db.translations import (
    clean_translation_sentence_text,
    load_translation_candidates,
    load_translation_sentence_id_lookup,
)
from subs2anki.llm.types import SentenceContext, TranslationCandidate
from subs2anki.sentence_ordering import (
    NormalizedLexemeOrderingRow,
    order_translation_candidates,
    summarise_ordering,
)


def rebuild_normalized_lexeme_ordering(
    db_path: Path,
    scope_id: int,
    algorithm: str = "closure_seed",
) -> dict[str, Any]:
    merge_metrics = rebuild_normalized_lexemes(db_path, scope_id=scope_id)
    candidates = load_translation_candidates(db_path, scope_id=scope_id)
    ordered_rows = order_translation_candidates(candidates, algorithm=algorithm)
    sentence_lookup = load_translation_sentence_id_lookup(db_path, scope_id=scope_id)

    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
    finally:
        engine.dispose()

    ordered_payload = []
    for row in ordered_rows:
        example_sentence_id = None
        if row.example_sentence is not None:
            example_sentence_id = sentence_lookup.get(
                (
                    row.normalized_lexeme_id,
                    row.example_sentence.source_srt,
                    row.example_sentence.sentence_text,
                )
            )
        ordered_payload.append(
            (
                row.normalized_lexeme_id,
                scope_id,
                row.order_index,
                example_sentence_id,
                row.algorithm,
                json.dumps(row.dependency_normalized_forms, ensure_ascii=False),
                json.dumps(row.dependency_violations, ensure_ascii=False),
                json.dumps(row.external_unknown_forms, ensure_ascii=False),
                row.non_target_unknown_count,
                row.total_unknown_count,
                row.sentence_token_count,
                row.candidate_sentence_count,
                1 if row.cycle_break else 0,
            )
        )

    with sqlite3.connect(resolve_db_path(db_path), timeout=30) as connection:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute(
            "DELETE FROM normalized_lexeme_ordering WHERE scope_id = ?",
            (scope_id,),
        )
        connection.executemany(
            """
            INSERT INTO normalized_lexeme_ordering (
                normalized_lexeme_id,
                scope_id,
                order_index,
                example_sentence_id,
                algorithm,
                dependency_normalized_forms_json,
                dependency_violations_json,
                external_unknown_forms_json,
                non_target_unknown_count,
                total_unknown_count,
                sentence_token_count,
                candidate_sentence_count,
                cycle_break
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ordered_payload,
        )
        connection.commit()

    metrics = summarise_ordering(ordered_rows)
    metrics["algorithm"] = algorithm
    metrics.update(merge_metrics)
    return metrics


def load_ordered_translation_candidates(db_path: Path, scope_id: int) -> list[TranslationCandidate]:
    candidates = load_translation_candidates(db_path, scope_id=scope_id)
    candidates_by_id = {candidate.normalized_lexeme_id: candidate for candidate in candidates}

    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
        with Session(engine) as session:
            ordering_rows = session.execute(
                select(
                    NormalizedLexemeOrdering.normalized_lexeme_id,
                    NormalizedLexemeOrdering.order_index,
                    Sentence.text,
                    SubtitleFile.path,
                )
                .join(Sentence, Sentence.id == NormalizedLexemeOrdering.example_sentence_id, isouter=True)
                .join(SubtitleFile, SubtitleFile.id == Sentence.subtitle_file_id, isouter=True)
                .where(NormalizedLexemeOrdering.scope_id == scope_id)
                .order_by(NormalizedLexemeOrdering.order_index)
            ).all()
    finally:
        engine.dispose()

    if not ordering_rows:
        return sorted(candidates, key=lambda candidate: candidate.normalized_lexeme_id)

    ordered_candidates: list[TranslationCandidate] = []
    seen_ids: set[int] = set()
    for normalized_lexeme_id, _, sentence_text, source_srt in ordering_rows:
        candidate = candidates_by_id.get(int(normalized_lexeme_id))
        if candidate is None:
            continue
        example_sentence = None
        if sentence_text is not None and source_srt is not None:
            cleaned_sentence_text = clean_translation_sentence_text(str(sentence_text))
            sentence_match = next(
                (
                    sentence
                    for sentence in candidate.sentences
                    if sentence.sentence_text == cleaned_sentence_text and sentence.source_srt == source_srt
                ),
                None,
            )
            if sentence_match is not None:
                example_sentence = sentence_match
            else:
                example_sentence = SentenceContext(
                    sentence_text=cleaned_sentence_text,
                    source_srt=str(source_srt),
                )
        ordered_candidates.append(candidate.model_copy(update={"example_sentence": example_sentence}))
        seen_ids.add(candidate.normalized_lexeme_id)

    for candidate in sorted(candidates, key=lambda value: value.normalized_lexeme_id):
        if candidate.normalized_lexeme_id in seen_ids:
            continue
        ordered_candidates.append(candidate)

    return ordered_candidates
