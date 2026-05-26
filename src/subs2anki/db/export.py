from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from subs2anki.db.core import create_engine_for_path
from subs2anki.db.models import OriginalForm, Sentence, TokenOccurrence


def collect_lemma_entries(session: Session, scope_id: int) -> list[dict[str, Any]]:
    statement = (
        select(TokenOccurrence)
        .options(
            joinedload(TokenOccurrence.original_form).joinedload(OriginalForm.lemma),
            joinedload(TokenOccurrence.sentence).joinedload(Sentence.subtitle_file),
        )
        .where(TokenOccurrence.scope_id == scope_id)
        .order_by(TokenOccurrence.id)
    )
    occurrences = session.scalars(statement).all()

    entries: dict[str, dict[str, Any]] = {}
    for occurrence in occurrences:
        original_form = occurrence.original_form
        lemma = original_form.lemma
        sentence = occurrence.sentence
        subtitle_file = sentence.subtitle_file

        entry = entries.setdefault(
            lemma.text,
            {
                "count": 0,
                "original_forms": [],
                "_original_forms_seen": set(),
                "sentences": [],
                "_sentence_keys": set(),
            },
        )
        entry["count"] += 1

        if original_form.text not in entry["_original_forms_seen"]:
            entry["_original_forms_seen"].add(original_form.text)
            entry["original_forms"].append(original_form.text)

        sentence_key = (subtitle_file.path, sentence.text)
        if sentence_key not in entry["_sentence_keys"]:
            entry["_sentence_keys"].add(sentence_key)
            entry["sentences"].append(
                {
                    "sentence_text": sentence.text,
                    "source_srt": subtitle_file.path,
                }
            )

    rows: list[dict[str, Any]] = []
    for lemmatised_form, entry in sorted(
        entries.items(),
        key=lambda item: (-int(item[1]["count"]), item[0]),
    ):
        rows.append(
            {
                "lemmatised_form": lemmatised_form,
                "count": int(entry["count"]),
                "original_forms": list(entry["original_forms"]),
                "sentences": list(entry["sentences"]),
            }
        )
    return rows


def write_jsonl(rows: Sequence[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return output_path


def export_lemma_entries_jsonl(db_path: Path, output_path: Path, scope_id: int) -> Path:
    engine = create_engine_for_path(db_path)
    with Session(engine) as session:
        rows = collect_lemma_entries(session, scope_id=scope_id)
    return write_jsonl(rows, output_path)
