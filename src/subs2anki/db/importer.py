from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import simplemma  # type: ignore
from sqlalchemy.orm import Session

from subs2anki.db.core import create_engine_for_path, create_schema, reset_database
from subs2anki.db.models import (
    Lemma,
    OriginalForm,
    OriginalFormFileCount,
    Sentence,
    SubtitleFile,
    TokenOccurrence,
)
from subs2anki.subtitles import (
    build_nlp,
    extract_document_sentences,
    extract_sentence_tokens,
    normalize_lemma,
    validate_input_paths,
)


@dataclass(slots=True)
class ImportStats:
    files: int = 0
    sentences: int = 0
    lemmas: int = 0
    original_forms: int = 0
    occurrences: int = 0


def get_or_create_lemma(
    session: Session,
    lemma_cache: dict[str, Lemma],
    lemma_text: str,
    stats: ImportStats,
) -> Lemma:
    lemma = lemma_cache.get(lemma_text)
    if lemma is not None:
        return lemma

    lemma = Lemma(text=lemma_text, total_occurrences=0)
    session.add(lemma)
    lemma_cache[lemma_text] = lemma
    stats.lemmas += 1
    return lemma


def get_or_create_original_form(
    session: Session,
    form_cache: dict[str, OriginalForm],
    form_text: str,
    lemma: Lemma,
    stats: ImportStats,
) -> OriginalForm:
    form = form_cache.get(form_text)
    if form is not None:
        if form.lemma is not lemma and form.lemma.text != lemma.text:
            raise RuntimeError(
                f"Original form {form_text!r} was assigned conflicting lemmas: "
                f"{form.lemma.text!r} and {lemma.text!r}"
            )
        return form

    form = OriginalForm(text=form_text, lemma=lemma, total_occurrences=0)
    session.add(form)
    form_cache[form_text] = form
    stats.original_forms += 1
    return form


def ingest_subtitle_file(
    session: Session,
    path: Path,
    nlp: Any,
    language: str,
    lemma_cache: dict[str, Lemma],
    form_cache: dict[str, OriginalForm],
    stats: ImportStats,
) -> None:
    subtitle_file = SubtitleFile(path=str(path), name=path.name)
    session.add(subtitle_file)
    session.flush()
    stats.files += 1

    file_counts: Counter[str] = Counter()

    for sequence_number, sentence_text in enumerate(extract_document_sentences(path, nlp), start=1):
        sentence = Sentence(
            subtitle_file=subtitle_file,
            sequence_number=sequence_number,
            text=sentence_text,
        )
        session.add(sentence)
        stats.sentences += 1

        for token_position, token_text in enumerate(extract_sentence_tokens(sentence_text), start=1):
            if not any(character.isalpha() for character in token_text):
                continue

            normalized_token = unicodedata.normalize("NFC", token_text)
            lemma_text = normalize_lemma(simplemma.lemmatize(normalized_token, lang=language))
            if not lemma_text:
                continue

            lemma = get_or_create_lemma(session, lemma_cache, lemma_text, stats)
            form = get_or_create_original_form(session, form_cache, normalized_token, lemma, stats)

            session.add(
                TokenOccurrence(
                    sentence=sentence,
                    original_form=form,
                    token_position=token_position,
                )
            )
            lemma.total_occurrences += 1
            form.total_occurrences += 1
            file_counts[form.text] += 1
            stats.occurrences += 1

    for form_text, occurrence_count in sorted(file_counts.items()):
        session.add(
            OriginalFormFileCount(
                original_form=form_cache[form_text],
                subtitle_file=subtitle_file,
                occurrence_count=occurrence_count,
            )
        )


def import_subtitles(
    files: Sequence[Path],
    db_path: Path,
    language: str,
) -> tuple[Path, ImportStats]:
    input_paths = validate_input_paths(files)
    resolved_db_path = reset_database(db_path)
    engine = create_engine_for_path(resolved_db_path)
    create_schema(engine)

    nlp = build_nlp(language)
    stats = ImportStats()
    lemma_cache: dict[str, Lemma] = {}
    form_cache: dict[str, OriginalForm] = {}

    with Session(engine) as session:
        for path in input_paths:
            ingest_subtitle_file(
                session,
                path=path,
                nlp=nlp,
                language=language,
                lemma_cache=lemma_cache,
                form_cache=form_cache,
                stats=stats,
            )
        session.commit()

    return resolved_db_path, stats
