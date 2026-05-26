from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from subs2anki.db.core import create_cache_schema, create_engine_for_path, create_schema, resolve_db_path


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(str(column[1]) == column_name for column in columns)


def migrate_legacy_db_to_scoped_schema(
    db_path: Path,
    scope_name: str,
    language: str = "el",
) -> tuple[Path, Path | None]:
    resolved_db_path = resolve_db_path(db_path)
    if not resolved_db_path.exists():
        raise RuntimeError(f"Database not found: {resolved_db_path}")

    with sqlite3.connect(resolved_db_path) as old_connection:
        if _table_exists(old_connection, "scopes") and _column_exists(old_connection, "subtitle_files", "scope_id"):
            return resolved_db_path, None

    migrated_path = resolved_db_path.with_name(f"{resolved_db_path.stem}.scoped-migration.sqlite3")
    if migrated_path.exists():
        migrated_path.unlink()

    engine = create_engine_for_path(migrated_path)
    try:
        create_schema(engine)
        create_cache_schema(engine)
    finally:
        engine.dispose()

    with sqlite3.connect(resolved_db_path) as old_connection, sqlite3.connect(migrated_path) as new_connection:
        old_connection.row_factory = sqlite3.Row
        new_connection.execute("PRAGMA foreign_keys=OFF")

        new_connection.execute(
            "INSERT INTO scopes (id, name, language) VALUES (?, ?, ?)",
            (1, scope_name, language),
        )

        copy_jobs = [
            (
                "subtitle_files",
                "id, scope_id, path, name",
                "SELECT id, 1, path, name FROM subtitle_files",
            ),
            (
                "lemmas",
                "id, scope_id, text, total_occurrences",
                "SELECT id, 1, text, total_occurrences FROM lemmas",
            ),
            (
                "sentences",
                "id, scope_id, subtitle_file_id, sequence_number, text",
                "SELECT id, 1, subtitle_file_id, sequence_number, text FROM sentences",
            ),
            (
                "original_forms",
                "id, scope_id, text, lemma_id, total_occurrences",
                "SELECT id, 1, text, lemma_id, total_occurrences FROM original_forms",
            ),
            (
                "original_form_file_counts",
                "id, scope_id, original_form_id, subtitle_file_id, occurrence_count",
                "SELECT id, 1, original_form_id, subtitle_file_id, occurrence_count FROM original_form_file_counts",
            ),
            (
                "token_occurrences",
                "id, scope_id, sentence_id, original_form_id, token_position",
                "SELECT id, 1, sentence_id, original_form_id, token_position FROM token_occurrences",
            ),
            (
                "reviewed_lexemes",
                "id, scope_id, source_lemma_id, normalized_form, word_class, created_at, updated_at",
                "SELECT id, 1, source_lemma_id, normalized_form, word_class, created_at, updated_at FROM reviewed_lexemes",
            ),
            (
                "reviewed_lexeme_original_forms",
                "reviewed_lexeme_id, original_form_id",
                "SELECT reviewed_lexeme_id, original_form_id FROM reviewed_lexeme_original_forms",
            ),
            (
                "normalized_lexemes",
                "id, scope_id, normalized_form, word_class, created_at, updated_at",
                "SELECT id, 1, normalized_form, word_class, created_at, updated_at FROM normalized_lexemes",
            ),
            (
                "normalized_lexeme_reviewed_lexemes",
                "normalized_lexeme_id, reviewed_lexeme_id",
                "SELECT normalized_lexeme_id, reviewed_lexeme_id FROM normalized_lexeme_reviewed_lexemes",
            ),
            (
                "normalized_lexeme_ordering",
                (
                    "normalized_lexeme_id, scope_id, order_index, example_sentence_id, algorithm, "
                    "dependency_normalized_forms_json, dependency_violations_json, external_unknown_forms_json, "
                    "non_target_unknown_count, total_unknown_count, sentence_token_count, candidate_sentence_count, "
                    "cycle_break, created_at, updated_at"
                ),
                (
                    "SELECT normalized_lexeme_id, 1, order_index, example_sentence_id, algorithm, "
                    "dependency_normalized_forms_json, dependency_violations_json, external_unknown_forms_json, "
                    "non_target_unknown_count, total_unknown_count, sentence_token_count, candidate_sentence_count, "
                    "cycle_break, created_at, updated_at FROM normalized_lexeme_ordering"
                ),
            ),
            (
                "prompt_cache",
                (
                    "cache_key, model, system_prompt_hash, user_prompt_hash, parameters_hash, schema_hash, "
                    "system_prompt, user_prompt, parameters_json, input_json, status, response_json, "
                    "reasoning_content, elapsed_ms, error_text, created_at, updated_at"
                ),
                (
                    "SELECT cache_key, model, system_prompt_hash, user_prompt_hash, parameters_hash, schema_hash, "
                    "system_prompt, user_prompt, parameters_json, input_json, status, response_json, "
                    "reasoning_content, elapsed_ms, error_text, created_at, updated_at FROM prompt_cache"
                ),
            ),
        ]

        for table_name, columns, query in copy_jobs:
            if not _table_exists(old_connection, table_name):
                continue
            rows = old_connection.execute(query).fetchall()
            if not rows:
                continue
            placeholders = ", ".join("?" for _ in rows[0])
            new_connection.executemany(
                f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
                [tuple(row) for row in rows],
            )

        new_connection.commit()
        new_connection.execute("PRAGMA foreign_keys=ON")

    backup_path = resolved_db_path.with_name(f"{resolved_db_path.stem}.pre_scoped.sqlite3")
    if not backup_path.exists():
        shutil.copy2(resolved_db_path, backup_path)
    shutil.move(str(migrated_path), str(resolved_db_path))
    return resolved_db_path, backup_path
