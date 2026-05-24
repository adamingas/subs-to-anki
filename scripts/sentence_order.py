#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "orjson>=3.10.16",
#   "pydantic>=2.11.0",
#   "typer>=0.16.0",
# ]
# ///

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import typer
from pydantic import BaseModel, ConfigDict, Field

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEFAULT_INPUT_PATH = Path("data/lemma_normalisation_preprocess.jsonl")
DEFAULT_ENTRIES_INPUT_PATH = Path("data/lemma_entries.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/sentence_order.full.jsonl")
DEFAULT_METRICS_DB_PATH = Path("data/sentence_order_metrics.sqlite3")
SRT_TAG_PATTERN = re.compile(r"\{\\[^}]+\}")
TOKEN_PATTERN = re.compile(r"[A-Za-zΑ-Ωα-ωΆ-ώΪΫϊϋΐΰ]+")
GREEK_PATTERN = re.compile(r"[Α-Ωα-ωΆ-ώΪΫϊϋΐΰ]")


class SentenceContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    sentence_text: str
    source_srt: str


class TranslationRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    normalized_form: str
    word_class: str | None = None
    translations: list[str] = Field(default_factory=list)
    example_sentence: SentenceContext | None = None
    original_forms: list[str] = Field(default_factory=list)
    source_candidate_lemmas: list[str] = Field(default_factory=list)
    occurrence_count: int = 0


class LemmaEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    lemmatised_form: str
    count: int
    original_forms: list[str] = Field(default_factory=list)
    sentences: list[SentenceContext] = Field(default_factory=list)


EntryIndex = dict[str, list[LemmaEntry]]


@dataclass(frozen=True)
class DependencyInfo:
    form: str
    normalized_form: str


@dataclass(frozen=True)
class SentenceAnalysis:
    sentence: SentenceContext | None
    dependencies: tuple[DependencyInfo, ...]
    external_unknown_forms: tuple[str, ...]
    token_count: int
    candidate_sentence_count: int


FormMaps = tuple[dict[str, str], dict[str, str], dict[str, str]]


def load_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    rows: list[BaseModel] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            rows.append(model.model_validate(orjson.loads(raw_line)))
        except Exception as exc:
            raise RuntimeError(f"Invalid JSONL row at {path}:{line_number}: {exc}") from exc
    return rows


def unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def unique_dependencies(values: list[DependencyInfo]) -> list[DependencyInfo]:
    seen: set[str] = set()
    unique: list[DependencyInfo] = []
    for value in values:
        if value.form in seen:
            continue
        seen.add(value.form)
        unique.append(value)
    return unique


def unique_sentences(sentences: list[SentenceContext]) -> list[SentenceContext]:
    seen: set[tuple[str, str]] = set()
    unique: list[SentenceContext] = []
    for sentence in sentences:
        key = (sentence.source_srt, sentence.sentence_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(sentence)
    return unique


def form_pattern(form: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(form)}(?!\w)", re.IGNORECASE)


def sentence_matches_any_form(sentence_text: str, forms: list[str]) -> bool:
    return any(form_pattern(form).search(sentence_text) for form in forms)


def build_entry_index(entries: list[LemmaEntry]) -> EntryIndex:
    index: EntryIndex = {}
    for entry in entries:
        for form in entry.original_forms:
            index.setdefault(form, []).append(entry)
    return index


def find_matching_entries(row: TranslationRow, entry_index: EntryIndex) -> list[LemmaEntry]:
    seen: set[int] = set()
    matching: list[LemmaEntry] = []
    for form in row.original_forms:
        for entry in entry_index.get(form, []):
            key = id(entry)
            if key in seen:
                continue
            seen.add(key)
            matching.append(entry)
    return matching


def enrich_rows(rows: list[TranslationRow], entry_index: EntryIndex) -> list[TranslationRow]:
    enriched: list[TranslationRow] = []
    for row in rows:
        matching_entries = find_matching_entries(row, entry_index)
        update: dict[str, Any] = {
            "occurrence_count": sum(entry.count for entry in matching_entries),
            "source_candidate_lemmas": unique_strings(
                [entry.lemmatised_form for entry in matching_entries if entry.lemmatised_form]
            ),
        }
        enriched.append(row.model_copy(update=update))
    return enriched


def clean_sentence_text(sentence_text: str) -> str:
    return SRT_TAG_PATTERN.sub(" ", sentence_text)


def tokenize(sentence_text: str) -> list[str]:
    return TOKEN_PATTERN.findall(clean_sentence_text(sentence_text))


def is_greek_token(token: str) -> bool:
    return bool(GREEK_PATTERN.search(token))


def is_noisy_sentence(sentence_text: str) -> bool:
    text = sentence_text.strip()
    return bool(re.search(r"\.{2,}|\d", text))


def build_form_maps(rows: list[TranslationRow]) -> FormMaps:
    exact_form_map: dict[str, str] = {}
    casefold_form_map: dict[str, str] = {}
    form_label_map: dict[str, str] = {}
    for row in rows:
        for form in row.original_forms:
            if not form:
                continue
            exact_form_map.setdefault(form, form)
            casefold_form_map.setdefault(form.casefold(), form)
            form_label_map.setdefault(form, row.normalized_form)
    return exact_form_map, casefold_form_map, form_label_map


def map_token(token: str, exact_form_map: dict[str, str], casefold_form_map: dict[str, str]) -> str | None:
    exact = exact_form_map.get(token)
    if exact is not None:
        return exact
    return casefold_form_map.get(token.casefold())


def analyze_sentence(
    row: TranslationRow,
    sentence: SentenceContext,
    exact_form_map: dict[str, str],
    casefold_form_map: dict[str, str],
    form_label_map: dict[str, str],
) -> tuple[list[DependencyInfo], list[str], int]:
    dependencies: list[DependencyInfo] = []
    external_unknown_forms: list[str] = []
    target_forms = set(row.original_forms)
    target_casefold_forms = {form.casefold() for form in row.original_forms}
    tokens = tokenize(sentence.sentence_text)
    for token in tokens:
        form = map_token(token, exact_form_map, casefold_form_map)
        if form is None:
            if is_greek_token(token):
                external_unknown_forms.append(token.casefold())
            continue
        if form in target_forms or form.casefold() in target_casefold_forms:
            continue
        dependencies.append(DependencyInfo(form=form, normalized_form=form_label_map.get(form, form)))

    return unique_dependencies(dependencies), unique_strings(external_unknown_forms), len(tokens)


def collect_candidate_sentences(row: TranslationRow, entry_index: EntryIndex) -> list[SentenceContext]:
    candidates: list[SentenceContext] = []
    for entry in find_matching_entries(row, entry_index):
        candidates.extend(
            sentence
            for sentence in entry.sentences
            if sentence_matches_any_form(sentence.sentence_text, row.original_forms)
        )
    if not candidates and row.example_sentence is not None:
        candidates.append(row.example_sentence)
    return unique_sentences(candidates)


def build_sentence_analyses(
    row: TranslationRow,
    entry_index: EntryIndex,
    exact_form_map: dict[str, str],
    casefold_form_map: dict[str, str],
    form_label_map: dict[str, str],
) -> list[SentenceAnalysis]:
    candidates = collect_candidate_sentences(row, entry_index)
    if not candidates:
        return [SentenceAnalysis(None, (), (), 0, 0)]
    clean_candidates = [sentence for sentence in candidates if not is_noisy_sentence(sentence.sentence_text)]
    scored_candidates = clean_candidates or candidates
    analyses: list[SentenceAnalysis] = []

    for sentence in scored_candidates:
        dependencies, external_unknown_forms, token_count = analyze_sentence(
            row,
            sentence,
            exact_form_map,
            casefold_form_map,
            form_label_map,
        )
        analyses.append(
            SentenceAnalysis(
                sentence=sentence,
                dependencies=tuple(dependencies),
                external_unknown_forms=tuple(external_unknown_forms),
                token_count=token_count,
                candidate_sentence_count=len(candidates),
            )
        )

    return sorted(
        analyses,
        key=lambda analysis: (
            len(analysis.external_unknown_forms),
            len(analysis.dependencies),
            analysis.token_count,
            analysis.sentence.sentence_text if analysis.sentence else "",
        ),
    )


def dependency_demand(analyses_by_row: list[list[SentenceAnalysis]]) -> Counter[str]:
    demand: Counter[str] = Counter()
    for analyses in analyses_by_row:
        row_forms: set[str] = set()
        for analysis in analyses:
            row_forms.update(dependency.form for dependency in analysis.dependencies)
        demand.update(row_forms)
    return demand


def zero_requirement_sets(analyses_by_row: list[list[SentenceAnalysis]]) -> list[list[frozenset[str]]]:
    requirements_by_row: list[list[frozenset[str]]] = []
    for analyses in analyses_by_row:
        row_requirements: list[frozenset[str]] = []
        seen: set[frozenset[str]] = set()
        for analysis in analyses:
            if analysis.sentence is None or analysis.external_unknown_forms:
                continue
            requirements = frozenset(dependency.form.casefold() for dependency in analysis.dependencies)
            if requirements in seen:
                continue
            seen.add(requirements)
            row_requirements.append(requirements)
        requirements_by_row.append(sorted(row_requirements, key=lambda values: (len(values), sorted(values))))
    return requirements_by_row


def row_form_keys(row: TranslationRow) -> set[str]:
    return {form.casefold() for form in row.original_forms}


def row_can_be_zero(requirements_by_row: list[list[frozenset[str]]], input_index: int, known_keys: set[str]) -> bool:
    return any(requirements <= known_keys for requirements in requirements_by_row[input_index])


def closure_size_after_seed(
    seed_index: int,
    remaining: set[int],
    known_keys: set[str],
    requirements_by_row: list[list[frozenset[str]]],
    row_form_key_sets: list[set[str]],
) -> int:
    simulated_known = set(known_keys)
    simulated_known.update(row_form_key_sets[seed_index])
    simulated_remaining = set(remaining)
    simulated_remaining.remove(seed_index)
    closure_size = 0
    changed = True
    while changed:
        changed = False
        for input_index in list(simulated_remaining):
            if not row_can_be_zero(requirements_by_row, input_index, simulated_known):
                continue
            simulated_remaining.remove(input_index)
            simulated_known.update(row_form_key_sets[input_index])
            closure_size += 1
            changed = True
    return closure_size


def known_form_set(row: TranslationRow) -> set[str]:
    return set(row.original_forms) | {form.casefold() for form in row.original_forms}


def is_known_form(form: str, known: set[str]) -> bool:
    return form in known or form.casefold() in known


def best_analysis_for_known(
    analyses: list[SentenceAnalysis],
    known: set[str],
) -> tuple[SentenceAnalysis, list[DependencyInfo], tuple[int, int, int, str]]:
    best: tuple[SentenceAnalysis, list[DependencyInfo], tuple[int, int, int, str]] | None = None
    for analysis in analyses:
        dependency_violations = [
            dependency for dependency in analysis.dependencies if not is_known_form(dependency.form, known)
        ]
        if analysis.sentence is None:
            score = (999, 999, 999, "")
        else:
            score = (
                len(dependency_violations),
                len(analysis.external_unknown_forms),
                analysis.token_count,
                analysis.sentence.sentence_text,
            )
        if best is None or score < best[2]:
            best = (analysis, dependency_violations, score)
            if score[:2] == (0, 0):
                break
    if best is None:
        raise RuntimeError("No sentence analyses available.")
    return best


def unlock_potential(row: TranslationRow, known: set[str], demand: Counter[str]) -> int:
    return sum(demand[form] for form in row.original_forms if not is_known_form(form, known))


def choice_sort_key(
    algorithm: str,
    score: tuple[int, int, int, str],
    row: TranslationRow,
    input_index: int,
    known: set[str],
    demand: Counter[str],
) -> tuple[int, ...]:
    total_unknown = score[0] + score[1]
    unlock = unlock_potential(row, known, demand)
    if algorithm == "greedy_unlock":
        return (total_unknown, score[0], score[1], -unlock, score[2], -row.occurrence_count, input_index)
    if algorithm == "greedy_frequency_unlock":
        return (total_unknown, score[0], score[1], -row.occurrence_count, -unlock, score[2], input_index)
    if algorithm == "greedy_unlock_frequency":
        return (total_unknown, score[0], score[1], -unlock, -row.occurrence_count, score[2], input_index)
    return (total_unknown, score[0], score[1], score[2], -row.occurrence_count, input_index)


def build_ordered_rows(
    rows: list[TranslationRow],
    analyses_by_row: list[list[SentenceAnalysis]],
    algorithm: str,
) -> list[dict[str, Any]]:
    remaining = set(range(len(rows)))
    known: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    demand = dependency_demand(analyses_by_row)
    requirements_by_row = zero_requirement_sets(analyses_by_row)
    row_form_key_sets = [row_form_keys(row) for row in rows]

    while remaining:
        best_choice: tuple[tuple[int, ...], int, SentenceAnalysis, list[DependencyInfo]] | None = None
        if algorithm == "closure_seed":
            zero_choice: tuple[tuple[int, ...], int, SentenceAnalysis, list[DependencyInfo]] | None = None
            seed_choice: tuple[tuple[int, ...], int, SentenceAnalysis, list[DependencyInfo]] | None = None
            known_keys = {form.casefold() for form in known}
            for input_index in remaining:
                row = rows[input_index]
                analysis, dependency_violations, score = best_analysis_for_known(analyses_by_row[input_index], known)
                total_unknown = score[0] + score[1]
                if total_unknown == 0:
                    choice_score = choice_sort_key("greedy_unlock", score, row, input_index, known, demand)
                    if zero_choice is None or choice_score < zero_choice[0]:
                        zero_choice = (choice_score, input_index, analysis, dependency_violations)
                    continue
                closure_size = closure_size_after_seed(
                    input_index,
                    remaining,
                    known_keys,
                    requirements_by_row,
                    row_form_key_sets,
                )
                choice_score = (
                    1 if analysis.sentence is None else 0,
                    -closure_size,
                    total_unknown,
                    score[0],
                    score[1],
                    score[2],
                    -row.occurrence_count,
                    input_index,
                )
                if seed_choice is None or choice_score < seed_choice[0]:
                    seed_choice = (choice_score, input_index, analysis, dependency_violations)
            best_choice = zero_choice or seed_choice
        else:
            for input_index in remaining:
                row = rows[input_index]
                analysis, dependency_violations, score = best_analysis_for_known(analyses_by_row[input_index], known)
                choice_score = choice_sort_key(algorithm, score, row, input_index, known, demand)
                if best_choice is None or choice_score < best_choice[0]:
                    best_choice = (choice_score, input_index, analysis, dependency_violations)

        if best_choice is None:
            raise RuntimeError("No remaining row could be ordered.")

        choice_score, input_index, analysis, dependency_violations = best_choice
        row = rows[input_index]
        non_target_unknown_count = (
            len(dependency_violations) + len(analysis.external_unknown_forms)
            if analysis.sentence is not None
            else 999
        )
        output_rows.append(
            {
                "order_index": len(output_rows) + 1,
                "input_index": input_index + 1,
                "normalized_form": row.normalized_form,
                "word_class": row.word_class,
                "translations": row.translations,
                "example_sentence": analysis.sentence.model_dump(mode="json") if analysis.sentence else None,
                "dependency_normalized_forms": unique_strings(
                    [dependency.normalized_form for dependency in analysis.dependencies]
                ),
                "dependency_violations": [dependency.form for dependency in dependency_violations],
                "external_unknown_forms": list(analysis.external_unknown_forms),
                "non_target_unknown_count_after_ordering": non_target_unknown_count,
                "total_unknown_count_after_ordering": non_target_unknown_count + 1,
                "sentence_token_count": analysis.token_count,
                "candidate_sentence_count": analysis.candidate_sentence_count,
                "cycle_break": len(dependency_violations) > 0,
                "original_forms": row.original_forms,
                "source_candidate_lemmas": row.source_candidate_lemmas,
                "occurrence_count": row.occurrence_count,
            }
        )
        known.update(known_form_set(row))
        remaining.remove(input_index)

    return output_rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


def summarise_metrics(ordered_rows: list[dict[str, Any]], algorithm: str) -> dict[str, Any]:
    unknown_counts = Counter(row["non_target_unknown_count_after_ordering"] for row in ordered_rows)
    first_no_sentence = next(
        (index + 1 for index, row in enumerate(ordered_rows) if row["example_sentence"] is None),
        None,
    )
    no_sentence = sum(1 for row in ordered_rows if row["example_sentence"] is None)
    dependency_violations = sum(len(row["dependency_violations"]) for row in ordered_rows)
    external_unknown_forms = sum(len(row["external_unknown_forms"]) for row in ordered_rows)
    non_target_unknowns = [
        row["non_target_unknown_count_after_ordering"]
        for row in ordered_rows
        if row["example_sentence"] is not None
    ]
    return {
        "algorithm": algorithm,
        "row_count": len(ordered_rows),
        "only_target_unknown": sum(
            1 for row in ordered_rows if row["non_target_unknown_count_after_ordering"] == 0
        ),
        "dependency_violations": dependency_violations,
        "external_unknown_forms": external_unknown_forms,
        "total_non_target_unknowns": dependency_violations + external_unknown_forms,
        "cycle_breaks": sum(1 for row in ordered_rows if row["cycle_break"]),
        "no_sentence": no_sentence,
        "first_no_sentence": first_no_sentence,
        "rows_with_external_unknowns": sum(1 for row in ordered_rows if row["external_unknown_forms"]),
        "rows_with_dependency_violations": sum(1 for row in ordered_rows if row["dependency_violations"]),
        "max_non_target_unknown_count": max(non_target_unknowns, default=0),
        "unknown_counts": sorted(unknown_counts.items()),
    }


def save_metrics(
    metrics_db_path: Path,
    metrics: dict[str, Any],
    input_path: Path,
    entries_input_path: Path,
    output_path: Path,
) -> int:
    metrics_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(metrics_db_path, timeout=30) as connection:
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sentence_order_attempts (
                attempt_number INTEGER PRIMARY KEY,
                date_time TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                input_path TEXT NOT NULL,
                entries_input_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                only_target_unknown INTEGER NOT NULL,
                dependency_violations INTEGER NOT NULL,
                external_unknown_forms INTEGER NOT NULL,
                total_non_target_unknowns INTEGER NOT NULL,
                cycle_breaks INTEGER NOT NULL,
                no_sentence INTEGER NOT NULL,
                first_no_sentence INTEGER,
                rows_with_external_unknowns INTEGER NOT NULL,
                rows_with_dependency_violations INTEGER NOT NULL,
                max_non_target_unknown_count INTEGER NOT NULL,
                unknown_counts_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            )
            """
        )
        attempt_number = int(
            connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM sentence_order_attempts"
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO sentence_order_attempts (
                attempt_number,
                date_time,
                algorithm,
                input_path,
                entries_input_path,
                output_path,
                row_count,
                only_target_unknown,
                dependency_violations,
                external_unknown_forms,
                total_non_target_unknowns,
                cycle_breaks,
                no_sentence,
                first_no_sentence,
                rows_with_external_unknowns,
                rows_with_dependency_violations,
                max_non_target_unknown_count,
                unknown_counts_json,
                metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_number,
                datetime.now(timezone.utc).isoformat(),
                metrics["algorithm"],
                str(input_path),
                str(entries_input_path),
                str(output_path),
                metrics["row_count"],
                metrics["only_target_unknown"],
                metrics["dependency_violations"],
                metrics["external_unknown_forms"],
                metrics["total_non_target_unknowns"],
                metrics["cycle_breaks"],
                metrics["no_sentence"],
                metrics["first_no_sentence"],
                metrics["rows_with_external_unknowns"],
                metrics["rows_with_dependency_violations"],
                metrics["max_non_target_unknown_count"],
                orjson.dumps(metrics["unknown_counts"]).decode(),
                orjson.dumps(metrics).decode(),
            ),
        )
    return attempt_number


@app.command()
def main(
    input_path: Path = typer.Option(
        DEFAULT_INPUT_PATH,
        "--input",
        "--translations-input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to normalized rows or translation-pass JSONL output.",
    ),
    entries_input_path: Path = typer.Option(
        DEFAULT_ENTRIES_INPUT_PATH,
        "--entries-input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to lemma entries JSONL with sentence contexts.",
    ),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--output",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to write ordered sentence JSONL.",
    ),
    metrics_db_path: Path = typer.Option(
        DEFAULT_METRICS_DB_PATH,
        "--metrics-db",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to SQLite DB for sentence-ordering attempt metrics.",
    ),
    algorithm: str = typer.Option(
        "closure_seed",
        "--algorithm",
        help="Ordering algorithm: closure_seed, greedy, greedy_unlock, greedy_unlock_frequency, or greedy_frequency_unlock.",
    ),
) -> None:
    valid_algorithms = {
        "closure_seed",
        "greedy",
        "greedy_unlock",
        "greedy_unlock_frequency",
        "greedy_frequency_unlock",
    }
    if algorithm not in valid_algorithms:
        raise typer.BadParameter(f"algorithm must be one of: {', '.join(sorted(valid_algorithms))}")

    rows = [row for row in load_jsonl(input_path, TranslationRow) if isinstance(row, TranslationRow)]
    entries = [entry for entry in load_jsonl(entries_input_path, LemmaEntry) if isinstance(entry, LemmaEntry)]
    entry_index = build_entry_index(entries)
    rows = enrich_rows(rows, entry_index)
    exact_form_map, casefold_form_map, form_label_map = build_form_maps(rows)
    analyses_by_row = [
        build_sentence_analyses(row, entry_index, exact_form_map, casefold_form_map, form_label_map)
        for row in rows
    ]
    ordered_rows = build_ordered_rows(rows, analyses_by_row, algorithm)
    write_jsonl(ordered_rows, output_path)
    metrics = summarise_metrics(ordered_rows, algorithm)
    attempt_number = save_metrics(metrics_db_path, metrics, input_path, entries_input_path, output_path)

    typer.echo(f"output={output_path}")
    typer.echo(f"metrics_db={metrics_db_path}")
    typer.echo(f"attempt_number={attempt_number}")
    typer.echo(f"row_count={metrics['row_count']}")
    typer.echo(f"only_target_unknown={metrics['only_target_unknown']}")
    typer.echo(f"dependency_violations={metrics['dependency_violations']}")
    typer.echo(f"external_unknown_forms={metrics['external_unknown_forms']}")
    typer.echo(f"cycle_breaks={metrics['cycle_breaks']}")
    typer.echo(f"no_sentence={metrics['no_sentence']}")
    typer.echo(f"first_no_sentence={metrics['first_no_sentence']}")
    typer.echo(f"unknown_counts={metrics['unknown_counts']}")


if __name__ == "__main__":
    app()
