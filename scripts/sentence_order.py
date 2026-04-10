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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import typer
from pydantic import BaseModel, ConfigDict, Field

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEFAULT_INPUT_PATH = Path("data/lemma_normalisation_preprocess.jsonl")
DEFAULT_ENTRIES_INPUT_PATH = Path("data/lemma_entries.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/sentence_order.jsonl")
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
class SentenceAnalysis:
    sentence: SentenceContext | None
    dependencies: tuple[str, ...]
    external_unknown_forms: tuple[str, ...]
    token_count: int
    candidate_sentence_count: int


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


def build_form_maps(rows: list[TranslationRow]) -> tuple[dict[str, str], dict[str, str]]:
    exact_sets: dict[str, set[str]] = {}
    casefold_sets: dict[str, set[str]] = {}
    for row in rows:
        for form in row.original_forms:
            exact_sets.setdefault(form, set()).add(row.normalized_form)
            casefold_sets.setdefault(form.casefold(), set()).add(row.normalized_form)
        exact_sets.setdefault(row.normalized_form, set()).add(row.normalized_form)
        casefold_sets.setdefault(row.normalized_form.casefold(), set()).add(row.normalized_form)

    exact = {form: next(iter(forms)) for form, forms in exact_sets.items() if len(forms) == 1}
    casefold = {form: next(iter(forms)) for form, forms in casefold_sets.items() if len(forms) == 1}
    return exact, casefold


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
) -> tuple[list[str], list[str], int]:
    dependencies: list[str] = []
    external_unknown_forms: list[str] = []
    for token in tokenize(sentence.sentence_text):
        normalized = map_token(token, exact_form_map, casefold_form_map)
        if normalized is None:
            if is_greek_token(token):
                external_unknown_forms.append(token.lower())
            continue
        if normalized != row.normalized_form:
            dependencies.append(normalized)

    return unique_strings(dependencies), unique_strings(external_unknown_forms), len(tokenize(sentence.sentence_text))


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


def best_analysis_for_known(
    analyses: list[SentenceAnalysis],
    known: set[str],
) -> tuple[SentenceAnalysis, list[str], tuple[int, int, int, str]]:
    best: tuple[SentenceAnalysis, list[str], tuple[int, int, int, str]] | None = None
    for analysis in analyses:
        dependency_violations = [dependency for dependency in analysis.dependencies if dependency not in known]
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


def build_ordered_rows(rows: list[TranslationRow], analyses_by_row: list[list[SentenceAnalysis]]) -> list[dict[str, Any]]:
    remaining = set(range(len(rows)))
    known: set[str] = set()
    output_rows: list[dict[str, Any]] = []

    while remaining:
        best_choice: tuple[tuple[int, int, int, int, int, int], int, SentenceAnalysis, list[str]] | None = None
        for input_index in remaining:
            row = rows[input_index]
            analysis, dependency_violations, score = best_analysis_for_known(analyses_by_row[input_index], known)
            choice_score = (
                score[0] + score[1],
                score[0],
                score[1],
                score[2],
                -row.occurrence_count,
                input_index,
            )
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
                "dependency_normalized_forms": list(analysis.dependencies),
                "dependency_violations": dependency_violations,
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
        known.add(row.normalized_form)
        remaining.remove(input_index)

    return output_rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


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
) -> None:
    rows = [row for row in load_jsonl(input_path, TranslationRow) if isinstance(row, TranslationRow)]
    entries = [entry for entry in load_jsonl(entries_input_path, LemmaEntry) if isinstance(entry, LemmaEntry)]
    entry_index = build_entry_index(entries)
    rows = enrich_rows(rows, entry_index)
    exact_form_map, casefold_form_map = build_form_maps(rows)
    analyses_by_row = [
        build_sentence_analyses(row, entry_index, exact_form_map, casefold_form_map)
        for row in rows
    ]
    ordered_rows = build_ordered_rows(rows, analyses_by_row)
    write_jsonl(ordered_rows, output_path)

    typer.echo(f"output={output_path}")
    typer.echo(f"row_count={len(ordered_rows)}")
    typer.echo(
        f"only_target_unknown={sum(1 for row in ordered_rows if row['non_target_unknown_count_after_ordering'] == 0)}"
    )
    typer.echo(f"dependency_violations={sum(len(row['dependency_violations']) for row in ordered_rows)}")
    typer.echo(f"cycle_breaks={sum(1 for row in ordered_rows if row['cycle_break'])}")


if __name__ == "__main__":
    app()
