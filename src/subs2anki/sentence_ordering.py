from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from subs2anki.llm.types import SentenceContext, TranslationCandidate

TOKEN_PATTERN = re.compile(r"[A-Za-zΑ-Ωα-ωΆ-ώΪΫϊϋΐΰ]+")
GREEK_PATTERN = re.compile(r"[Α-Ωα-ωΆ-ώΪΫϊϋΐΰ]")


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


class NormalizedLexemeOrderingRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_lexeme_id: int
    order_index: int
    example_sentence: SentenceContext | None = None
    dependency_normalized_forms: list[str]
    dependency_violations: list[str]
    external_unknown_forms: list[str]
    non_target_unknown_count: int
    total_unknown_count: int
    sentence_token_count: int
    candidate_sentence_count: int
    cycle_break: bool
    algorithm: str


def filter_sentence_length_candidates(analyses: list[SentenceAnalysis]) -> list[SentenceAnalysis]:
    has_multi_word_option = any(
        analysis.sentence is not None and analysis.token_count > 1
        for analysis in analyses
    )
    if not has_multi_word_option:
        return analyses
    filtered = [
        analysis
        for analysis in analyses
        if analysis.sentence is None or analysis.token_count > 1
    ]
    return filtered or analyses


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


def tokenize(sentence_text: str) -> list[str]:
    return TOKEN_PATTERN.findall(sentence_text)


def is_greek_token(token: str) -> bool:
    return bool(GREEK_PATTERN.search(token))


def is_noisy_sentence(sentence_text: str) -> bool:
    text = sentence_text.strip()
    return bool(re.search(r"\.{2,}|\d", text))


def build_form_maps(rows: list[TranslationCandidate]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
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
    row: TranslationCandidate,
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


def build_sentence_analyses(
    row: TranslationCandidate,
    exact_form_map: dict[str, str],
    casefold_form_map: dict[str, str],
    form_label_map: dict[str, str],
) -> list[SentenceAnalysis]:
    candidates = unique_sentences(list(row.sentences))
    if not candidates and row.example_sentence is not None:
        candidates = [row.example_sentence]
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

    filtered_analyses = filter_sentence_length_candidates(analyses)
    return sorted(
        filtered_analyses,
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


def row_form_keys(row: TranslationCandidate) -> set[str]:
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


def known_form_set(row: TranslationCandidate) -> set[str]:
    return set(row.original_forms) | {form.casefold() for form in row.original_forms}


def is_known_form(form: str, known: set[str]) -> bool:
    return form in known or form.casefold() in known


def best_analysis_for_known(
    analyses: list[SentenceAnalysis],
    known: set[str],
) -> tuple[SentenceAnalysis, list[DependencyInfo], tuple[int, int, int, str]]:
    best: tuple[SentenceAnalysis, list[DependencyInfo], tuple[int, int, int, str]] | None = None
    for analysis in filter_sentence_length_candidates(analyses):
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


def unlock_potential(row: TranslationCandidate, known: set[str], demand: Counter[str]) -> int:
    return sum(demand[form] for form in row.original_forms if not is_known_form(form, known))


def choice_sort_key(
    algorithm: str,
    score: tuple[int, int, int, str],
    row: TranslationCandidate,
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
    rows: list[TranslationCandidate],
    analyses_by_row: list[list[SentenceAnalysis]],
    algorithm: str,
) -> list[NormalizedLexemeOrderingRow]:
    remaining = set(range(len(rows)))
    known: set[str] = set()
    output_rows: list[NormalizedLexemeOrderingRow] = []
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

        _, input_index, analysis, dependency_violations = best_choice
        row = rows[input_index]
        non_target_unknown_count = (
            len(dependency_violations) + len(analysis.external_unknown_forms)
            if analysis.sentence is not None
            else 999
        )
        output_rows.append(
            NormalizedLexemeOrderingRow(
                normalized_lexeme_id=row.normalized_lexeme_id,
                order_index=len(output_rows) + 1,
                example_sentence=analysis.sentence,
                dependency_normalized_forms=unique_strings(
                    [dependency.normalized_form for dependency in analysis.dependencies]
                ),
                dependency_violations=[dependency.form for dependency in dependency_violations],
                external_unknown_forms=list(analysis.external_unknown_forms),
                non_target_unknown_count=non_target_unknown_count,
                total_unknown_count=non_target_unknown_count + 1,
                sentence_token_count=analysis.token_count,
                candidate_sentence_count=analysis.candidate_sentence_count,
                cycle_break=len(dependency_violations) > 0,
                algorithm=algorithm,
            )
        )
        known.update(known_form_set(row))
        remaining.remove(input_index)

    return output_rows


def order_translation_candidates(
    rows: list[TranslationCandidate],
    algorithm: str = "closure_seed",
) -> list[NormalizedLexemeOrderingRow]:
    valid_algorithms = {
        "closure_seed",
        "greedy",
        "greedy_unlock",
        "greedy_unlock_frequency",
        "greedy_frequency_unlock",
    }
    if algorithm not in valid_algorithms:
        raise RuntimeError(f"algorithm must be one of: {', '.join(sorted(valid_algorithms))}")

    exact_form_map, casefold_form_map, form_label_map = build_form_maps(rows)
    analyses_by_row = [
        build_sentence_analyses(row, exact_form_map, casefold_form_map, form_label_map)
        for row in rows
    ]
    return build_ordered_rows(rows, analyses_by_row, algorithm)


def summarise_ordering(rows: list[NormalizedLexemeOrderingRow]) -> dict[str, Any]:
    non_target_unknowns = Counter(row.non_target_unknown_count for row in rows)
    no_sentence_rows = [row.order_index for row in rows if row.example_sentence is None]
    return {
        "row_count": len(rows),
        "no_sentence": len(no_sentence_rows),
        "first_no_sentence": no_sentence_rows[0] if no_sentence_rows else None,
        "cycle_breaks": sum(1 for row in rows if row.cycle_break),
        "dependency_violations": sum(len(row.dependency_violations) for row in rows),
        "external_unknown_forms": sum(len(row.external_unknown_forms) for row in rows),
        "unknown_counts": sorted(non_target_unknowns.items()),
    }
