# Sentence Ordering Task

## Goal

Build a sentence-ordering pipeline for Greek Anki cards that chooses one example sentence per normalized word and orders the words so that each card's example sentence contains as few unknown non-target words as possible.

The ideal card has exactly one unknown word in the example sentence: the target word itself. Any other Greek word in the chosen sentence should already have appeared earlier in the ordered card list.

This task is independent of translation quality. It can run directly on normalized Greek lexeme rows before translations exist.

## Inputs

### Normalized word input

Primary input:

```text
data/lemma_normalisation_preprocess.jsonl
```

Each row is JSONL with this shape:

```json
{"original_forms":["..."],"normalized_form":"...","word_class":"..."}
```

Fields:

- `normalized_form`: learner-facing normalized word/lexeme to make a card for.
- `word_class`: upstream word class metadata, kept in output.
- `original_forms`: observed surface forms in subtitles that map to this normalized word.

The file can contain duplicate `normalized_form` values in different rows. The ordering script must preserve all input rows, not collapse duplicates.

### Sentence/context input

Sentence input:

```text
data/lemma_entries.jsonl
```

Each row has this rough shape:

```json
{
  "lemmatised_form":"...",
  "count":123,
  "original_forms":["..."],
  "sentences":[
    {"sentence_text":"...","source_srt":"data/raw/example.srt"}
  ]
}
```

Fields used:

- `lemmatised_form`: upstream source lemma candidate.
- `count`: occurrence count for that upstream lemma row.
- `original_forms`: observed subtitle forms.
- `sentences`: subtitle sentence contexts containing those forms.

The ordering script matches normalized rows to lemma entries by overlap between the normalized row's `original_forms` and the lemma entry's `original_forms`.

### Optional translation-pass input

The same ordering script can also accept translation-pass output:

```text
data/translation_pass.head25.jsonl
```

Those rows include translations, but translations are not required for sentence ordering.

## Output

Full ordering output currently written to:

```text
data/sentence_order.full.jsonl
```

Each output row includes:

```json
{
  "order_index": 1,
  "input_index": 4,
  "normalized_form": "είμαι",
  "word_class": "verb",
  "translations": [],
  "example_sentence": {
    "sentence_text": "Είμαι.",
    "source_srt": "data/raw/...srt"
  },
  "dependency_normalized_forms": [],
  "dependency_violations": [],
  "external_unknown_forms": [],
  "non_target_unknown_count_after_ordering": 0,
  "total_unknown_count_after_ordering": 1,
  "sentence_token_count": 1,
  "candidate_sentence_count": 224,
  "cycle_break": false,
  "original_forms": ["..."],
  "source_candidate_lemmas": ["..."],
  "occurrence_count": 233
}
```

Important fields:

- `order_index`: final card order.
- `input_index`: 1-based source row index from the input file.
- `example_sentence`: selected sentence for the card.
- `dependency_normalized_forms`: known normalized words from this input set that appear in the selected sentence besides the target.
- `dependency_violations`: dependency words in the selected sentence that have not appeared earlier in the ordering.
- `external_unknown_forms`: Greek tokens in the selected sentence that could not be mapped to any normalized row in the current input set.
- `non_target_unknown_count_after_ordering`: `len(dependency_violations) + len(external_unknown_forms)`.
- `total_unknown_count_after_ordering`: non-target unknown count plus the target word itself.
- `candidate_sentence_count`: number of candidate sentences found for this row.
- `cycle_break`: true when the selected sentence still depends on a not-yet-known normalized word.

## Optimization objective

Primary objective:

1. Maximize rows where `non_target_unknown_count_after_ordering == 0`.

That means the selected sentence contains only one unknown word: the target.

Secondary objectives:

2. Minimize dependency violations.
3. Minimize external unknown Greek forms.
4. Prefer shorter sentences when unknown counts tie.
5. Prefer higher-frequency words earlier when sentence difficulty ties.
6. Push rows with no usable sentence to the end.
7. Avoid obvious subtitle-noise sentences where possible.

The loss for a candidate row/sentence at a given point in the ordering is conceptually:

```text
loss =
  unknown_non_target_count
  + dependency_violation_penalty
  + external_unknown_penalty
  + sentence_length_penalty
  + no_sentence_penalty
  - frequency_tiebreak
```

The current implementation uses lexicographic greedy scoring rather than one numeric weighted loss.

## Dependency model

For each candidate sentence for a target row:

1. Tokenize the Greek sentence.
2. Map each observed token to a `normalized_form` using the union of all input `original_forms`.
3. Ignore the target word itself.
4. Any other mapped normalized form becomes a dependency.
5. Any Greek token that cannot be mapped to an input normalized form becomes an `external_unknown_form`.

If a target word's selected sentence contains another normalized word, that other word should ideally appear earlier in the card order.

Example:

```text
Target: έχω
Sentence: Κι εσύ έχεις.
Dependencies: και, εσύ
```

So `και` and `εσύ` should come before `έχω` for that chosen sentence to have only the target unknown.

## Algorithm idea requested

The user suggested a dependency-graph approach:

1. Start with words that have only one occurrence.
2. Look at their example sentence.
3. Find what other words they depend on.
4. All one-occurrence words should tend to be late.
5. Any words they depend on must come before them.
6. Then recursively inspect the words that those last words depend on.
7. Build up a dependency graph.
8. Order the graph so prerequisites come before dependent words.

This captures the core requirement: sentences create prerequisite edges from a target word to the other words in its sentence.

## Current implemented approach

The current script is:

```text
scripts/sentence_order.py
```

Run command:

```bash
uv run scripts/sentence_order.py \
  --input data/lemma_normalisation_preprocess.jsonl \
  --entries-input data/lemma_entries.jsonl \
  --output data/sentence_order.full.jsonl
```

The script now uses dynamic greedy sentence selection:

1. Load all normalized rows.
2. Load all lemma entries.
3. Build an index from every observed original form to matching lemma entries.
4. Enrich normalized rows with:
   - `occurrence_count`
   - `source_candidate_lemmas`
5. Build a vocabulary map from every input row's `original_forms` to `normalized_form`.
6. For every row, collect all candidate sentences from matching lemma entries.
7. Filter obvious subtitle-noise candidates where possible:
   - repeated dots / ellipsis-like text
   - digits
8. Pre-analyze every candidate sentence into:
   - mapped normalized dependencies
   - external unknown forms
   - token count
9. Start with an empty `known` set.
10. Repeatedly choose the remaining row and sentence with best current score:
    - fewest total non-target unknowns
    - fewest dependency violations
    - fewest external unknowns
    - shortest sentence
    - highest occurrence count as tiebreaker
    - earlier input index as final tiebreaker
11. Add the chosen row to output.
12. Add its `normalized_form` to the `known` set.
13. Continue until all rows are ordered.

This differs from a static graph topological sort because the best sentence for a word can change as the known set grows. Dynamic selection lets the script choose the easiest sentence available at the exact point the word is scheduled.

## Metrics

The script prints summary metrics after running.

Initial static approach on full input:

```text
row_count=1156
only_target_unknown=781
dependency_violations=402
cycle_breaks=311
```

Improved dynamic greedy approach on full input:

```text
row_count=1156
only_target_unknown=895
dependency_violations=213
cycle_breaks=168
```

Additional validation from the latest run:

```text
no_sentence=3
first_no_sentence=1154
unknown_counts=[(0, 895), (1, 202), (2, 46), (3, 7), (4, 3)]
```

Meaning:

- 1156 total input rows were preserved.
- 895 rows have example sentences where the only unknown is the target word.
- 202 rows have one extra non-target unknown.
- 46 rows have two extra non-target unknowns.
- 7 rows have three extra non-target unknowns.
- 3 rows have four extra non-target unknowns.
- 3 rows have no usable sentence and are pushed to the end.

## Known limitations

- Tokenization is regex-based, not linguistically complete.
- Surface-form mapping depends entirely on `original_forms`; if a sentence token is absent from the normalized vocabulary map, it becomes an external unknown.
- Duplicate normalized forms are preserved as separate input rows, but dependency resolution maps dependencies by normalized form.
- The algorithm is greedy and may miss globally better orderings.
- It does not yet explicitly force one-occurrence words late, except indirectly through frequency tiebreaking.
- Some selected sentences are very short fragments, which can be useful for minimizing unknowns but may be pedagogically weak.
- Subtitle noise filtering is simple and only catches obvious cases.

## Future improvements

Potential improvements:

1. Add an explicit low-frequency-late penalty so one-occurrence words are delayed unless they are needed as dependencies.
2. Add beam search or local swaps after greedy ordering.
3. Penalize sentence fragments that are too short or lack enough semantic context.
4. Improve subtitle-noise detection.
5. Improve Greek token normalization, punctuation handling, apostrophe handling, and final-sigma normalization.
6. Prefer sentences where the target appears in a clear lexical context, not just one-word dialogue.
7. Add modes for function words:
   - `unknown`: function words are ordinary learnable targets.
   - `known`: articles, particles, prepositions, conjunctions, pronouns can be pre-seeded as known.
8. Add separate metrics for:
   - dependency violations
   - external unknown forms
   - no-sentence rows
   - average non-target unknown count
   - median non-target unknown count
   - max non-target unknown count
