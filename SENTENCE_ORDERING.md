# Sentence Ordering Task

## Goal

Build a sentence-ordering pipeline for Greek Anki cards that assigns one example sentence to each input row and orders rows so each chosen sentence contains as few unknown non-target Greek words as possible. The ideal card has exactly one unknown in its example sentence: the target row itself.

## Core Rule

Knownness is based on `original_forms`, not `normalized_form`.

- `original_forms` are the exact word forms seen in subtitle sentences.
- When a row is scheduled, all of its `original_forms` become known.
- Any later sentence may depend on any of those forms and count them as already seen.
- If the same form appears in multiple rows, no disambiguation is needed. The first earlier row that introduced that form makes it known.
- `normalized_form` is the card label and output label. It is not the unit used to decide whether a later sentence word is known.

## Inputs

### `data/lemma_normalisation_preprocess.jsonl`

Each row:

```json
{"original_forms":["..."],"normalized_form":"...","word_class":"..."}
```

- `normalized_form`: learner-facing lexeme for the card.
- `word_class`: passthrough metadata.
- `original_forms`: exact forms seen in subtitles for this row.

Duplicate `normalized_form` values must be preserved. Do not collapse rows.

### `data/lemma_entries.jsonl`

Each row:

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

- `lemmatised_form`: upstream lemma candidate.
- `count`: occurrence count.
- `original_forms`: observed subtitle forms.
- `sentences`: subtitle sentence contexts.

Match normalized rows to lemma entries by overlap between the row's `original_forms` and the entry's `original_forms`.

### Optional input

`data/translation_pass.head25.jsonl` can also be used. Translations are irrelevant for sentence ordering.

## Output

Write:

```text
data/sentence_order.full.jsonl
```

Representative row:

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
- `input_index`: 1-based row index from the input file.
- `example_sentence`: selected sentence for the card.
- `dependency_normalized_forms`: reporting/debug labels for matched non-target words in the selected sentence. Knownness must still be decided from `original_forms`.
- `dependency_violations`: matched non-target words not yet covered by previously seen `original_forms`.
- `external_unknown_forms`: Greek tokens in the selected sentence that do not match any input `original_forms`.
- `non_target_unknown_count_after_ordering`: `len(dependency_violations) + len(external_unknown_forms)`.
- `total_unknown_count_after_ordering`: non-target unknown count plus the target.
- `candidate_sentence_count`: number of candidate sentences found for this row.
- `cycle_break`: true when the selected sentence still has at least one non-target matched word not yet covered by the known `original_forms` set.

## Objective

Primary objective:

1. Maximize rows where `non_target_unknown_count_after_ordering == 0`.

Secondary objectives:

2. Minimize dependency violations.
3. Minimize external unknown forms.
4. Prefer shorter sentences when unknown counts tie.
5. Prefer higher-frequency rows earlier when sentence difficulty ties.
6. Push rows with no usable sentence to the end.
7. Avoid obvious subtitle-noise sentences where possible.

## Dependency Model

For each candidate sentence:

1. Tokenize the Greek sentence.
2. Compare each token against the union of all input `original_forms`.
3. Ignore tokens that belong to the target row's own `original_forms`.
4. Any other matched token is a dependency. It is known if it is already in the known `original_forms` set; otherwise it is a dependency violation.
5. Any Greek token that matches no input `original_forms` is an `external_unknown_form`.

Example:

```text
Target row:
  normalized_form: έχω
  original_forms: έχω, έχεις, έχει, έχουμε, έχετε, έχουν

Sentence:
  Κι εσύ έχεις.
```

Here `έχεις` counts as the target because it is one of the target row's `original_forms`. `κι`/`και` and `εσύ` must already be known for this to be an ideal sentence.

## Required Algorithm

Script:

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

Required greedy flow:

1. Load normalized rows and lemma entries.
2. Match rows to lemma entries by overlapping `original_forms`.
3. Collect candidate sentences, filter obvious subtitle noise, and pre-analyze tokens, dependencies, external unknowns, and token counts.
4. Repeatedly pick the remaining row and sentence with the best lexicographic score:
   - fewest total non-target unknowns
   - fewest dependency violations
   - fewest external unknowns
   - shortest sentence
   - highest occurrence count
   - earliest input row
5. After scheduling a row, add all of its `original_forms` to the known set.
6. Continue until all rows are ordered.

This is dynamic sentence selection, not a static topological sort. The best sentence for a row can change as more `original_forms` become known.

Implementation note: if the code keeps `known` as `normalized_form` values, that is wrong for this task. The known set must be the union of seen `original_forms`.

## Baseline Metrics

Current dynamic greedy baseline from the existing implementation:

```text
row_count=1156
only_target_unknown=895
dependency_violations=213
cycle_breaks=168
no_sentence=3
first_no_sentence=1154
unknown_counts=[(0, 895), (1, 202), (2, 46), (3, 7), (4, 3)]
```

These numbers are a baseline, not a guarantee after the `original_forms` known-set change.

## Known Limitations

- Tokenization is regex-based and not linguistically complete.
- Matching depends entirely on `original_forms`; if a sentence token is missing from that vocabulary, it becomes an external unknown.
- The algorithm is greedy and may miss better global orderings.
- One-occurrence rows are not explicitly pushed late except indirectly through frequency tiebreaking.
- Very short fragments can win because they minimize unknowns even when they are pedagogically weak.
- Subtitle-noise filtering is still simple.

## Future Improvements

1. Add an explicit low-frequency-late penalty so one-occurrence rows are delayed unless needed as dependencies.
2. Add beam search or local reorder passes after greedy ordering.
3. Improve fragment/noise filtering and sentence-quality heuristics.
4. Improve Greek token normalization, punctuation handling, apostrophe handling, and final-sigma normalization.
5. Add configurable function-word modes and richer summary metrics.
