# Translation-first Anki Pipeline Plan

1. **Create new script**
   - New file: `scripts/translation_pass.py`.
   - Do not modify card generation yet.
   - Use `scripts/lemma_review.py` as the model for structure:
     - Typer CLI
     - async OpenAI-compatible calls
     - SQLite prompt cache
     - structured Pydantic response
     - retry logic with `tenacity`
     - JSONL output writer
     - reasoning effort included in cache key/payload

2. **Inputs**
   - Main input: `data/lemma_normalisation_preprocess.jsonl`.
   - Sentence/context input: `data/lemma_entries.jsonl`.
   - The normalized input rows have:
     ```json
     {"original_forms":["..."],"normalized_form":"...","word_class":"..."}
     ```
   - `lemma_entries.jsonl` supplies:
     - source candidate lemma
     - occurrence count
     - original forms
     - subtitle sentence contexts

3. **Candidate construction**
   - Load all normalized rows.
   - Load all lemma entries.
   - For each normalized row:
     - preserve `normalized_form`
     - preserve `word_class`
     - preserve `original_forms`
     - find matching lemma entries whose `original_forms` contain the row’s forms
     - aggregate:
       - `occurrence_count`
       - `source_candidate_lemmas`
       - matching sentences
   - Deduplicate original forms and sentences while preserving order.
   - Pick one example sentence for first-pass output using deterministic seed, not optimized yet.

4. **First pass limit/order**
   - Add CLI `--limit`, default `25`.
   - For first validation run, translate around 25 words only.
   - Use input order unless you prefer top-frequency; top-frequency can be added as `--sort frequency`.
   - First pass includes function words as ordinary unknown/learnable items conceptually, but no sentence-ordering logic yet.

5. **Model settings**
   - Default model:
     ```text
     qwen/qwen3.5-122b-a10b
     ```
   - Default reasoning effort:
     ```text
     none
     ```
   - Include:
     ```python
     extra_body={
       "chat_template_kwargs": {"enable_thinking": thinking},
       "reasoning": {"effort": reasoning_effort},
       "cache_prompt": True,
     }
     ```
   - Include model, prompt hashes, parameters hash, and schema hash in cache key.

6. **Translation prompt**
   - Task: translate the Greek `normalized_form` into English.
   - Translation must be for the normalized form, not the example sentence.
   - Use `word_class`, `original_forms`, and sentences for disambiguation.
   - Return **all common learner-relevant meanings** as a list of English strings.
   - Meanings should be concise, but not artificially narrowed to only one subtitle sense.
   - Function words should be translated/explained as learner-facing English meanings, e.g. articles/particles/prepositions.
   - No transliteration, no justification, no sentence translation.

7. **Structured model response**
   - Response shape:
     ```json
     {"translations":["be","exist"]}
     ```
   - Pydantic:
     ```python
     class TranslationResponse(BaseModel):
         translations: list[str] = Field(..., min_length=1)
     ```
   - Sanitize response:
     - strip whitespace
     - remove empty strings
     - dedupe translations preserving order
     - fail if empty after sanitization

8. **Output JSONL**
   - One line per successfully translated normalized word.
   - No failed rows in output.
   - Each row:
     ```json
     {
       "normalized_form": "...",
       "word_class": "...",
       "translations": ["...", "..."],
       "example_sentence": {
         "sentence_text": "...",
         "source_srt": "..."
       },
       "original_forms": ["..."],
       "source_candidate_lemmas": ["..."],
       "occurrence_count": 123
     }
     ```
   - Optional cache/model metadata can be printed in CLI summary, not necessarily written per row unless useful.

9. **Retries/failure behavior**
   - Use retry logic like `lemma_review.py`.
   - Retry transient API errors:
     - rate limit
     - timeout
     - connection error
     - API error
   - If a candidate still fails:
     - mark cache row as failed
     - omit it from output JSONL
     - report failure count and errors in CLI summary
   - No fallback fake translation.

10. **Validation comes immediately after translation pass**
   - Run only ~25 words.
   - Inspect JSONL manually before doing anything else.
   - Validate:
     - translations are English
     - translations are lists
     - multiple meanings are captured where appropriate
     - output is not translating the whole sentence
     - function words look acceptable
     - verbs/nouns/adjectives/proper nouns are handled well
     - example sentence is present and sensible enough as context
   - Adjust prompt/schema if needed before scaling.

11. **Only after translation quality is accepted: sentence-learning optimization**
   - Build a normalized vocabulary map from all input `original_forms`.
   - The union of `original_forms` should cover corpus words.
   - Tokenize sentences and map observed forms to normalized forms.
   - Maintain a growing known set.
   - Choose one sentence per target word such that each sentence has as few unknown words as possible, ideally only the target.

12. **Function-word modes for later sentence optimization**
   - Mode `unknown` first:
     - function words are normal learnable items
     - they count as unknown until learned
   - Mode `known` later:
     - articles, particles, prepositions, conjunctions, pronouns, etc. can be pre-seeded as known and/or excluded from target cards.
   - Compare both modes after translation validation.

13. **Later ordering/loss function**
   - Start with high-frequency words early, low-frequency words late.
   - For each candidate `(target_word, sentence)`, loss should heavily penalize unknown non-target words.
   - Example shape:
     ```text
     loss =
       high_penalty * unknown_non_target_words
       + sentence_length_penalty
       + subtitle_noise_penalty
       + low_frequency_early_penalty
       - target_frequency_bonus
     ```
   - Begin greedy frequency-descending.
   - Later improve with local swaps/beam search if needed.

14. **Execution command after implementation**
   ```bash
   uv run scripts/translation_pass.py \
     --input data/lemma_normalisation_preprocess.jsonl \
     --entries-input data/lemma_entries.jsonl \
     --output data/translation_pass.head25.jsonl \
     --cache-db data/translation_pass_cache.sqlite3 \
     --model qwen/qwen3.5-122b-a10b \
     --reasoning-effort none \
     --limit 25
   ```

One remaining choice: first 25 should be raw input order, or top 25 by occurrence count?
