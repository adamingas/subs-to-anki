from __future__ import annotations

import json

from subs2anki.llm.types import SentenceContext, TranslationCandidate, TranslationPromptInput, TranslationResponse

TRANSLATION_SYSTEM_PROMPT = """You are creating concise learner-facing English translations for Modern Greek study cards.

Task:
Given one reviewed learner-facing normalized form and one selected Greek example sentence, return:
- a short learner-facing English translation for the target word
- a natural English translation of the selected example sentence

Rules:
- Translate `normalized_form`, not the source lemmas.
- Use `word_class`, `original_forms`, and `example_sentence` to disambiguate the sense.
- `english_translation` should be short and compact, suitable for the back of a flashcard.
- `example_sentence_translation` should be a natural full-sentence English translation of `example_sentence.sentence_text`.
- Prefer lowercase for `english_translation` unless the target is a proper noun.
- Do not explain, justify, transliterate, or add notes.
- Do not include quotation marks.

Input JSON:
{"normalized_form":"ηρεμώ","word_class":"verb","source_lemmas":["ηρέμησε","ηρεμήσω"],"original_forms":["ηρέμησε","Ηρέμησε","ηρεμήσω"],"example_sentence":{"sentence_text":"Θέλω να ηρεμήσω και δεν θέλω να πάρω χάπια.","source_srt":"data/raw/example.srt"},"occurrence_count":7}
Output JSON:
{"english_translation":"calm down; relax","example_sentence_translation":"I want to calm down and I don't want to take pills."}

Input JSON:
{"normalized_form":"σπίτι","word_class":"noun","source_lemmas":["σπίτι"],"original_forms":["σπίτι"],"example_sentence":{"sentence_text":"Χωρίς συγγνώμη, σπίτι δεν γυρίζεις.","source_srt":"data/raw/example.srt"},"occurrence_count":14}
Output JSON:
{"english_translation":"home; house","example_sentence_translation":"Without an apology, you're not going back home."}

Output format:
Return JSON with exactly this shape:
{"english_translation":"...","example_sentence_translation":"..."}
"""


def pick_example_sentence(candidate: TranslationCandidate) -> SentenceContext:
    if candidate.example_sentence is not None:
        return candidate.example_sentence
    if not candidate.sentences:
        raise RuntimeError(
            f"No sentence context available for normalized lexeme id {candidate.normalized_lexeme_id}."
        )
    return candidate.sentences[0]


def build_translation_prompt_input(candidate: TranslationCandidate) -> TranslationPromptInput:
    return TranslationPromptInput(
        normalized_form=candidate.normalized_form,
        word_class=candidate.word_class,
        source_lemmas=list(candidate.source_lemmas),
        original_forms=list(candidate.original_forms),
        example_sentence=pick_example_sentence(candidate),
        occurrence_count=candidate.occurrence_count,
    )


def build_translation_user_prompt(prompt_input: TranslationPromptInput) -> str:
    return json.dumps(prompt_input.model_dump(mode="json"), ensure_ascii=False)


def sanitize_translation_response(response: TranslationResponse) -> TranslationResponse:
    english_translation = response.english_translation.strip()
    example_sentence_translation = response.example_sentence_translation.strip()
    if not english_translation:
        raise RuntimeError("Translation response did not include an english_translation value.")
    if not example_sentence_translation:
        raise RuntimeError(
            "Translation response did not include an example_sentence_translation value."
        )
    return TranslationResponse(
        english_translation=english_translation,
        example_sentence_translation=example_sentence_translation,
    )
