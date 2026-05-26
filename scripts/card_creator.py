#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, ConfigDict

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEFAULT_INPUT_PATH = Path("data/translation_pass.all.jsonl")
DEFAULT_OUTPUT_TSV_PATH = Path("data/anki_cards.tsv")
DEFAULT_OUTPUT_APKG_PATH = Path("data/anki_cards.apkg")
DEFAULT_DECK_NAME = "Greek Lemmas"
ANKI_MODEL_ID = 1387426501


class SentenceContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    sentence_text: str
    source_srt: str


class TranslationPassRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    normalized_lexeme_id: int
    normalized_form: str
    word_class: str
    source_lemmas: list[str]
    original_forms: list[str]
    occurrence_count: int
    example_sentence: SentenceContext
    english_translation: str | None = None
    example_sentence_translation: str | None = None


class CardRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_lexeme_id: int
    word: str
    part_of_speech: str
    meaning: str
    example_sentence: str
    sentence_translation: str


def load_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]:
    rows: list[BaseModel] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            rows.append(model.model_validate(json.loads(raw_line)))
        except Exception as exc:
            raise RuntimeError(f"Invalid JSONL row at {path}:{line_number}: {exc}") from exc
    return rows


def filter_rows(
    rows: list[TranslationPassRow],
    *,
    exclude_proper_nouns: bool,
) -> list[TranslationPassRow]:
    if not exclude_proper_nouns:
        return rows
    return [row for row in rows if row.word_class != "proper_noun"]


def build_card_rows(rows: list[TranslationPassRow], limit: int | None) -> list[CardRow]:
    selected = rows[:limit] if limit is not None else rows
    cards: list[CardRow] = []
    seen_words: set[str] = set()
    for row in selected:
        if not row.english_translation or not row.example_sentence_translation:
            continue
        word = row.normalized_form.strip()
        dedupe_key = word.casefold()
        if dedupe_key in seen_words:
            continue
        seen_words.add(dedupe_key)
        cards.append(
            CardRow(
                normalized_lexeme_id=row.normalized_lexeme_id,
                word=word,
                part_of_speech=row.word_class.replace("_", " ").strip(),
                meaning=row.english_translation.strip(),
                example_sentence=row.example_sentence.sentence_text.strip(),
                sentence_translation=row.example_sentence_translation.strip(),
            )
        )
    return cards


def write_anki_tsv(rows: list[CardRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            values = [
                row.word,
                row.part_of_speech,
                row.meaning,
                row.example_sentence,
                row.sentence_translation,
            ]
            cleaned = [value.replace("\t", " ").replace("\n", " ").strip() for value in values]
            handle.write("\t".join(cleaned) + "\n")


def build_genanki_model() -> Any:
    import genanki

    return genanki.Model(
        ANKI_MODEL_ID,
        "Greek Lemma Card",
        fields=[
            {"name": "Word"},
            {"name": "PartOfSpeech"},
            {"name": "Meaning"},
            {"name": "ExampleSentence"},
            {"name": "SentenceTranslation"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": """
<div class="customCard">
  <div class="targetWordContainerFront">
    <div class="targetWord">{{Word}}</div>
  </div>
</div>
""",
                "afmt": """
<div class="customCard cardBack">
  <div class="targetWordContainerBack borderBottom">
    <span class="targetWord">{{Word}}</span>
    {{#PartOfSpeech}}<span class="partOfSpeech">{{PartOfSpeech}}</span>{{/PartOfSpeech}}
  </div>

  <div class="section borderBottom">
    <div class="header">Meaning:</div>
    <div class="indent">
      <div class="definitionsText">{{Meaning}}</div>
    </div>
  </div>

  <div class="section borderBottom">
    <div class="header">Example sentence:</div>
    <div class="indent">
      <div class="exampleSentenceWrapper">
        <span class="exampleSentence">{{ExampleSentence}}</span>
      </div>
      <div class="sentenceTranslation">{{hint:SentenceTranslation}}</div>
    </div>
  </div>
</div>
""",
            }
        ],
        css="""
:root {
  --max-width-card: 400px;
  --font-size-card: 18px;
  --font-size-targetWord: 26px;
  --font-size-header: 16px;

  --color-text-primary: #18191f;
  --color-nightMode-text-primary: #fbfafe;
  --color-card-background: #ffffff;
  --color-nightMode-card-background: #0b0716;
  --color-box-shadow: rgba(18, 62, 119, 0.1);
  --color-audio-button: #8369ed;
  --color-hint: #6b7280;
  --color-nightMode-hint: #9ca3af;
  --color-sentence-translation: #6b7280;
  --color-nightMode-sentence-translation: #9ca3af;
  --color-header: #9ca3af;
  --color-nightMode-header: rgba(255, 255, 255, 0.5);
  --color-divider: #e5e7eb;
  --color-nightMode-divider: #1f2937;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  margin: 0 !important;
  overflow-wrap: break-word;
}

.card {
  padding: 16px;
}

.customCard {
  margin: 0 auto;
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  background-color: var(--color-card-background);
  box-shadow: 1px 3px 10px var(--color-box-shadow);
  border-radius: 8px;
  min-height: 200px;
  max-width: var(--max-width-card);
  font-weight: 400;
  font-size: var(--font-size-card);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--color-text-primary);
}

.nightMode .customCard {
  color: var(--color-nightMode-text-primary);
  background-color: var(--color-nightMode-card-background);
}

.cardBack {
  justify-content: flex-start;
}

.targetWord {
  font-size: var(--font-size-targetWord);
  font-weight: 600;
}

.targetWordContainerFront {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 32px 16px;
}

.targetWordContainerBack {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  align-self: stretch;
  padding: 32px 16px 24px 16px;
  gap: 12px;
}

.partOfSpeech {
  font-size: 14px;
  font-weight: 400;
  color: var(--color-hint);
  font-style: italic;
}

.nightMode .partOfSpeech {
  color: var(--color-nightMode-hint);
}

.section {
  display: flex;
  flex-direction: column;
  align-self: stretch;
  gap: 6px;
  padding: 12px 16px 16px 16px;
}

.borderBottom {
  border-bottom: 1px solid var(--color-divider);
}

.nightMode .borderBottom {
  border-color: var(--color-nightMode-divider);
}

.header {
  color: var(--color-header);
  font-size: var(--font-size-header);
  font-weight: 400;
}

.nightMode .header {
  color: var(--color-nightMode-header);
}

.exampleSentenceWrapper {
  display: flex;
  align-items: center;
  position: relative;
  gap: 5px;
}

.exampleSentence {
  color: var(--color-hint);
}

.nightMode .exampleSentence {
  color: var(--color-nightMode-hint);
}

.sentenceTranslation {
  color: var(--color-sentence-translation);
  font-weight: 400;
  font-style: italic;
  padding-top: 6px;
  padding-left: 4px;
}

.nightMode .sentenceTranslation {
  color: var(--color-nightMode-sentence-translation);
}

.definitionsText {
  line-height: 1.5;
  font-size: 26px;
  font-weight: 700;
  color: var(--color-text-primary);
}

.nightMode .definitionsText {
  color: var(--color-nightMode-text-primary);
}

.indent {
  padding-left: 12px;
}
""",
    )


def write_anki_package(rows: list[CardRow], output_path: Path, deck_name: str) -> None:
    import genanki

    output_path.parent.mkdir(parents=True, exist_ok=True)
    deck = genanki.Deck(ANKI_MODEL_ID, deck_name)
    model = build_genanki_model()

    for due_position, row in enumerate(rows, start=1):
        note = genanki.Note(
            model=model,
            fields=[
                row.word,
                row.part_of_speech,
                row.meaning,
                row.example_sentence,
                row.sentence_translation,
            ],
            guid=genanki.guid_for(f"{row.normalized_lexeme_id}:{row.word}"),
            due=due_position,
        )
        deck.add_note(note)

    genanki.Package(deck).write_to_file(output_path)


@app.command()
def main(
    input_path: Path = typer.Option(
        DEFAULT_INPUT_PATH,
        "--input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to translation-pass JSONL output.",
    ),
    output_tsv_path: Path = typer.Option(
        DEFAULT_OUTPUT_TSV_PATH,
        "--output-tsv",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save an Anki import TSV.",
    ),
    output_apkg_path: Path = typer.Option(
        DEFAULT_OUTPUT_APKG_PATH,
        "--output-apkg",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Path to save an Anki package file.",
    ),
    deck_name: str = typer.Option(
        DEFAULT_DECK_NAME,
        "--deck-name",
        help="Deck name to use for the generated .apkg file.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Optional limit on the number of cards to export.",
    ),
    exclude_proper_nouns: bool = typer.Option(
        False,
        "--exclude-proper-nouns/--include-proper-nouns",
        help="Exclude rows tagged as proper_noun from the exported cards.",
    ),
) -> None:
    rows = [row for row in load_jsonl(input_path, TranslationPassRow) if isinstance(row, TranslationPassRow)]
    rows = filter_rows(rows, exclude_proper_nouns=exclude_proper_nouns)
    cards = build_card_rows(rows, limit)
    write_anki_tsv(cards, output_tsv_path)
    write_anki_package(cards, output_apkg_path, deck_name)

    typer.echo(f"input={input_path}")
    typer.echo(f"output_tsv={output_tsv_path}")
    typer.echo(f"output_apkg={output_apkg_path}")
    typer.echo(f"exclude_proper_nouns={exclude_proper_nouns}")
    typer.echo(f"card_count={len(cards)}")
    if cards:
        typer.echo(f"sample_word={cards[0].word}")
        typer.echo(f"sample_meaning={cards[0].meaning}")
        typer.echo(f"sample_example_sentence={cards[0].example_sentence}")


if __name__ == "__main__":
    app()
