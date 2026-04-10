#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "orjson>=3.10.16",
#   "simplemma>=1.1.2",
#   "srt>=3.5.3",
#   "stanza>=1.8.2",
#   "typer>=0.16.0",
#   "wn>=0.13.0",
# ]
# ///

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import orjson
import simplemma  # type: ignore
import srt  # type: ignore
import stanza  # type: ignore
import typer
import wn  # type: ignore

app = typer.Typer(add_completion=False, no_args_is_help=True)

HTML_TAG_RE = re.compile(r"<[^>]+>")
TIMECODE_LINE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}"
)
DIALOGUE_DASH_RE = re.compile(r"(^|\n)\s*[-–—]+\s*", re.UNICODE)
LEADING_APOSTROPHE_RE = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=\n))['’](?=\w)", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
OMW_COLLECTION_URL = "https://github.com/omwn/omw-data/releases/download/v1.4/omw-1.4.tar.xz"


def clean_subtitle_text(text: str) -> str:
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = DIALOGUE_DASH_RE.sub(r"\1", text)
    text = LEADING_APOSTROPHE_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_text_blocks(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8-sig")
    try:
        return [clean_subtitle_text(subtitle.content) for subtitle in srt.parse(content)]
    except srt.SRTParseError:
        blocks: list[str] = []
        pending: list[str] = []

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                if pending:
                    blocks.append(clean_subtitle_text(" ".join(pending)))
                    pending = []
                continue
            if line.isdigit() or TIMECODE_LINE_RE.match(line):
                continue
            pending.append(line)

        if pending:
            blocks.append(clean_subtitle_text(" ".join(pending)))

        return [block for block in blocks if block]


def validate_inputs(paths: Sequence[Path]) -> list[Path]:
    resolved_paths = [path.expanduser() for path in paths]
    missing = [str(path) for path in resolved_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input SRT files not found: {', '.join(missing)}")
    return resolved_paths


def build_nlp(language: str) -> Any:
    stanza.download(language, processors="tokenize", verbose=False)
    return stanza.Pipeline(
        language,
        processors="tokenize",
        tokenize_no_ssplit=False,
        verbose=False,
    )


def normalize_lemma(lemma: str) -> str:
    normalized = unicodedata.normalize("NFC", lemma).lower().strip().lstrip("'’")
    if normalized.endswith("σ"):
        normalized = normalized[:-1] + "ς"
    return normalized


def normalize_sentence(text: str) -> str:
    return clean_subtitle_text(text)


def extract_sentence_tokens(text: str) -> list[str]:
    return [match.group(0) for match in WORD_RE.finditer(text)]


class EnglishLemmaLookup:
    def __init__(self, target_lexicon: str = "oewn:2024") -> None:
        wn.download(OMW_COLLECTION_URL)
        wn.download(target_lexicon)
        self._target_lexicon = target_lexicon
        self._cache: dict[str, list[str]] = {}

    def lookup(self, lemma: str) -> list[str]:
        cached = self._cache.get(lemma)
        if cached is not None:
            return cached

        english_lemmas: set[str] = set()
        for word in wn.words(lemma, lang="el"):
            translations_by_sense = word.translate(lexicon=self._target_lexicon)
            for translated_words in translations_by_sense.values():
                for translated_word in translated_words:
                    translated = normalize_lemma(translated_word.lemma())
                    if translated:
                        english_lemmas.add(translated)

        result = sorted(english_lemmas)
        self._cache[lemma] = result
        return result


def collect_lemma_rows(
    paths: Sequence[Path],
    language: str,
    english_lookup: EnglishLemmaLookup | None = None,
) -> list[dict[str, Any]]:
    nlp = build_nlp(language)
    entries: dict[str, dict[str, Any]] = {}

    for path in paths:
        document_text = "\n".join(extract_text_blocks(path))
        if not document_text:
            continue

        document = nlp(document_text)
        for sentence in document.sentences:
            sentence_text = normalize_sentence(sentence.text)
            if not sentence_text:
                continue

            for token_text in extract_sentence_tokens(sentence_text):
                if not any(character.isalpha() for character in token_text):
                    continue

                lemmatised_form = normalize_lemma(simplemma.lemmatize(token_text, lang=language))
                if not lemmatised_form:
                    continue

                entry = entries.setdefault(
                    lemmatised_form,
                    {
                        "count": 0,
                        "original_forms": [],
                        "_original_forms_seen": set(),
                        "sentences": [],
                        "_sentence_keys": set(),
                    },
                )
                entry["count"] += 1

                if token_text not in entry["_original_forms_seen"]:
                    entry["_original_forms_seen"].add(token_text)
                    entry["original_forms"].append(token_text)

                sentence_key = (str(path), sentence_text)
                if sentence_key not in entry["_sentence_keys"]:
                    entry["_sentence_keys"].add(sentence_key)
                    entry["sentences"].append(
                        {
                            "sentence_text": sentence_text,
                            "source_srt": str(path),
                        }
                    )

    rows: list[dict[str, Any]] = []
    for lemmatised_form, entry in sorted(
        entries.items(),
        key=lambda item: (-int(item[1]["count"]), item[0]),
    ):
        row = {
            "lemmatised_form": lemmatised_form,
            "count": int(entry["count"]),
            "original_forms": list(entry["original_forms"]),
            "sentences": list(entry["sentences"]),
        }
        if english_lookup is not None:
            row["english_lemmas"] = english_lookup.lookup(lemmatised_form)
        rows.append(row)

    return rows


def write_jsonl(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


def run(
    files: Sequence[Path],
    output: Path,
    language: str,
    english_lemmas: bool,
) -> Path:
    input_paths = validate_inputs(files)
    english_lookup = EnglishLemmaLookup() if english_lemmas else None
    rows = collect_lemma_rows(
        input_paths,
        language=language,
        english_lookup=english_lookup,
    )
    output_path = output.expanduser()
    write_jsonl(rows, output_path)
    return output_path


@app.command()
def main(
    files: list[Path] = typer.Argument(..., help="Input .srt files"),
    output: Path = typer.Option(
        Path("data/lemma_entries.jsonl"),
        "--output",
        "-o",
        help="Output JSONL path",
    ),
    language: str = typer.Option(
        "el",
        "--language",
        help="Stanza language code for tokenization and lemmatization",
    ),
    english_lemmas: bool = typer.Option(
        False,
        "--english-lemmas",
        help="Add English lemma equivalents using WordNet lexicons",
    ),
) -> None:
    try:
        output_path = run(
            files,
            output=output,
            language=language,
            english_lemmas=english_lemmas,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(output_path)


if __name__ == "__main__":
    app()
