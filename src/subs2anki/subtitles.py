from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import srt  # type: ignore
import stanza  # type: ignore

HTML_TAG_RE = re.compile(r"<[^>]+>")
TIMECODE_LINE_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}"
)
DIALOGUE_DASH_RE = re.compile(r"(^|\n)\s*[-–—]+\s*", re.UNICODE)
LEADING_APOSTROPHE_RE = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=\n))['’](?=\w)", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


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
        blocks = [clean_subtitle_text(subtitle.content) for subtitle in srt.parse(content)]
    except srt.SRTParseError:
        blocks = []
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


def validate_input_paths(paths: Sequence[Path]) -> list[Path]:
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


def extract_document_sentences(path: Path, nlp: Any) -> list[str]:
    document_text = "\n".join(extract_text_blocks(path))
    if not document_text:
        return []

    document = nlp(document_text)
    sentences: list[str] = []
    for sentence in document.sentences:
        sentence_text = normalize_sentence(sentence.text)
        if sentence_text:
            sentences.append(sentence_text)
    return sentences
