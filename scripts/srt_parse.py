#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import typer

from subs2anki.db.importer import import_subtitles

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEFAULT_DB_PATH = Path("data/subs2anki.sqlite3")
DEFAULT_LANGUAGE = "el"


@app.command()
def main(
    files: list[Path] = typer.Argument(..., exists=True, readable=True, help="Subtitle files to import."),
    db_path: Path = typer.Option(
        DEFAULT_DB_PATH,
        "--db",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="SQLite database output path.",
    ),
    language: str = typer.Option(
        DEFAULT_LANGUAGE,
        "--language",
        help="Language code passed to the lemmatizer.",
    ),
) -> None:
    resolved_db_path, stats = import_subtitles(files=files, db_path=db_path, language=language)
    typer.echo(f"db={resolved_db_path}")
    typer.echo(
        " ".join(
            [
                f"files={stats.files}",
                f"sentences={stats.sentences}",
                f"lemmas={stats.lemmas}",
                f"original_forms={stats.original_forms}",
                f"occurrences={stats.occurrences}",
            ]
        )
    )


if __name__ == "__main__":
    app()
