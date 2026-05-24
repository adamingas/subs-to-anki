from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from subs2anki.db.models import Base, CacheBase


def resolve_db_path(db_path: Path) -> Path:
    return db_path.expanduser().resolve()


def create_engine_for_path(db_path: Path) -> Engine:
    return create_engine(f"sqlite:///{resolve_db_path(db_path)}", future=True)


def reset_database(db_path: Path) -> Path:
    resolved_db_path = resolve_db_path(db_path)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    if resolved_db_path.exists():
        resolved_db_path.unlink()
    return resolved_db_path


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def create_cache_schema(engine: Engine) -> None:
    CacheBase.metadata.create_all(engine)
