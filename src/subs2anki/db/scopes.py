from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from subs2anki.db.core import create_engine_for_path, create_schema
from subs2anki.db.models import Scope


def create_scope(session: Session, name: str, language: str) -> Scope:
    existing = session.scalar(select(Scope).where(Scope.name == name))
    if existing is not None:
        raise RuntimeError(f"Scope {name!r} already exists.")
    scope = Scope(name=name, language=language)
    session.add(scope)
    session.flush()
    return scope


def list_scopes(db_path: Path) -> list[Scope]:
    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
        with Session(engine) as session:
            return list(session.scalars(select(Scope).order_by(Scope.created_at, Scope.id)))
    finally:
        engine.dispose()


def resolve_scope_id(db_path: Path, scope_name: str | None = None) -> int:
    engine = create_engine_for_path(db_path)
    try:
        create_schema(engine)
        with Session(engine) as session:
            if scope_name is not None:
                scope = session.scalar(select(Scope).where(Scope.name == scope_name))
                if scope is None:
                    raise RuntimeError(f"Unknown scope {scope_name!r}.")
                return scope.id

            count = session.scalar(select(func.count()).select_from(Scope)) or 0
            if count == 1:
                only_scope_id = session.scalar(select(Scope.id))
                if only_scope_id is None:
                    raise RuntimeError("Could not resolve the only scope in the database.")
                return int(only_scope_id)
            if count == 0:
                raise RuntimeError("No scopes found in the database.")
            scope_names = list(session.scalars(select(Scope.name).order_by(Scope.created_at, Scope.id)))
            raise RuntimeError(
                "Multiple scopes found. Provide --scope. Available scopes: "
                + ", ".join(scope_names)
            )
    finally:
        engine.dispose()
