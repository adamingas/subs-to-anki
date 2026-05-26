from __future__ import annotations

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class CacheBase(DeclarativeBase):
    pass


@event.listens_for(Engine, "connect")
def enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Scope(Base):
    __tablename__ = "scopes"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    language: Mapped[str] = mapped_column(String(32), nullable=False, default="el")
    created_at: Mapped[str] = mapped_column(
        Text,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )

    subtitle_files: Mapped[list["SubtitleFile"]] = relationship(
        back_populates="scope",
        cascade="all, delete-orphan",
    )
    sentences: Mapped[list["Sentence"]] = relationship(
        back_populates="scope",
        cascade="all, delete-orphan",
    )
    lemmas: Mapped[list["Lemma"]] = relationship(
        back_populates="scope",
        cascade="all, delete-orphan",
    )
    original_forms: Mapped[list["OriginalForm"]] = relationship(
        back_populates="scope",
        cascade="all, delete-orphan",
    )
    reviewed_lexemes: Mapped[list["ReviewedLexeme"]] = relationship(
        back_populates="scope",
        cascade="all, delete-orphan",
    )
    normalized_lexemes: Mapped[list["NormalizedLexeme"]] = relationship(
        back_populates="scope",
        cascade="all, delete-orphan",
    )


class SubtitleFile(Base):
    __tablename__ = "subtitle_files"
    __table_args__ = (UniqueConstraint("scope_id", "path"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_id: Mapped[int] = mapped_column(ForeignKey("scopes.id"), index=True)
    path: Mapped[str] = mapped_column(String(1024), index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)

    scope: Mapped[Scope] = relationship(back_populates="subtitle_files")
    sentences: Mapped[list["Sentence"]] = relationship(
        back_populates="subtitle_file",
        cascade="all, delete-orphan",
        order_by="Sentence.sequence_number",
    )
    form_counts: Mapped[list["OriginalFormFileCount"]] = relationship(
        back_populates="subtitle_file",
        cascade="all, delete-orphan",
    )


class Sentence(Base):
    __tablename__ = "sentences"
    __table_args__ = (UniqueConstraint("subtitle_file_id", "sequence_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_id: Mapped[int] = mapped_column(ForeignKey("scopes.id"), index=True)
    subtitle_file_id: Mapped[int] = mapped_column(ForeignKey("subtitle_files.id"), index=True)
    sequence_number: Mapped[int] = mapped_column(index=True)
    text: Mapped[str] = mapped_column(Text)

    scope: Mapped[Scope] = relationship(back_populates="sentences")
    subtitle_file: Mapped[SubtitleFile] = relationship(back_populates="sentences")
    occurrences: Mapped[list["TokenOccurrence"]] = relationship(
        back_populates="sentence",
        cascade="all, delete-orphan",
        order_by="TokenOccurrence.token_position",
    )


class Lemma(Base):
    __tablename__ = "lemmas"
    __table_args__ = (UniqueConstraint("scope_id", "text"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_id: Mapped[int] = mapped_column(ForeignKey("scopes.id"), index=True)
    text: Mapped[str] = mapped_column(String(255), index=True)
    total_occurrences: Mapped[int] = mapped_column(default=0, nullable=False)

    scope: Mapped[Scope] = relationship(back_populates="lemmas")
    original_forms: Mapped[list["OriginalForm"]] = relationship(
        back_populates="lemma",
        cascade="all, delete-orphan",
        order_by="OriginalForm.text",
    )
    reviewed_lexemes: Mapped[list["ReviewedLexeme"]] = relationship(
        back_populates="source_lemma",
        cascade="all, delete-orphan",
    )


class OriginalForm(Base):
    __tablename__ = "original_forms"
    __table_args__ = (UniqueConstraint("scope_id", "text"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_id: Mapped[int] = mapped_column(ForeignKey("scopes.id"), index=True)
    text: Mapped[str] = mapped_column(String(255), index=True)
    lemma_id: Mapped[int] = mapped_column(ForeignKey("lemmas.id"), index=True)
    total_occurrences: Mapped[int] = mapped_column(default=0, nullable=False)

    scope: Mapped[Scope] = relationship(back_populates="original_forms")
    lemma: Mapped[Lemma] = relationship(back_populates="original_forms")
    occurrences: Mapped[list["TokenOccurrence"]] = relationship(
        back_populates="original_form",
        cascade="all, delete-orphan",
    )
    file_counts: Mapped[list["OriginalFormFileCount"]] = relationship(
        back_populates="original_form",
        cascade="all, delete-orphan",
    )
    reviewed_lexeme_links: Mapped[list["ReviewedLexemeOriginalForm"]] = relationship(
        back_populates="original_form",
        cascade="all, delete-orphan",
    )


class OriginalFormFileCount(Base):
    __tablename__ = "original_form_file_counts"
    __table_args__ = (UniqueConstraint("original_form_id", "subtitle_file_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_id: Mapped[int] = mapped_column(ForeignKey("scopes.id"), index=True)
    original_form_id: Mapped[int] = mapped_column(ForeignKey("original_forms.id"), index=True)
    subtitle_file_id: Mapped[int] = mapped_column(ForeignKey("subtitle_files.id"), index=True)
    occurrence_count: Mapped[int] = mapped_column(default=0, nullable=False)

    original_form: Mapped[OriginalForm] = relationship(back_populates="file_counts")
    subtitle_file: Mapped[SubtitleFile] = relationship(back_populates="form_counts")


class TokenOccurrence(Base):
    __tablename__ = "token_occurrences"
    __table_args__ = (UniqueConstraint("sentence_id", "token_position"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_id: Mapped[int] = mapped_column(ForeignKey("scopes.id"), index=True)
    sentence_id: Mapped[int] = mapped_column(ForeignKey("sentences.id"), index=True)
    original_form_id: Mapped[int] = mapped_column(ForeignKey("original_forms.id"), index=True)
    token_position: Mapped[int] = mapped_column(index=True)

    sentence: Mapped[Sentence] = relationship(back_populates="occurrences")
    original_form: Mapped[OriginalForm] = relationship(back_populates="occurrences")


class ReviewedLexeme(Base):
    __tablename__ = "reviewed_lexemes"

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_id: Mapped[int] = mapped_column(ForeignKey("scopes.id"), index=True)
    source_lemma_id: Mapped[int] = mapped_column(ForeignKey("lemmas.id"), index=True)
    normalized_form: Mapped[str] = mapped_column(String(255), index=True)
    word_class: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[str] = mapped_column(
        Text,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at: Mapped[str] = mapped_column(
        Text,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )

    scope: Mapped[Scope] = relationship(back_populates="reviewed_lexemes")
    source_lemma: Mapped[Lemma] = relationship(back_populates="reviewed_lexemes")
    reviewed_original_forms: Mapped[list["ReviewedLexemeOriginalForm"]] = relationship(
        back_populates="reviewed_lexeme",
        cascade="all, delete-orphan",
    )
    normalized_lexeme_links: Mapped[list["NormalizedLexemeReviewedLexeme"]] = relationship(
        back_populates="reviewed_lexeme",
        cascade="all, delete-orphan",
    )


class ReviewedLexemeOriginalForm(Base):
    __tablename__ = "reviewed_lexeme_original_forms"

    reviewed_lexeme_id: Mapped[int] = mapped_column(
        ForeignKey("reviewed_lexemes.id"),
        primary_key=True,
    )
    original_form_id: Mapped[int] = mapped_column(
        ForeignKey("original_forms.id"),
        primary_key=True,
    )

    reviewed_lexeme: Mapped[ReviewedLexeme] = relationship(back_populates="reviewed_original_forms")
    original_form: Mapped[OriginalForm] = relationship(back_populates="reviewed_lexeme_links")


class NormalizedLexeme(Base):
    __tablename__ = "normalized_lexemes"
    __table_args__ = (UniqueConstraint("scope_id", "normalized_form"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_id: Mapped[int] = mapped_column(ForeignKey("scopes.id"), index=True)
    normalized_form: Mapped[str] = mapped_column(String(255), index=True)
    word_class: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[str] = mapped_column(
        Text,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at: Mapped[str] = mapped_column(
        Text,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )

    scope: Mapped[Scope] = relationship(back_populates="normalized_lexemes")
    reviewed_lexeme_links: Mapped[list["NormalizedLexemeReviewedLexeme"]] = relationship(
        back_populates="normalized_lexeme",
        cascade="all, delete-orphan",
    )
    ordering: Mapped["NormalizedLexemeOrdering | None"] = relationship(
        back_populates="normalized_lexeme",
        cascade="all, delete-orphan",
        uselist=False,
    )


class NormalizedLexemeReviewedLexeme(Base):
    __tablename__ = "normalized_lexeme_reviewed_lexemes"

    normalized_lexeme_id: Mapped[int] = mapped_column(
        ForeignKey("normalized_lexemes.id"),
        primary_key=True,
    )
    reviewed_lexeme_id: Mapped[int] = mapped_column(
        ForeignKey("reviewed_lexemes.id"),
        primary_key=True,
    )

    normalized_lexeme: Mapped[NormalizedLexeme] = relationship(back_populates="reviewed_lexeme_links")
    reviewed_lexeme: Mapped[ReviewedLexeme] = relationship(back_populates="normalized_lexeme_links")


class NormalizedLexemeOrdering(Base):
    __tablename__ = "normalized_lexeme_ordering"
    __table_args__ = (UniqueConstraint("scope_id", "order_index"),)

    normalized_lexeme_id: Mapped[int] = mapped_column(
        ForeignKey("normalized_lexemes.id"),
        primary_key=True,
    )
    scope_id: Mapped[int] = mapped_column(ForeignKey("scopes.id"), index=True)
    order_index: Mapped[int] = mapped_column(index=True)
    example_sentence_id: Mapped[int | None] = mapped_column(ForeignKey("sentences.id"), nullable=True, index=True)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False, default="closure_seed")
    dependency_normalized_forms_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    dependency_violations_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    external_unknown_forms_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    non_target_unknown_count: Mapped[int] = mapped_column(nullable=False, default=0)
    total_unknown_count: Mapped[int] = mapped_column(nullable=False, default=0)
    sentence_token_count: Mapped[int] = mapped_column(nullable=False, default=0)
    candidate_sentence_count: Mapped[int] = mapped_column(nullable=False, default=0)
    cycle_break: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[str] = mapped_column(
        Text,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at: Mapped[str] = mapped_column(
        Text,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )

    normalized_lexeme: Mapped[NormalizedLexeme] = relationship(back_populates="ordering")
    example_sentence: Mapped[Sentence | None] = relationship()


class PromptCacheEntry(CacheBase):
    __tablename__ = "prompt_cache"

    cache_key: Mapped[str] = mapped_column(Text, primary_key=True)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt_hash: Mapped[str] = mapped_column(Text, nullable=False)
    user_prompt_hash: Mapped[str] = mapped_column(Text, nullable=False)
    parameters_hash: Mapped[str] = mapped_column(Text, nullable=False)
    schema_hash: Mapped[str] = mapped_column(Text, nullable=False, default="")
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False)
    input_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    response_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    elapsed_ms: Mapped[float | None] = mapped_column(nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(
        Text,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
    updated_at: Mapped[str] = mapped_column(
        Text,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )
