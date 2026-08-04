"""SQLAlchemy database setup for persistent application data."""

from __future__ import annotations

from functools import lru_cache
from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=8)
def get_engine(database_url: str) -> Engine:
    connect_args = (
        {"check_same_thread": False}
        if database_url.startswith("sqlite")
        else {}
    )
    return create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=8)
def get_session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(database_url),
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
    )


def init_database() -> None:
    # Import model modules before creating metadata.
    from app.models import chat as _chat_models  # noqa: F401

    settings = get_settings()
    Base.metadata.create_all(bind=get_engine(settings.database_url))


def get_db() -> Generator[Session, None, None]:
    settings = get_settings()
    database = get_session_factory(settings.database_url)()
    try:
        yield database
    except Exception:
        database.rollback()
        raise
    finally:
        database.close()
