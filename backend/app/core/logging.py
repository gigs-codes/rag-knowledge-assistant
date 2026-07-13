"""
Logging setup.

Why: print() statements disappear once you deploy behind a process manager
or container — you need structured, leveled logs you can actually filter
in production (grep by level, ship to a log aggregator later). This gives
every module a named logger via `get_logger(__name__)` so log lines are
traceable back to their source file.
"""
import logging
import sys

from app.core.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
