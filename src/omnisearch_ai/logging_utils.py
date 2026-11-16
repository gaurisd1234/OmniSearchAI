"""Shared standard-library logging setup."""

from __future__ import annotations

import logging

from omnisearch_ai.config import load_config


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure and return the project logger using a consistent format."""
    selected_level = (level or load_config().log_level).upper()
    numeric_level = getattr(logging, selected_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {selected_level}")
    logger = logging.getLogger("omnisearch_ai")
    logger.setLevel(numeric_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger
