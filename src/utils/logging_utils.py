"""Small logging helper used across scripts."""
from __future__ import annotations

import logging
import os
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with a single stream handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = (level or os.getenv("CFB_TIX_LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
