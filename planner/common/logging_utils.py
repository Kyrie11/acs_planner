from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO, use_stdout: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    handler: logging.Handler = logging.StreamHandler(sys.stdout if use_stdout else sys.stderr)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
