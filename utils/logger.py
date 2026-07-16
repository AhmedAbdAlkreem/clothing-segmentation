
"""Centralized logging setup — replaces scattered print() calls."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: Path = None) -> logging.Logger:
    """Returns a configured logger that writes to console and, optionally,
    to a file under log_dir. Guards against duplicate handlers if called
    multiple times for the same logger name (common issue in notebooks).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # already configured

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
