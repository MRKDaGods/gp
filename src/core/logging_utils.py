"""Structured logging setup using loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    rotation: str = "10 MB",
) -> None:
    """Configure loguru logger for the pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output. None = console only.
        rotation: Log file rotation size.
    """
    # Remove default handler
    logger.remove()

    # Console handler with color
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    # File handler (if requested)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format=fmt,
            level=level,
            rotation=rotation,
            retention="7 days",
        )

    logger.info(f"Logging initialized at {level} level")
