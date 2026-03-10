"""
utils/logger.py
---------------
Centralised logging factory for the Adversarial Robustness Engine.
All modules obtain their logger through ``get_logger(__name__)`` so that
log level, format, and handlers are configured in one place.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(name)-40s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_file: bool = False,
) -> None:
    """
    Configure root logger.  Call once at application startup.

    Parameters
    ----------
    level:       Logging level string (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    log_dir:     Directory for rotating log files.  Created if absent.
    log_to_file: Whether to add a FileHandler in addition to StreamHandler.
    """
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional file handler
    if log_to_file and log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"are_{ts}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Quieten noisy third-party loggers
    for noisy in ("PIL", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Ensures root is at least INFO-configured."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
