"""
logger_setup.py

Centralized logging configuration for NeuroGesture AI. Every module in the
project should call `get_logger(__name__)` rather than instantiating its own
handlers, so log format, rotation, and destinations stay consistent.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(log_dir: str = "logs", log_level: str = "INFO", max_log_files: int = 10) -> None:
    """
    Configure the root 'neurogesture' logger once per process.

    Adds:
      - A console handler (INFO+ by default, human-readable).
      - A rotating file handler that writes a fresh timestamped log per run,
        keeping at most `max_log_files` historical files.
    """
    global _initialized
    if _initialized:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("neurogesture")
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"neurogesture_{timestamp}.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=max_log_files, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _prune_old_logs(log_path, max_log_files)

    _initialized = True
    root_logger.info("Logging initialized. Writing to %s", log_file)


def _prune_old_logs(log_path: Path, max_log_files: int) -> None:
    """Delete oldest log files beyond the retention limit."""
    log_files = sorted(log_path.glob("neurogesture_*.log"), key=lambda p: p.stat().st_mtime)
    excess = len(log_files) - max_log_files
    for old_file in log_files[:max(0, excess)]:
        try:
            old_file.unlink()
        except OSError:
            pass


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, e.g. get_logger(__name__)."""
    if not _initialized:
        setup_logging()
    return logging.getLogger(f"neurogesture.{name}")
