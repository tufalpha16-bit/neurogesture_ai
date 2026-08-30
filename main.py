"""
main.py

Entry point for NeuroGesture AI. Run this file to launch the desktop
application:

    python main.py

Requires an activated virtual environment with dependencies installed
from requirements.txt. See README.md for full setup instructions.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.core.logger_setup import get_logger, setup_logging
from src.ui.main_window import MainWindow
from src.utils.config_loader import get_config


def main() -> int:
    config = get_config("config/config.yaml")
    setup_logging(
        log_dir=config.logging.log_dir,
        log_level=config.logging.log_level,
        max_log_files=config.logging.max_log_files,
    )
    logger = get_logger("main")
    logger.info("Starting NeuroGesture AI...")

    app = QApplication(sys.argv)
    app.setApplicationName("NeuroGesture AI")

    window = MainWindow(config)
    window.show()

    exit_code = app.exec()
    logger.info("NeuroGesture AI exited with code %d.", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
