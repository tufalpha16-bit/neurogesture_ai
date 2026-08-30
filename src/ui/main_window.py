"""
main_window.py

Top-level NeuroGesture AI window.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow,
    QStatusBar,
    QTabWidget,
)

from src.core.logger_setup import get_logger
from src.ui.dashboard_tab import DashboardTab
from src.ui.recorder_widget import RecorderWidget
from src.ui.video_widget import VideoCaptureThread
from src.utils.config_loader import AppConfig


logger = get_logger(__name__)


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #121417;
    color: #E6E6E6;
    font-family: 'Segoe UI', sans-serif;
}

QTabWidget::pane {
    border: none;
}

QTabBar::tab {
    background-color: #1B1E23;
    color: #9AA0A8;
    padding: 8px 18px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #202429;
    color: #4FD1C5;
    font-weight: 600;
}

QFrame#videoFrame {
    background-color: #1B1E23;
    border: 1px solid #2A2E35;
    border-radius: 10px;
}

QFrame#statCard {
    background-color: #1B1E23;
    border: 1px solid #2A2E35;
    border-radius: 8px;
    padding: 8px;
}

QLabel[class="statValue"] {
    font-size: 18px;
    font-weight: 600;
    color: #4FD1C5;
}

QLabel[class="statLabel"] {
    font-size: 11px;
    color: #9AA0A8;
}

QPushButton {
    background-color: #202429;
    color: #E6E6E6;
    border: 1px solid #2A2E35;
    border-radius: 6px;
    padding: 8px 14px;
}

QPushButton:hover {
    background-color: #262B32;
    border-color: #4FD1C5;
}

QPushButton:disabled {
    color: #5A5F66;
}

QLineEdit, QListWidget {
    background-color: #1B1E23;
    border: 1px solid #2A2E35;
    border-radius: 6px;
    padding: 6px;
    color: #E6E6E6;
}

QProgressBar {
    background-color: #1B1E23;
    border: 1px solid #2A2E35;
    border-radius: 6px;
    text-align: center;
    color: #E6E6E6;
}

QProgressBar::chunk {
    background-color: #4FD1C5;
    border-radius: 6px;
}

QStatusBar {
    background-color: #1B1E23;
    color: #9AA0A8;
}
"""


class MainWindow(QMainWindow):
    """NeuroGesture AI main application window."""

    def __init__(
        self,
        config: AppConfig,
    ) -> None:

        super().__init__()

        self._config = config

        self.setWindowTitle(
            config.ui.window_title
        )

        self.resize(
            config.ui.window_width,
            config.ui.window_height,
        )

        self.setStyleSheet(
            DARK_STYLESHEET
        )

        self._capture_thread = None

        self._build_ui()
        self._start_capture()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:

        self.tabs = QTabWidget()

        self.setCentralWidget(
            self.tabs
        )

        self.dashboard_tab = (
            DashboardTab()
        )

        self.recorder_tab = (
            RecorderWidget(
                self._config
            )
        )

        self.tabs.addTab(
            self.dashboard_tab,
            "Live Dashboard",
        )

        self.tabs.addTab(
            self.recorder_tab,
            "Gesture Recorder",
        )

        self.setStatusBar(
            QStatusBar()
        )

        self.statusBar().showMessage(
            "Initializing..."
        )

    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #

    def _start_capture(self) -> None:

        self._capture_thread = (
            VideoCaptureThread(
                self._config
            )
        )

        self._capture_thread.frame_ready.connect(
            self.dashboard_tab.update_frame
        )

        self._capture_thread.frame_ready.connect(
            self.recorder_tab.update_frame
        )

        self._capture_thread.detection_ready.connect(
            self.dashboard_tab.update_detection
        )

        self._capture_thread.detection_ready.connect(
            self.recorder_tab.on_detection
        )

        self._capture_thread.fps_updated.connect(
            self.dashboard_tab.update_fps
        )

        self._capture_thread.status_changed.connect(
            self._on_status_changed
        )

        self._capture_thread.error_occurred.connect(
            self._on_error
        )

        self._capture_thread.start()

    # ------------------------------------------------------------------ #
    # Status
    # ------------------------------------------------------------------ #

    def _on_status_changed(
        self,
        status: str,
    ) -> None:

        self.dashboard_tab.update_status(
            status
        )

        self.statusBar().showMessage(
            f"Camera: {status}"
        )

        logger.info(
            "Camera status changed to '%s'.",
            status,
        )

    # ------------------------------------------------------------------ #
    # Error
    # ------------------------------------------------------------------ #

    def _on_error(
        self,
        message: str,
    ) -> None:

        self.dashboard_tab.show_error(
            message
        )

        self.statusBar().showMessage(
            f"Error: {message}"
        )

        logger.error(
            "Video thread error: %s",
            message,
        )

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #

    def closeEvent(
        self,
        event,
    ) -> None:

        logger.info(
            "Shutting down NeuroGesture AI..."
        )

        # Stop camera first.
        if self._capture_thread is not None:

            self._capture_thread.stop()

        # Then stop TensorFlow worker.
        self.dashboard_tab.shutdown()

        event.accept()