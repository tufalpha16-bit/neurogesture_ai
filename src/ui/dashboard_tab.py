"""
dashboard_tab.py

Smooth live dashboard.

TensorFlow prediction runs in GestureWorker rather than inside the
camera/UI callback.

Computer actions are handled by ActionController in background threads.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QThread,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.core.action_controller import ActionController
from src.core.gesture_worker import GestureWorker
from src.core.hand_detector import DetectionResult
from src.core.logger_setup import get_logger


logger = get_logger(__name__)


class StatCard(QFrame):
    """Small card showing one live metric."""

    def __init__(
        self,
        label: str,
        initial_value: str = "--",
    ) -> None:

        super().__init__()

        self.setObjectName(
            "statCard"
        )

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            10,
            8,
            10,
            8,
        )

        layout.setSpacing(2)

        self.value_label = QLabel(
            initial_value
        )

        self.value_label.setProperty(
            "class",
            "statValue",
        )

        caption_label = QLabel(
            label
        )

        caption_label.setProperty(
            "class",
            "statLabel",
        )

        layout.addWidget(
            self.value_label
        )

        layout.addWidget(
            caption_label
        )

    def set_value(
        self,
        value: str,
    ) -> None:

        self.value_label.setText(
            value
        )


class DashboardTab(QWidget):
    """Smooth webcam dashboard."""

    request_prediction = Signal(object)
    request_reset = Signal()

    def __init__(self) -> None:

        super().__init__()

        # -------------------------------------------------------------- #
        # State
        # -------------------------------------------------------------- #

        self._last_gesture: str | None = None
        self._last_confidence = 0.0

        self._last_executed_gesture: str | None = None

        # -------------------------------------------------------------- #
        # Action controller
        # -------------------------------------------------------------- #

        self._action_controller = (
            ActionController(
                cooldown_seconds=1.5
            )
        )

        # -------------------------------------------------------------- #
        # Gesture worker thread
        # -------------------------------------------------------------- #

        self._gesture_thread = QThread(
            self
        )

        self._gesture_worker = (
            GestureWorker()
        )

        self._gesture_worker.moveToThread(
            self._gesture_thread
        )

        self.request_prediction.connect(
            self._gesture_worker.process_frame,
            Qt.ConnectionType.QueuedConnection,
        )

        self.request_reset.connect(
            self._gesture_worker.reset,
            Qt.ConnectionType.QueuedConnection,
        )

        self._gesture_thread.started.connect(
            self._gesture_worker.initialize
        )

        self._gesture_worker.prediction_ready.connect(
            self._on_prediction_ready,
            Qt.ConnectionType.QueuedConnection,
        )

        self._gesture_worker.prediction_error.connect(
            self._on_prediction_error,
            Qt.ConnectionType.QueuedConnection,
        )

        self._gesture_thread.start()

        # -------------------------------------------------------------- #
        # UI
        # -------------------------------------------------------------- #

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:

        body_layout = QHBoxLayout(
            self
        )

        body_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        body_layout.setSpacing(12)

        # -------------------------------------------------------------- #
        # Video
        # -------------------------------------------------------------- #

        video_frame = QFrame()

        video_frame.setObjectName(
            "videoFrame"
        )

        video_layout = QVBoxLayout(
            video_frame
        )

        self.video_label = QLabel(
            "Waiting for camera frames..."
        )

        self.video_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.video_label.setMinimumSize(
            800,
            600,
        )

        video_layout.addWidget(
            self.video_label
        )

        body_layout.addWidget(
            video_frame,
            stretch=3,
        )

        # -------------------------------------------------------------- #
        # Sidebar
        # -------------------------------------------------------------- #

        sidebar_layout = QVBoxLayout()

        sidebar_layout.setSpacing(10)

        self.fps_card = StatCard(
            "FPS",
            "--",
        )

        self.camera_status_card = StatCard(
            "Camera Status",
            "Connecting",
        )

        self.ai_status_card = StatCard(
            "AI Status",
            "Watching",
        )

        self.hands_card = StatCard(
            "Hands Detected",
            "0",
        )

        sidebar_layout.addWidget(
            self.fps_card
        )

        sidebar_layout.addWidget(
            self.camera_status_card
        )

        sidebar_layout.addWidget(
            self.ai_status_card
        )

        sidebar_layout.addWidget(
            self.hands_card
        )

        sidebar_layout.addStretch(1)

        body_layout.addLayout(
            sidebar_layout,
            stretch=1,
        )

    # ------------------------------------------------------------------ #
    # Frame
    # ------------------------------------------------------------------ #

    @Slot(QImage)
    def update_frame(
        self,
        image: QImage,
    ) -> None:

        if image.isNull():
            return

        pixmap = QPixmap.fromImage(
            image
        )

        if pixmap.isNull():
            return

        target_size = (
            self.video_label.size()
        )

        if (
            target_size.width() <= 0
            or target_size.height() <= 0
        ):
            return

        pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

        self.video_label.setPixmap(
            pixmap
        )

        self.video_label.setText("")

    # ------------------------------------------------------------------ #
    # FPS
    # ------------------------------------------------------------------ #

    @Slot(float)
    def update_fps(
        self,
        fps: float,
    ) -> None:

        self.fps_card.set_value(
            f"{fps:.1f}"
        )

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #

    @Slot(object)
    def update_detection(
        self,
        detection: DetectionResult,
    ) -> None:
        """
        IMPORTANT:
        This method does NOT run TensorFlow.

        It only prepares the landmark vector and sends it to the
        background GestureWorker.
        """

        self.hands_card.set_value(
            str(detection.num_hands)
        )

        # -------------------------------------------------------------- #
        # No hand
        # -------------------------------------------------------------- #

        if detection.num_hands == 0:

            self.ai_status_card.set_value(
                "Watching"
            )

            self.request_reset.emit()

            self._last_gesture = None
            self._last_confidence = 0.0

            # Allow the next gesture to execute.
            self._last_executed_gesture = None

            return

        # -------------------------------------------------------------- #
        # Landmark vector
        # -------------------------------------------------------------- #

        try:

            vector = (
                detection.flattened_vector(
                    max_hands=2
                )
            )

        except Exception as exc:

            logger.exception(
                "Could not create landmark vector: %s",
                exc,
            )

            self.ai_status_card.set_value(
                "Tracking"
            )

            return

        # -------------------------------------------------------------- #
        # Send to background worker
        # -------------------------------------------------------------- #

        self.request_prediction.emit(
            vector
        )

        # Don't overwrite an already recognized gesture.
        if self._last_gesture is None:

            self.ai_status_card.set_value(
                "Tracking"
            )

    # ------------------------------------------------------------------ #
    # Prediction result
    # ------------------------------------------------------------------ #

    @Slot(str, float)
    def _on_prediction_ready(
        self,
        gesture: str,
        confidence: float,
    ) -> None:

        self._last_gesture = gesture
        self._last_confidence = confidence

        display_name = (
            gesture
            .replace("_", " ")
            .title()
        )

        self.ai_status_card.set_value(
            f"{display_name}\n"
            f"{confidence * 100:.0f}%"
        )

        logger.info(
            "DASHBOARD GESTURE: %s (%.1f%%)",
            gesture,
            confidence * 100,
        )

        # -------------------------------------------------------------- #
        # Execute only once until hand is removed.
        # -------------------------------------------------------------- #

        if (
            gesture
            == self._last_executed_gesture
        ):
            return

        accepted = (
            self._action_controller.execute(
                gesture
            )
        )

        if accepted:

            self._last_executed_gesture = (
                gesture
            )

            logger.info(
                "ACTION ACCEPTED: %s",
                gesture,
            )

    # ------------------------------------------------------------------ #
    # Prediction error
    # ------------------------------------------------------------------ #

    @Slot(str)
    def _on_prediction_error(
        self,
        message: str,
    ) -> None:

        logger.error(
            "Gesture worker error: %s",
            message,
        )

    # ------------------------------------------------------------------ #
    # Camera status
    # ------------------------------------------------------------------ #

    @Slot(str)
    def update_status(
        self,
        status: str,
    ) -> None:

        self.camera_status_card.set_value(
            status
        )

    # ------------------------------------------------------------------ #
    # Camera error
    # ------------------------------------------------------------------ #

    @Slot(str)
    def show_error(
        self,
        message: str,
    ) -> None:

        self.video_label.setText(
            f"Camera error:\n{message}"
        )

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:
        """Stop the gesture worker cleanly."""

        if (
            hasattr(self, "_gesture_thread")
            and self._gesture_thread.isRunning()
        ):

            self._gesture_thread.quit()

            if not self._gesture_thread.wait(
                3000
            ):

                logger.warning(
                    "Gesture worker did not stop within timeout."
                )