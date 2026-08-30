"""
video_widget.py

A QThread-based video capture worker that runs the camera read + MediaPipe
detection loop off the GUI thread, emitting finished frames as Qt signals.
Keeping this off the main thread is what keeps the PySide6 UI responsive
even at 30 FPS with hand-landmark processing in the loop.

Status lifecycle (emitted via status_changed):
    "Connecting" -> while the camera device and hand-landmark model are
                    being opened/initialized
    "Connected"  -> camera opened AND detector initialized successfully;
                    the capture loop is now running and frames/FPS are
                    flowing
    "Error"      -> camera or detector initialization failed, or a frame
                    read failed mid-session; the specific error message is
                    always emitted via error_occurred (never fails silently)
    "Stopped"    -> the loop exited normally (app closing)
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.core.camera_manager import CameraError, CameraManager
from src.core.hand_detector import DetectionResult, HandDetector, HandDetectorError
from src.core.logger_setup import get_logger
from src.utils.config_loader import AppConfig

logger = get_logger(__name__)


class VideoCaptureThread(QThread):
    """
    Background thread that owns the camera and hand detector, continuously
    grabbing frames, running detection, and emitting results to the UI.
    """

    frame_ready = Signal(QImage)          # annotated frame, ready to paint
    detection_ready = Signal(object)      # DetectionResult, for downstream consumers (recorder, etc.)
    fps_updated = Signal(float)
    error_occurred = Signal(str)
    status_changed = Signal(str)          # "Connecting", "Connected", "Stopped", "Error"

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._running = False
        self._camera: CameraManager | None = None
        self._detector: HandDetector | None = None

    def run(self) -> None:
        """Main capture loop, executed on the background thread."""
        self._running = True
        self.status_changed.emit("Connecting")

        try:
            self._camera = CameraManager(self._config.camera)
            self._camera.open()
        except CameraError as exc:
            logger.error("Camera initialization failed: %s", exc)
            self.error_occurred.emit(str(exc))
            self.status_changed.emit("Error")
            return

        try:
            self._detector = HandDetector(
                self._config.mediapipe, model_path=self._config.mediapipe.model_path
            )
        except HandDetectorError as exc:
            logger.error("Hand detector initialization failed: %s", exc)
            self.error_occurred.emit(str(exc))
            self.status_changed.emit("Error")
            self._camera.release()
            return

        # Both the camera and the model are ready; the capture loop is about
        # to start producing frames and FPS updates.
        self.status_changed.emit("Connected")

        while self._running:
            try:
                frame_result = self._camera.read_frame()
            except CameraError as exc:
                logger.error("Frame read failed: %s", exc)
                self.error_occurred.emit(str(exc))
                self.status_changed.emit("Error")
                break

            detection: DetectionResult = self._detector.process(frame_result.frame, draw=True)

            display_frame = detection.annotated_frame if detection.annotated_frame is not None else frame_result.frame
            qimage = self._to_qimage(display_frame)

            self.frame_ready.emit(qimage)
            self.detection_ready.emit(detection)
            self.fps_updated.emit(frame_result.fps)

        self._cleanup()
        self.status_changed.emit("Stopped")

    @staticmethod
    def _to_qimage(frame_bgr: np.ndarray) -> QImage:
        """Convert an OpenCV BGR frame into a QImage for display in Qt widgets."""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        qimage = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        # .copy() detaches the QImage from the numpy buffer, which is about
        # to be overwritten by the next frame.
        return qimage.copy()

    def stop(self) -> None:
        """Signal the loop to stop and wait for the thread to finish cleanly."""
        self._running = False
        self.wait(2000)

    def _cleanup(self) -> None:
        if self._detector is not None:
            self._detector.close()
        if self._camera is not None:
            self._camera.release()
