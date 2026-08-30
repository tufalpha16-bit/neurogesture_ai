"""
camera_manager.py

Wraps OpenCV's VideoCapture with error handling, reconnect logic, and FPS
tracking so the rest of the app never talks to cv2 directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.core.logger_setup import get_logger
from src.utils.config_loader import CameraConfig

logger = get_logger(__name__)


class CameraError(Exception):
    """Raised when the webcam cannot be opened or read from."""


@dataclass
class FrameResult:
    """A single captured frame plus timing metadata."""
    frame: np.ndarray
    fps: float
    timestamp: float


class CameraManager:
    """
    Manages the laptop webcam lifecycle: opening, reading frames, computing
    a rolling FPS estimate, and releasing the device cleanly on shutdown.
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._prev_frame_time: float = 0.0
        self._fps_smoothed: float = 0.0

    def open(self) -> None:
        """Open the webcam device. Raises CameraError if it cannot be opened."""
        logger.info("Opening camera at device index %d ...", self._config.device_index)

        # CAP_DSHOW improves startup reliability and reduces latency on Windows.
        self._cap = cv2.VideoCapture(self._config.device_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            # Fallback to default backend in case DirectShow isn't available.
            self._cap = cv2.VideoCapture(self._config.device_index)

        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open webcam at device index {self._config.device_index}. "
                "Check that no other application is using the camera and that "
                "Windows camera privacy permissions allow desktop app access."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.target_fps)

        logger.info(
            "Camera opened successfully (%dx%d @ target %d FPS).",
            self._config.frame_width, self._config.frame_height, self._config.target_fps,
        )

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> FrameResult:
        """
        Read a single frame from the webcam.

        Raises CameraError if a frame could not be captured (e.g. device
        disconnected mid-session).
        """
        if not self.is_open():
            raise CameraError("Camera is not open. Call open() first.")

        success, frame = self._cap.read()  # type: ignore[union-attr]
        if not success or frame is None:
            raise CameraError("Failed to read frame from webcam. It may have been disconnected.")

        if self._config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        now = time.perf_counter()
        instant_fps = 1.0 / (now - self._prev_frame_time) if self._prev_frame_time else 0.0
        self._prev_frame_time = now
        # Exponential smoothing keeps the on-screen FPS counter from jittering.
        alpha = 0.1
        self._fps_smoothed = (alpha * instant_fps) + ((1 - alpha) * self._fps_smoothed) if self._fps_smoothed else instant_fps

        return FrameResult(frame=frame, fps=round(self._fps_smoothed, 1), timestamp=now)

    def release(self) -> None:
        """Release the camera device. Safe to call multiple times."""
        if self._cap is not None:
            self._cap.release()
            logger.info("Camera released.")
            self._cap = None

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
