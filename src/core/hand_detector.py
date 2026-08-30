"""
hand_detector.py

Wraps MediaPipe's **Tasks** HandLandmarker API to detect up to two hands per
frame and extract the 21 landmarks per hand (x, y, z each) needed for
downstream gesture-sequence modeling in later phases.

--------------------------------------------------------------------------
Why this file looks different from the original Phase 1/2 version:

mediapipe>=0.10.x (confirmed on 0.10.35) removed the legacy `mp.solutions`
namespace (`mp.solutions.hands`, `mp.solutions.drawing_utils`) that earlier
MediaPipe releases exposed. Importing it now raises:

    AttributeError: module 'mediapipe' has no attribute 'solutions'

The replacement is MediaPipe's Tasks API
(`mediapipe.tasks.python.vision.HandLandmarker`), which is what this module
now uses. The public surface of this file — `HandDetector`, `DetectionResult`,
`HandResult`, `.process()`, `.flattened_vector()` — is UNCHANGED, so nothing
in video_widget.py, dataset_manager.py, or recorder_widget.py needed to know
or care that the underlying MediaPipe API changed underneath it.
--------------------------------------------------------------------------

One operational difference: the Tasks API requires a local `.task` model
file (it is not bundled in the pip package). See DEFAULT_MODEL_PATH /
MODEL_DOWNLOAD_URL below and README.md's "Model Setup" section for exactly
where to put it. If the file is missing, HandDetector raises a clear,
actionable HandDetectorError at startup instead of failing silently or
crashing deep inside the capture thread.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from src.core.logger_setup import get_logger
from src.utils.config_loader import MediaPipeConfig

logger = get_logger(__name__)

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z

DEFAULT_MODEL_PATH = "models/hand_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Hand-skeleton connections, previously supplied by the removed
# mp.solutions.hands.HAND_CONNECTIONS. The Tasks API exposes the equivalent
# 21 (start, end) landmark-index pairs via HandLandmarksConnections.
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]

# BGR (OpenCV order), not RGB — accent teal #4FD1C5 and light gray.
_LINE_COLOR_BGR = (197, 209, 79)
_POINT_COLOR_BGR = (230, 230, 230)


class HandDetectorError(Exception):
    """Raised when the hand landmark model is missing or fails to initialize."""


@dataclass
class HandResult:
    """Landmark data for a single detected hand."""
    label: str                     # "Left" or "Right"
    landmarks: np.ndarray          # shape (21, 3), normalized [0,1] image coords + relative z
    handedness_score: float


@dataclass
class DetectionResult:
    """All hands detected in a single frame, plus the annotated frame for display."""
    hands: list[HandResult] = field(default_factory=list)
    annotated_frame: np.ndarray | None = None

    @property
    def num_hands(self) -> int:
        return len(self.hands)

    def flattened_vector(self, max_hands: int = 2) -> np.ndarray:
        """
        Return a fixed-length feature vector for this frame, suitable for
        feeding into a sequence model (Phase 4). Missing hands are zero-padded
        so every frame produces a vector of identical length regardless of
        how many hands were actually detected.
        """
        vec = np.zeros(max_hands * NUM_LANDMARKS * COORDS_PER_LANDMARK, dtype=np.float32)
        for i, hand in enumerate(self.hands[:max_hands]):
            start = i * NUM_LANDMARKS * COORDS_PER_LANDMARK
            end = start + NUM_LANDMARKS * COORDS_PER_LANDMARK
            vec[start:end] = hand.landmarks.flatten()
        return vec


class HandDetector:
    """
    Typed wrapper around MediaPipe Tasks' HandLandmarker: validates the
    model file up front, handles BGR->RGB conversion, monotonic VIDEO-mode
    timestamping, landmark extraction, and skeleton drawing (replacing the
    removed mp.solutions.drawing_utils with plain OpenCV drawing).
    """

    def __init__(self, config: MediaPipeConfig, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self._config = config
        self._model_path = Path(model_path)
        self._validate_model_present()

        try:
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=config.max_num_hands,
                min_hand_detection_confidence=config.min_detection_confidence,
                min_hand_presence_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            # MediaPipe's C++ layer raises a mix of exception types here
            # (RuntimeError, FileNotFoundError, etc.) depending on what went
            # wrong; we normalize all of them into one clear, typed error.
            raise HandDetectorError(
                f"Failed to initialize MediaPipe HandLandmarker from '{self._model_path}': {exc}"
            ) from exc

        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

        logger.info(
            "MediaPipe Tasks HandLandmarker initialized (max_hands=%d, model='%s').",
            config.max_num_hands, self._model_path,
        )

    def _validate_model_present(self) -> None:
        """Fail loudly and helpfully at startup if the .task model isn't where expected."""
        if not self._model_path.exists():
            raise HandDetectorError(
                f"Hand landmark model not found at '{self._model_path.resolve()}'.\n\n"
                "MediaPipe's Tasks API (used since the legacy mp.solutions.hands API "
                "was removed in newer mediapipe releases) requires a local .task model "
                "file that is NOT bundled with the pip package.\n\n"
                "To fix this:\n"
                f"  1. Download: {MODEL_DOWNLOAD_URL}\n"
                f"  2. Save it as: {self._model_path}\n"
                "     (create the 'models/' folder in the project root if it doesn't exist)\n"
                "  3. Restart the application.\n"
            )

    def _next_timestamp_ms(self) -> int:
        """
        Monotonically increasing millisecond timestamp required by the Tasks
        API's VIDEO running mode. Derived independently from wall-clock time
        rather than the camera's own frame timestamps, so it's guaranteed to
        strictly increase even if the camera reports jittery/duplicate times.
        """
        ts = int((time.monotonic() - self._start_time) * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def process(self, frame_bgr: np.ndarray, draw: bool = True) -> DetectionResult:
        """
        Run hand detection on a single BGR frame (as returned by OpenCV).

        Returns a DetectionResult with per-hand landmark arrays and, if
        draw=True, an annotated copy of the frame with skeleton overlays
        for the live dashboard feed.
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = self._next_timestamp_ms()
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        annotated = frame_bgr.copy() if draw else None
        detected_hands: list[HandResult] = []

        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            )

            label = "Unknown"
            score = 0.0
            if idx < len(result.handedness) and result.handedness[idx]:
                category = result.handedness[idx][0]
                label = category.category_name or "Unknown"
                score = category.score

            detected_hands.append(HandResult(label=label, landmarks=coords, handedness_score=score))

            if draw and annotated is not None:
                self._draw_landmarks(annotated, hand_landmarks)

        return DetectionResult(hands=detected_hands, annotated_frame=annotated)

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, hand_landmarks) -> None:
        """Draw the hand skeleton directly with OpenCV (replaces mp.solutions.drawing_utils)."""
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for start_idx, end_idx in _HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], _LINE_COLOR_BGR, 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, _POINT_COLOR_BGR, -1)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("MediaPipe HandLandmarker closed.")

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
