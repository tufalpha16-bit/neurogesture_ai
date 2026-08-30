"""
gesture_predictor.py

Loads the trained NeuroGesture LSTM model and predicts gestures from
a rolling sequence of 30 hand-landmark frames.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.core.logger_setup import get_logger

logger = get_logger(__name__)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 126

# Keep video smooth by predicting every 10 frames.
PREDICTION_INTERVAL = 10

# Minimum confidence required.
CONFIDENCE_THRESHOLD = 0.60

# Same prediction required this many times.
STABILITY_REQUIRED = 3


class GesturePredictor:
    """Predict gestures from a rolling 30-frame landmark sequence."""

    def __init__(
        self,
        model_path: str | Path = "models/trained/neurogesture_lstm.keras",
        labels_path: str | Path = "models/trained/labels.json",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:

        self._model_path = Path(model_path)
        self._labels_path = Path(labels_path)
        self._confidence_threshold = confidence_threshold

        self._model = None
        self._labels: list[str] = []

        self._buffer: deque[np.ndarray] = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self._frames_since_prediction = 0

        self._candidate_gesture: str | None = None
        self._candidate_count = 0

        self._stable_gesture: str | None = None
        self._stable_confidence = 0.0

        self._load_model()
        self._load_labels()

    def _load_model(self) -> None:
        """Load the trained TensorFlow model."""

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Gesture model not found: {self._model_path}"
            )

        self._model = tf.keras.models.load_model(
            self._model_path
        )

        logger.info(
            "Gesture model loaded from '%s'.",
            self._model_path,
        )

    def _load_labels(self) -> None:
        """Load gesture labels."""

        if not self._labels_path.exists():
            raise FileNotFoundError(
                f"Gesture labels not found: {self._labels_path}"
            )

        with self._labels_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        self._labels = data["classes"]

        logger.info(
            "Loaded %d gesture labels: %s",
            len(self._labels),
            self._labels,
        )

    def reset(self) -> None:
        """Reset everything when no hand is detected."""

        self._buffer.clear()
        self._frames_since_prediction = 0

        self._candidate_gesture = None
        self._candidate_count = 0

        self._stable_gesture = None
        self._stable_confidence = 0.0

    def add_frame(
        self,
        vector: np.ndarray,
    ) -> tuple[str, float] | None:
        """Add one landmark vector and return a stable gesture when ready."""

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.shape != (FEATURE_SIZE,):
            raise ValueError(
                f"Expected landmark vector shape "
                f"({FEATURE_SIZE},), got {vector.shape}"
            )

        self._buffer.append(vector)

        # Need 30 frames.
        if len(self._buffer) < SEQUENCE_LENGTH:
            return None

        self._frames_since_prediction += 1

        # Predict every 10 frames.
        if self._frames_since_prediction < PREDICTION_INTERVAL:
            return None

        self._frames_since_prediction = 0

        sequence = np.stack(
            self._buffer,
            axis=0,
        )

        model_input = np.expand_dims(
            sequence,
            axis=0,
        )

        predictions = self._model.predict(
            model_input,
            verbose=0,
        )[0]

        class_index = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[class_index]
        )

        if class_index >= len(self._labels):
            logger.error(
                "Invalid class index %d. Labels: %d",
                class_index,
                len(self._labels),
            )
            return None

        gesture = self._labels[class_index]

        # Print useful prediction information.
        logger.info(
            "RAW GESTURE: %s (%.1f%%)",
            gesture,
            confidence * 100,
        )

        # Ignore weak predictions.
        if confidence < self._confidence_threshold:
            return None

        # Same gesture again.
        if gesture == self._candidate_gesture:
            self._candidate_count += 1
        else:
            self._candidate_gesture = gesture
            self._candidate_count = 1

        logger.info(
            "GESTURE CANDIDATE: %s (%d/%d)",
            gesture,
            self._candidate_count,
            STABILITY_REQUIRED,
        )

        # Not stable yet.
        if self._candidate_count < STABILITY_REQUIRED:
            return None

        # Stable gesture.
        self._stable_gesture = gesture
        self._stable_confidence = confidence

        logger.info(
            "STABLE GESTURE: %s (%.1f%%)",
            self._stable_gesture,
            self._stable_confidence * 100,
        )

        return (
            self._stable_gesture,
            self._stable_confidence,
        )