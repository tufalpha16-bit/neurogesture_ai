"""
gesture_worker.py

Runs GesturePredictor away from the Qt UI thread so TensorFlow inference
does not freeze or slow down the live camera dashboard.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import QObject, Signal, Slot

from src.core.gesture_predictor import GesturePredictor
from src.core.logger_setup import get_logger


logger = get_logger(__name__)


class GestureWorker(QObject):
    """Background worker for LSTM gesture prediction."""

    prediction_ready = Signal(str, float)
    prediction_error = Signal(str)

    def __init__(self) -> None:
        super().__init__()

        self._predictor: GesturePredictor | None = None

    @Slot()
    def initialize(self) -> None:
        """Load the gesture predictor inside the worker thread."""

        try:
            self._predictor = GesturePredictor()

            logger.info(
                "Background gesture predictor initialized."
            )

        except Exception as exc:
            self._predictor = None

            logger.exception(
                "Background gesture predictor failed: %s",
                exc,
            )

            self.prediction_error.emit(
                str(exc)
            )

    @Slot(object)
    def process_frame(
        self,
        vector: object,
    ) -> None:
        """Run one landmark vector through the predictor."""

        if self._predictor is None:
            return

        try:
            array = np.asarray(
                vector,
                dtype=np.float32,
            )

            prediction = (
                self._predictor.add_frame(
                    array
                )
            )

            if prediction is None:
                return

            gesture, confidence = prediction

            self.prediction_ready.emit(
                gesture,
                confidence,
            )

        except Exception as exc:
            logger.exception(
                "Background gesture prediction failed: %s",
                exc,
            )

            self.prediction_error.emit(
                str(exc)
            )

    @Slot()
    def reset(self) -> None:
        """Reset the rolling prediction sequence."""

        if self._predictor is not None:
            self._predictor.reset()