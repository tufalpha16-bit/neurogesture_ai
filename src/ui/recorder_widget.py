"""
recorder_widget.py

The "Gesture Recorder" tab: lets the user create new gesture classes and
record labeled sequence samples for them, entirely through the UI — no
source code edits required, satisfying requirement #3/#5 of the spec.

Recording flow per sample:
    1. User selects (or creates) a gesture and clicks "Record Sample".
    2. A 3-second on-screen countdown gives them time to get into position.
    3. The widget then captures `sequence_length` consecutive frames' worth
       of landmark vectors (fed to it via `on_detection` from the shared
       VideoCaptureThread) and shows live progress.
    4. On completion, the sequence is handed to GestureDatasetManager,
       saved to disk, and the gesture's sample count updates immediately.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.dataset_manager import DatasetError, GestureDatasetManager
from src.core.hand_detector import DetectionResult
from src.core.logger_setup import get_logger
from src.utils.config_loader import AppConfig

logger = get_logger(__name__)

COUNTDOWN_SECONDS = 3


class RecorderWidget(QWidget):
    """Gesture creation + sample recording UI, backed by GestureDatasetManager."""

    # Emitted whenever the on-disk dataset changes, so other tabs (e.g. the
    # Phase 3 training panel) can refresh their gesture lists without polling.
    dataset_changed = Signal()

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._dataset = GestureDatasetManager(
            storage_path=config.dataset.storage_path,
            sequence_length=config.dataset.sequence_length,
            feature_dim=config.model.input_landmarks * config.model.coordinates_per_landmark * config.mediapipe.max_num_hands,
        )

        self._recording = False
        self._countdown_remaining = 0
        self._frame_buffer: list[np.ndarray] = []
        self._selected_gesture: str | None = None

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        self._build_ui()
        self._refresh_gesture_list()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(12)

        # --- Left: small live preview + recording controls ---
        left_frame = QFrame()
        left_frame.setObjectName("videoFrame")
        left_layout = QVBoxLayout(left_frame)

        self.preview_label = QLabel("Waiting for camera...")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(560, 420)
        left_layout.addWidget(self.preview_label)

        self.status_label = QLabel("Select or create a gesture to begin.")
        self.status_label.setProperty("class", "statLabel")
        left_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self._config.dataset.sequence_length)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.record_button = QPushButton("● Record Sample")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self._start_recording)
        left_layout.addWidget(self.record_button)

        root_layout.addWidget(left_frame, stretch=2)

        # --- Right: gesture management panel ---
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        create_label = QLabel("Create New Gesture")
        create_label.setProperty("class", "statLabel")
        right_layout.addWidget(create_label)

        create_row = QHBoxLayout()
        self.new_gesture_input = QLineEdit()
        self.new_gesture_input.setPlaceholderText("e.g. Swipe Left, Pinch, Draw C")
        self.new_gesture_input.returnPressed.connect(self._create_gesture)
        create_row.addWidget(self.new_gesture_input)

        create_button = QPushButton("Create")
        create_button.clicked.connect(self._create_gesture)
        create_row.addWidget(create_button)
        right_layout.addLayout(create_row)

        gestures_label = QLabel("Your Gestures")
        gestures_label.setProperty("class", "statLabel")
        right_layout.addWidget(gestures_label)

        self.gesture_list = QListWidget()
        self.gesture_list.itemSelectionChanged.connect(self._on_gesture_selected)
        right_layout.addWidget(self.gesture_list, stretch=1)

        self.samples_needed_label = QLabel("")
        self.samples_needed_label.setProperty("class", "statLabel")
        right_layout.addWidget(self.samples_needed_label)

        delete_button = QPushButton("Delete Selected Gesture")
        delete_button.clicked.connect(self._delete_gesture)
        right_layout.addWidget(delete_button)

        root_layout.addLayout(right_layout, stretch=1)

    # ------------------------------------------------------------------ #
    # Gesture management
    # ------------------------------------------------------------------ #
    def _create_gesture(self) -> None:
        name = self.new_gesture_input.text()
        try:
            self._dataset.create_gesture(name)
        except DatasetError as exc:
            QMessageBox.warning(self, "Cannot Create Gesture", str(exc))
            return

        self.new_gesture_input.clear()
        self._refresh_gesture_list(select=name)
        self.dataset_changed.emit()

    def _delete_gesture(self) -> None:
        if not self._selected_gesture:
            QMessageBox.information(self, "No Gesture Selected", "Select a gesture to delete first.")
            return

        confirm = QMessageBox.question(
            self,
            "Delete Gesture",
            f"Delete '{self._selected_gesture}' and all its recorded samples? "
            "This cannot be undone.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            self._dataset.delete_gesture(self._selected_gesture)
        except DatasetError as exc:
            QMessageBox.warning(self, "Cannot Delete Gesture", str(exc))
            return

        self._selected_gesture = None
        self._refresh_gesture_list()
        self.dataset_changed.emit()

    def _refresh_gesture_list(self, select: str | None = None) -> None:
        self.gesture_list.clear()
        for info in self._dataset.list_gestures():
            item = QListWidgetItem(f"{info.name}  ({info.sample_count} samples)")
            item.setData(Qt.ItemDataRole.UserRole, info.name)
            self.gesture_list.addItem(item)
            if select and info.name == select:
                self.gesture_list.setCurrentItem(item)

        if select is None and self.gesture_list.count() == 0:
            self.record_button.setEnabled(False)
            self.samples_needed_label.setText("")

    def _on_gesture_selected(self) -> None:
        items = self.gesture_list.selectedItems()
        if not items:
            self._selected_gesture = None
            self.record_button.setEnabled(False)
            return

        self._selected_gesture = items[0].data(Qt.ItemDataRole.UserRole)
        self.record_button.setEnabled(not self._recording)

        count = self._dataset.get_sample_count(self._selected_gesture)
        target = self._config.dataset.samples_per_gesture
        self.samples_needed_label.setText(f"{count} / {target} samples recorded for '{self._selected_gesture}'")
        self.status_label.setText(f"Ready to record '{self._selected_gesture}'.")

    # ------------------------------------------------------------------ #
    # Recording flow
    # ------------------------------------------------------------------ #
    def _start_recording(self) -> None:
        if not self._selected_gesture or self._recording:
            return

        self._countdown_remaining = COUNTDOWN_SECONDS
        self.record_button.setEnabled(False)
        self.status_label.setText(f"Get ready... recording starts in {self._countdown_remaining}s")
        self._countdown_timer.start()

    def _on_countdown_tick(self) -> None:
        self._countdown_remaining -= 1
        if self._countdown_remaining > 0:
            self.status_label.setText(f"Get ready... recording starts in {self._countdown_remaining}s")
            return

        self._countdown_timer.stop()
        self._frame_buffer.clear()
        self._recording = True
        self.progress_bar.setValue(0)
        self.status_label.setText(f"● Recording '{self._selected_gesture}' — perform the gesture now")

    def _finish_recording(self) -> None:
        self._recording = False
        sequence = np.stack(self._frame_buffer, axis=0)
        self._frame_buffer.clear()

        try:
            new_count = self._dataset.add_sample(self._selected_gesture, sequence)
        except DatasetError as exc:
            QMessageBox.warning(self, "Failed to Save Sample", str(exc))
            self.status_label.setText("Save failed. Try recording again.")
            self.record_button.setEnabled(True)
            return

        target = self._config.dataset.samples_per_gesture
        self.samples_needed_label.setText(f"{new_count} / {target} samples recorded for '{self._selected_gesture}'")
        self.status_label.setText(f"✔ Sample #{new_count} saved for '{self._selected_gesture}'.")
        self.progress_bar.setValue(0)
        self.record_button.setEnabled(True)
        self._refresh_gesture_list(select=self._selected_gesture)
        self.dataset_changed.emit()

    # ------------------------------------------------------------------ #
    # Slots driven by MainWindow's shared VideoCaptureThread signals
    # ------------------------------------------------------------------ #
    def update_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pixmap)

    def on_detection(self, detection: DetectionResult) -> None:
        """Called on every processed frame; only acts while actively recording."""
        if not self._recording:
            return

        vector = detection.flattened_vector(max_hands=self._config.mediapipe.max_num_hands)
        self._frame_buffer.append(vector)
        self.progress_bar.setValue(len(self._frame_buffer))

        if len(self._frame_buffer) >= self._config.dataset.sequence_length:
            self._finish_recording()
