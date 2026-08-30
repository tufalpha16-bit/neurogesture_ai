"""
dataset_manager.py

Manages the on-disk gesture dataset: creating new gesture classes, saving
recorded landmark sequences as labeled samples, and reporting sample counts.

Layout on disk:

    data/gestures/
        metadata.json                 <- registry of all gestures + settings used to record them
        <gesture_name>/
            sample_0001.npy           <- shape (sequence_length, feature_dim)
            sample_0002.npy
            ...

Each .npy file is a stack of per-frame flattened landmark vectors, produced
by HandDetector.DetectionResult.flattened_vector() in Phase 1. Keeping the
exact same vector format end-to-end means Phase 3's LSTM training reads
these files with zero conversion logic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.core.logger_setup import get_logger

logger = get_logger(__name__)

_VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\- ]{2,40}$")


class DatasetError(Exception):
    """Raised for invalid gesture names, duplicate gestures, or I/O failures."""


@dataclass
class GestureInfo:
    """Summary of one recorded gesture class."""
    name: str
    sample_count: int
    created_at: str
    sequence_length: int
    feature_dim: int


@dataclass
class _Metadata:
    sequence_length: int
    feature_dim: int
    gestures: dict[str, dict] = field(default_factory=dict)  # name -> {created_at, sample_count}


class GestureDatasetManager:
    """
    Owns all reads/writes to the gesture dataset directory. The UI layer
    (RecorderWidget) never touches the filesystem directly — everything
    goes through this class so the on-disk format can evolve in one place.
    """

    def __init__(self, storage_path: str, sequence_length: int, feature_dim: int) -> None:
        self._root = Path(storage_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self._root / "metadata.json"
        self._sequence_length = sequence_length
        self._feature_dim = feature_dim
        self._metadata = self._load_metadata()

    # ------------------------------------------------------------------ #
    # Metadata persistence
    # ------------------------------------------------------------------ #
    def _load_metadata(self) -> _Metadata:
        if not self._metadata_path.exists():
            meta = _Metadata(sequence_length=self._sequence_length, feature_dim=self._feature_dim)
            self._save_metadata(meta)
            return meta

        try:
            raw = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            return _Metadata(
                sequence_length=raw.get("sequence_length", self._sequence_length),
                feature_dim=raw.get("feature_dim", self._feature_dim),
                gestures=raw.get("gestures", {}),
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read dataset metadata, starting fresh: %s", exc)
            meta = _Metadata(sequence_length=self._sequence_length, feature_dim=self._feature_dim)
            self._save_metadata(meta)
            return meta

    def _save_metadata(self, meta: _Metadata | None = None) -> None:
        meta = meta or self._metadata
        payload = {
            "sequence_length": meta.sequence_length,
            "feature_dim": meta.feature_dim,
            "gestures": meta.gestures,
        }
        self._metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Gesture lifecycle
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_name(name: str) -> str:
        name = name.strip()
        if not _VALID_NAME_PATTERN.match(name):
            raise DatasetError(
                "Gesture name must be 2-40 characters: letters, numbers, spaces, "
                "hyphens, or underscores only."
            )
        return name

    def create_gesture(self, name: str) -> GestureInfo:
        """Register a new gesture class and create its folder. No-op-safe: raises if it already exists."""
        name = self._validate_name(name)
        if name in self._metadata.gestures:
            raise DatasetError(f"Gesture '{name}' already exists.")

        gesture_dir = self._root / name
        gesture_dir.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now().isoformat(timespec="seconds")
        self._metadata.gestures[name] = {"created_at": created_at, "sample_count": 0}
        self._save_metadata()

        logger.info("Created new gesture '%s'.", name)
        return GestureInfo(
            name=name,
            sample_count=0,
            created_at=created_at,
            sequence_length=self._metadata.sequence_length,
            feature_dim=self._metadata.feature_dim,
        )

    def delete_gesture(self, name: str) -> None:
        """Remove a gesture class and all its recorded samples from disk."""
        if name not in self._metadata.gestures:
            raise DatasetError(f"Gesture '{name}' does not exist.")

        gesture_dir = self._root / name
        for sample_file in gesture_dir.glob("sample_*.npy"):
            sample_file.unlink()
        try:
            gesture_dir.rmdir()
        except OSError:
            pass

        del self._metadata.gestures[name]
        self._save_metadata()
        logger.info("Deleted gesture '%s' and all its samples.", name)

    def list_gestures(self) -> list[GestureInfo]:
        """Return all registered gestures with their current sample counts."""
        return [
            GestureInfo(
                name=name,
                sample_count=info.get("sample_count", 0),
                created_at=info.get("created_at", ""),
                sequence_length=self._metadata.sequence_length,
                feature_dim=self._metadata.feature_dim,
            )
            for name, info in sorted(self._metadata.gestures.items())
        ]

    # ------------------------------------------------------------------ #
    # Sample recording
    # ------------------------------------------------------------------ #
    def add_sample(self, gesture_name: str, sequence: np.ndarray) -> int:
        """
        Save one recorded sequence (shape: [sequence_length, feature_dim])
        for the given gesture. Returns the new total sample count.
        """
        if gesture_name not in self._metadata.gestures:
            raise DatasetError(f"Gesture '{gesture_name}' does not exist. Create it first.")

        expected_shape = (self._metadata.sequence_length, self._metadata.feature_dim)
        if sequence.shape != expected_shape:
            raise DatasetError(
                f"Sequence shape {sequence.shape} does not match expected {expected_shape}."
            )

        gesture_dir = self._root / gesture_name
        gesture_dir.mkdir(parents=True, exist_ok=True)

        next_index = self._metadata.gestures[gesture_name]["sample_count"] + 1
        sample_path = gesture_dir / f"sample_{next_index:04d}.npy"
        np.save(sample_path, sequence.astype(np.float32))

        self._metadata.gestures[gesture_name]["sample_count"] = next_index
        self._save_metadata()

        logger.info("Saved sample #%d for gesture '%s' -> %s", next_index, gesture_name, sample_path)
        return next_index

    def get_sample_count(self, gesture_name: str) -> int:
        if gesture_name not in self._metadata.gestures:
            raise DatasetError(f"Gesture '{gesture_name}' does not exist.")
        return self._metadata.gestures[gesture_name]["sample_count"]
