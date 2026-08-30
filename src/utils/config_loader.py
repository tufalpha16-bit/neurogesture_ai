"""
config_loader.py

Loads and validates the application's YAML configuration file, exposing it
as a typed, dot-accessible object so the rest of the codebase never touches
raw dictionaries or hardcodes settings.

Usage:
    from src.utils.config_loader import get_config
    cfg = get_config()
    print(cfg.camera.device_index)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("neurogesture.config")


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed, or invalid."""


@dataclass
class CameraConfig:
    device_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    flip_horizontal: bool = True


@dataclass
class MediaPipeConfig:
    max_num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    model_complexity: int = 1
    # Path to the .task model file required by MediaPipe's Tasks HandLandmarker
    # API (mediapipe>=0.10.x no longer bundles a model via mp.solutions.hands).
    # See README.md "Model Setup" for the download link and placement.
    model_path: str = "models/hand_landmarker.task"


@dataclass
class DatasetConfig:
    sequence_length: int = 30
    samples_per_gesture: int = 40
    storage_path: str = "data/gestures"


@dataclass
class ModelConfig:
    storage_path: str = "data/models"
    active_model: str = "default_model.h5"
    input_landmarks: int = 21
    coordinates_per_landmark: int = 3
    architecture: str = "LSTM"
    hidden_units: int = 64
    dropout: float = 0.3
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001


@dataclass
class PredictionConfig:
    confidence_threshold: float = 0.80
    smoothing_window: int = 5
    cooldown_seconds: float = 1.0


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_level: str = "INFO"
    max_log_files: int = 10


@dataclass
class UIConfig:
    theme: str = "dark"
    window_title: str = "NeuroGesture AI"
    window_width: int = 1400
    window_height: int = 850


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)


_SECTION_MAP = {
    "camera": CameraConfig,
    "mediapipe": MediaPipeConfig,
    "dataset": DatasetConfig,
    "model": ModelConfig,
    "prediction": PredictionConfig,
    "logging": LoggingConfig,
    "ui": UIConfig,
}

_cached_config: AppConfig | None = None


def _build_section(section_cls: type, raw: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass section, falling back to defaults for missing keys."""
    raw = raw or {}
    valid_fields = {f for f in section_cls.__dataclass_fields__}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    unknown = set(raw) - valid_fields
    if unknown:
        logger.warning("Ignoring unknown config keys in %s: %s", section_cls.__name__, unknown)
    return section_cls(**filtered)


def load_config(config_path: str | Path = "config/config.yaml") -> AppConfig:
    """
    Load configuration from a YAML file into a validated AppConfig object.

    Falls back to built-in defaults (with a warning) if the file is missing,
    so the application can still start with sane values during first-time setup.
    """
    path = Path(config_path)

    if not path.exists():
        logger.warning("Config file not found at '%s'. Using default configuration.", path)
        return AppConfig()

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config at '{path}': {exc}") from exc

    kwargs = {
        section_name: _build_section(section_cls, raw_data.get(section_name))
        for section_name, section_cls in _SECTION_MAP.items()
    }
    logger.info("Configuration loaded successfully from '%s'.", path)
    return AppConfig(**kwargs)


def get_config(config_path: str | Path = "config/config.yaml", force_reload: bool = False) -> AppConfig:
    """Return a cached, process-wide AppConfig instance (loads once unless forced)."""
    global _cached_config
    if _cached_config is None or force_reload:
        _cached_config = load_config(config_path)
    return _cached_config
