# Phase 1 — Architecture Notes

## Why a background QThread for capture?

MediaPipe's hand detection runs at several milliseconds per frame even on
modest hardware, and combined with OpenCV frame reads, doing this on the
Qt GUI thread would cause visible stutter or full freezes during window
resize/interaction. `VideoCaptureThread` (in `src/ui/video_widget.py`) owns
the camera and detector, and communicates with the UI purely through Qt
signals (`frame_ready`, `detection_ready`, `fps_updated`, `status_changed`,
`error_occurred`). This keeps `MainWindow` simple: it only ever reacts to
signals, never blocks.

## Why dataclasses for config instead of raw dicts?

`src/utils/config_loader.py` parses `config.yaml` into typed dataclasses
(`CameraConfig`, `MediaPipeConfig`, etc.) rather than passing dictionaries
around. This gives:
- IDE autocomplete (`config.camera.device_index` instead of `config["camera"]["device_index"]`)
- A single place that defines valid keys and defaults, so a typo in the YAML
  produces a clear warning instead of a silent `KeyError` deep in the app
- Forward compatibility: Phase 3's `ModelConfig` fields (architecture,
  hidden_units, epochs, etc.) already exist in the schema now, even though
  training doesn't happen until Phase 3 — so the config file the user edits
  today won't need restructuring later.

## Why a flattened landmark vector method now?

`DetectionResult.flattened_vector()` in `src/core/hand_detector.py` already
produces a fixed-length, zero-padded feature vector (`max_hands × 21 × 3`)
per frame, even though nothing consumes it yet in Phase 1. This is the exact
shape Phase 2's dataset recorder will stack into sequences of length
`sequence_length` (default 30 frames, configurable), and Phase 4's LSTM will
train on. Deciding this format now avoids a breaking change to recorded
datasets later.

## Why CAP_DSHOW on Windows?

`CameraManager.open()` requests OpenCV's DirectShow backend first
(`cv2.CAP_DSHOW`), which is significantly faster to initialize and more
reliable on Windows than the default Media Foundation backend in many
laptop webcam / driver combinations. It falls back to the default backend
automatically if DirectShow isn't available.

## What's intentionally deferred

- **GPU usage in Phase 1:** hand tracking at this stage is CPU-only and
  lightweight; GPU acceleration matters for Phase 3's LSTM training, where
  we'll add TensorFlow GPU-device detection and a documented setup path for
  the RTX 4050.
- **Gesture recording, training, prediction, action-mapping, analytics:**
  each is a self-contained module in Phases 2–6, built on top of the
  `CameraManager` / `HandDetector` primitives established here, so nothing
  in this phase needs to be rewritten as the app grows.
