# Bug Fix: `AttributeError: module 'mediapipe' has no attribute 'solutions'`

## Root cause

`src/core/hand_detector.py` was originally written against MediaPipe's
legacy **Solutions** API (`mp.solutions.hands`, `mp.solutions.drawing_utils`).
Google has removed the `solutions` namespace from the `mediapipe` PyPI
package in the versions being installed now (confirmed against the exact
version in use, **mediapipe 0.10.35**, where `mp.solutions` no longer
exists at all — `hasattr(mediapipe, 'solutions')` is `False`).

This meant `HandDetector.__init__()` crashed the instant it tried to build
`mp.solutions.hands.Hands(...)`, which happens inside the background
`VideoCaptureThread` — so the UI never got past "Connecting to webcam..."
and the real error only showed up in the terminal.

## What changed

Only **one file's implementation** needed to change: `src/core/hand_detector.py`.
It was rewritten to use MediaPipe's **Tasks** API
(`mediapipe.tasks.python.vision.HandLandmarker`), which is the currently
supported replacement. Every other file that *uses* `HandDetector` was
touched only for cosmetic/status reasons, not because the detector's public
interface changed:

| File | Why it changed |
|---|---|
| `src/core/hand_detector.py` | **Core fix.** Migrated from `mp.solutions.hands` to `mediapipe.tasks.python.vision.HandLandmarker`. Public class/method signatures (`HandDetector`, `DetectionResult`, `.process()`, `.flattened_vector()`) are unchanged, so nothing downstream needed to adapt. |
| `src/utils/config_loader.py` | Added one new field, `MediaPipeConfig.model_path`, defaulting to `"models/hand_landmarker.task"`. |
| `config/config.yaml` | Added the corresponding `mediapipe.model_path` setting. |
| `src/ui/video_widget.py` | Split camera-init and detector-init into separate `try/except` blocks (`CameraError` vs. the new `HandDetectorError`) so a missing model file is reported distinctly from a webcam problem. Renamed the "ready" status from `"Live"` to `"Connected"` per the requested status lifecycle. |
| `requirements.txt` | Pinned `mediapipe==0.10.35` to match the version actually being used, with a note explaining the API change. |
| `models/` (new folder) | Where the required `.task` model file must be placed. Includes its own `README.md` with the download link. |

**Not changed:** `camera_manager.py`, `dataset_manager.py`,
`dashboard_tab.py`, `recorder_widget.py`, `main_window.py`'s tab structure,
`main.py`, the `.npy` dataset format, or the shared single-capture-thread
architecture. The Gesture Recorder tab and dataset manager needed zero
changes because `HandDetector`'s public output shape (21 landmarks × 3
coords × up to 2 hands, via `flattened_vector()`) is identical before and
after the migration.

## Why the Tasks API needs a separate model file

Unlike the old Solutions API (which bundled a model inside the package),
the Tasks API loads its model from an explicit `.task` file path at
runtime. This is documented in `models/README.md`, and `HandDetector`
validates the file's presence *before* attempting to initialize the
landmarker, raising `HandDetectorError` with the exact expected path and a
download link if it's missing — never a silent failure.

## Why VIDEO running mode (not IMAGE)

The Tasks API has three running modes: `IMAGE` (single unrelated photos),
`VIDEO` (a sequence of frames with monotonically increasing timestamps,
synchronous), and `LIVE_STREAM` (asynchronous, callback-based). Since
`VideoCaptureThread` already runs its own dedicated loop calling
`detector.process()` frame-by-frame synchronously, `VIDEO` mode was the
correct fit — it avoids the added complexity of `LIVE_STREAM`'s callback
threading model while still being appropriate for continuous frames (unlike
`IMAGE` mode, which doesn't track hand identity/tremor smoothing across
calls). `HandDetector` maintains its own monotonic millisecond clock
(`_next_timestamp_ms()`) independent of the camera's own frame timestamps,
so it's guaranteed to always increase even if the camera driver reports
jittery timestamps.

## Why the skeleton is now drawn with raw OpenCV instead of `drawing_utils`

`mp.solutions.drawing_utils` was part of the removed `solutions` namespace.
The Tasks API's `HandLandmarksConnections.HAND_CONNECTIONS` still exposes
the same 21 (start, end) landmark-index pairs that used to back
`drawing_utils`, so `HandDetector._draw_landmarks()` now draws the same
skeleton directly with `cv2.line` / `cv2.circle` — same visual result, one
fewer removed dependency.

## Camera status lifecycle (as requested)

`VideoCaptureThread.status_changed` now emits, in order:

1. `"Connecting"` — the instant `run()` starts, before camera or model init
2. `"Connected"` — only once **both** the camera has opened *and* the hand
   landmark model has initialized successfully; this is also when the
   frame/FPS signal stream begins
3. `"Error"` — if either camera or model init fails, or a frame read fails
   mid-session; the specific exception message is always emitted via
   `error_occurred` and logged, and the dashboard tab displays it directly
   in the video panel (never a silent failure)
4. `"Stopped"` — on clean shutdown
