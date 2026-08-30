# models/

This folder holds the MediaPipe **Tasks** hand-landmark model file, required
because `mediapipe>=0.10.x` removed the legacy `mp.solutions.hands` API
(which used to bundle its model automatically).

## Required file

```
models/hand_landmarker.task
```

## How to get it

1. Download it directly from Google's official MediaPipe model bucket:
   **https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task**
2. Save the downloaded file into this `models/` folder, named exactly
   `hand_landmarker.task`.
3. Launch the app. `HandDetector` checks for this file on startup and will
   raise a clear error naming the exact expected path if it's missing —
   it will not fail silently or crash with a confusing stack trace.

## Command-line alternative (PowerShell)

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
```

The float16 variant (~7-8 MB) is the standard general-purpose model and is
what `config/config.yaml` points to by default (`mediapipe.model_path`).
