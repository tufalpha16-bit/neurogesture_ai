@'
# NeuroGesture AI

Real-time hand gesture recognition using MediaPipe, TensorFlow, and an LSTM model.

## Supported Gestures

- next_slide
- previous_slide
- next_track
- previous_track
- pause
- volume_up
- volume_down
- screenshot
- no_gesture

## Features

- Real-time webcam hand tracking
- 30-frame gesture sequences
- LSTM gesture recognition
- Live confidence display
- Windows media controls
- Slide navigation
- Screenshot capture
- Background TensorFlow inference

## Run

Activate the environment:

.\venv312\Scripts\Activate.ps1

Start the application:

python -m src.main

## Training

Prepare dataset:

python .\src\training\prepare_dataset.py

Train model:

python .\src\training\train_lstm.py

## Model

Input: 30 frames x 126 features

Output: 9 gesture classes

Model:

models\trained\neurogesture_lstm.keras
'@ | Set-Content -Path .\README.md -Encoding UTF8