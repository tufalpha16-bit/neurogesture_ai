import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from src.core.hand_detector import HandDetector
from src.core.gesture_predictor import GesturePredictor


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="NeuroGesture AI",
    page_icon="🖐️",
    layout="wide",
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL_PATH = Path("models/trained/neurogesture_lstm.keras")
LABELS_PATH = Path("models/trained/labels.json")
HAND_MODEL_PATH = Path("models/hand_landmarker.task")


# ---------------------------------------------------------
# Initialize AI components
# ---------------------------------------------------------

@st.cache_resource
def load_ai():

    detector = HandDetector(
        config=type(
            "MediaPipeConfig",
            (),
            {
                "max_num_hands": 2,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
        )(),
        model_path=HAND_MODEL_PATH,
    )

    predictor = GesturePredictor(
        model_path=MODEL_PATH,
        labels_path=LABELS_PATH,
    )

    return detector, predictor


# ---------------------------------------------------------
# Video processor
# ---------------------------------------------------------

class GestureVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = None
        self.predictor = None

        self.lock = threading.Lock()

        self.gesture = "Waiting..."
        self.confidence = 0.0
        self.hands = 0

    def initialize(self):

        if self.detector is None or self.predictor is None:
            self.detector, self.predictor = load_ai()

    def recv(self, frame):

        self.initialize()

        image = frame.to_ndarray(format="bgr24")

        try:

            result = self.detector.process(
                image,
                draw=True,
            )

            self.hands = result.num_hands

            if result.num_hands > 0:

                vector = result.flattened_vector(
                    max_hands=2
                )

                prediction = self.predictor.add_frame(
                    vector
                )

                if prediction is not None:

                    gesture, confidence = prediction

                    with self.lock:
                        self.gesture = gesture
                        self.confidence = confidence

            else:

                self.predictor.reset()

                with self.lock:
                    self.gesture = "No hand detected"
                    self.confidence = 0.0

            output = result.annotated_frame

        except Exception as exc:

            cv2.putText(
                image,
                f"Error: {str(exc)[:80]}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            output = image

        return frame.from_ndarray(
            output,
            format="bgr24",
        )


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.title("🖐️ NeuroGesture AI")

st.write(
    "AI-powered hand gesture recognition system."
)

st.success(
    "NeuroGesture AI web application is running!"
)

st.header("🎥 Live Gesture Recognition")

st.write(
    "Allow camera access and perform one of the trained gestures."
)


ctx = webrtc_streamer(
    key="neurogesture-camera",
    video_processor_factory=GestureVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)


st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("🧠 Current Gesture")

    gesture_placeholder = st.empty()

with col2:

    st.subheader("🎯 Confidence")

    confidence_placeholder = st.empty()


if ctx.state.playing:

    st.info(
        "Camera is active. Hold a gesture steady for a moment..."
    )

    # Display basic live status.
    if ctx.video_processor:

        processor = ctx.video_processor

        gesture_placeholder.metric(
            "Gesture",
            processor.gesture,
        )

        confidence_placeholder.metric(
            "Confidence",
            f"{processor.confidence * 100:.1f}%",
        )


st.divider()

st.subheader("Supported Gestures")

st.write(
    """
    • Next Slide  
    • Previous Slide  
    • Next Track  
    • Previous Track  
    • Pause  
    • Volume Up  
    • Volume Down  
    • Screenshot  
    • No Gesture
    """
)

st.caption(
    "NeuroGesture AI • Computer Vision + MediaPipe + LSTM"
)