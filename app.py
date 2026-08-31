import threading
from pathlib import Path

import cv2
import streamlit as st
from streamlit_webrtc import (
    VideoProcessorBase,
    webrtc_streamer,
)

from src.core.hand_detector import HandDetector
from src.core.gesture_predictor import GesturePredictor


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="NeuroGesture AI",
    page_icon="🖐️",
    layout="wide",
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = Path(
    "models/trained/neurogesture_lstm.keras"
)

LABELS_PATH = Path(
    "models/trained/labels.json"
)

HAND_MODEL_PATH = Path(
    "models/hand_landmarker.task"
)


# =========================================================
# LOAD AI
# =========================================================

@st.cache_resource
def load_ai():

    config = type(
        "MediaPipeConfig",
        (),
        {
            "max_num_hands": 2,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
    )()

    detector = HandDetector(
        config=config,
        model_path=HAND_MODEL_PATH,
    )

    predictor = GesturePredictor(
        model_path=MODEL_PATH,
        labels_path=LABELS_PATH,
    )

    return detector, predictor


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class GestureVideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = None
        self.predictor = None

        self.lock = threading.Lock()

        self.gesture = "Waiting..."
        self.confidence = 0.0
        self.hands = 0

        self.frame_count = 0

    def initialize(self):

        if self.detector is None:

            self.detector, self.predictor = load_ai()

    def recv(self, frame):

        self.initialize()

        image = frame.to_ndarray(
            format="bgr24"
        )

        try:

            # -------------------------------------------------
            # Hand detection
            # -------------------------------------------------

            result = self.detector.process(
                image,
                draw=True,
            )

            hands = result.num_hands

            # -------------------------------------------------
            # Update hand count
            # -------------------------------------------------

            with self.lock:
                self.hands = hands

            # -------------------------------------------------
            # Gesture prediction
            # -------------------------------------------------

            if hands > 0:

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

                        self.confidence = float(
                            confidence
                        )

            else:

                self.predictor.reset()

                with self.lock:

                    self.gesture = (
                        "No hand detected"
                    )

                    self.confidence = 0.0

            # -------------------------------------------------
            # Return annotated frame
            # -------------------------------------------------

            output = result.annotated_frame

            if output is None:
                output = image

            return frame.from_ndarray(
                output,
                format="bgr24",
            )

        except Exception as exc:

            cv2.putText(
                image,
                "Processing error",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                image,
                str(exc)[:90],
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            return frame.from_ndarray(
                image,
                format="bgr24",
            )


# =========================================================
# HEADER
# =========================================================

st.title("🖐️ NeuroGesture AI")

st.write(
    "AI-powered real-time hand gesture recognition."
)

st.success(
    "System ready"
)


# =========================================================
# CAMERA
# =========================================================

st.header("🎥 Live Gesture Recognition")

st.write(
    "Allow camera access and hold a trained gesture "
    "steady for a moment."
)


ctx = webrtc_streamer(
    key="neurogesture-camera",
    video_processor_factory=GestureVideoProcessor,

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },

    media_stream_constraints={
        "video": True,
        "audio": False,
    },

    async_processing=True,
)


# =========================================================
# DASHBOARD
# =========================================================

st.divider()

col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("🧠 Gesture")

    gesture_box = st.empty()


with col2:

    st.subheader("🎯 Confidence")

    confidence_box = st.empty()


with col3:

    st.subheader("🖐️ Hands")

    hands_box = st.empty()


# =========================================================
# LIVE STATUS
# =========================================================

if ctx.state.playing:

    st.info(
        "Camera active — hold your gesture steady."
    )

    processor = ctx.video_processor

    if processor is not None:

        with processor.lock:

            current_gesture = processor.gesture
            current_confidence = processor.confidence
            current_hands = processor.hands

        gesture_box.metric(
            "Current Gesture",
            current_gesture,
        )

        confidence_box.metric(
            "Confidence",
            f"{current_confidence * 100:.1f}%",
        )

        hands_box.metric(
            "Detected Hands",
            str(current_hands),
        )

else:

    gesture_box.metric(
        "Current Gesture",
        "Camera off",
    )

    confidence_box.metric(
        "Confidence",
        "0.0%",
    )

    hands_box.metric(
        "Detected Hands",
        "0",
    )


# =========================================================
# SUPPORTED GESTURES
# =========================================================

st.divider()

st.subheader("Supported Gestures")

st.write(
    """
    🖐️ Next Slide

    🖐️ Previous Slide

    🖐️ Next Track

    🖐️ Previous Track

    🖐️ Pause

    🖐️ Volume Up

    🖐️ Volume Down

    🖐️ Screenshot

    🖐️ No Gesture
    """
)


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "NeuroGesture AI • MediaPipe + TensorFlow + LSTM"
)