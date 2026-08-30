"""
action_controller.py

Fast Windows action controller for NeuroGesture AI.

Actions:
- next_slide
- previous_slide
- next_track
- previous_track
- pause
- volume_up
- volume_down
- screenshot
"""

from __future__ import annotations

import ctypes
import threading
import time
from pathlib import Path

from src.core.logger_setup import get_logger


logger = get_logger(__name__)


# Windows virtual-key codes.
VK_LEFT = 0x25
VK_RIGHT = 0x27

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF

VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_KEYUP = 0x0002


class ActionController:
    """Executes recognized gestures as Windows actions."""

    def __init__(
        self,
        cooldown_seconds: float = 1.5,
    ) -> None:

        self._cooldown_seconds = (
            cooldown_seconds
        )

        self._last_gesture: str | None = None
        self._last_action_time = 0.0

        self._lock = threading.Lock()

        logger.info(
            "Action controller initialized."
        )

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def execute(
        self,
        gesture: str,
    ) -> bool:
        """
        Execute an action in the background.

        Returns True when the gesture was accepted.
        """

        if not gesture:
            return False

        gesture = gesture.strip().lower()

        if gesture == "no_gesture":
            return False

        now = time.monotonic()

        # -------------------------------------------------------------- #
        # Cooldown
        # -------------------------------------------------------------- #

        with self._lock:

            if (
                gesture == self._last_gesture
                and (
                    now
                    - self._last_action_time
                    < self._cooldown_seconds
                )
            ):
                return False

            self._last_gesture = gesture
            self._last_action_time = now

        # -------------------------------------------------------------- #
        # Run action in background.
        # -------------------------------------------------------------- #

        worker = threading.Thread(
            target=self._run_action,
            args=(gesture,),
            daemon=True,
        )

        worker.start()

        return True

    # ------------------------------------------------------------------ #
    # Dispatcher
    # ------------------------------------------------------------------ #

    def _run_action(
        self,
        gesture: str,
    ) -> None:

        try:

            if gesture == "next_slide":
                self.next_slide()

            elif gesture == "previous_slide":
                self.previous_slide()

            elif gesture == "next_track":
                self.next_track()

            elif gesture == "previous_track":
                self.previous_track()

            elif gesture == "pause":
                self.pause_media()

            elif gesture == "volume_up":
                self.volume_up()

            elif gesture == "volume_down":
                self.volume_down()

            elif gesture == "screenshot":
                self.take_screenshot()

            else:
                logger.warning(
                    "Unknown gesture: %s",
                    gesture,
                )
                return

            logger.info(
                "ACTION EXECUTED: %s",
                gesture,
            )

        except Exception as exc:

            logger.exception(
                "Action failed for '%s': %s",
                gesture,
                exc,
            )

    # ------------------------------------------------------------------ #
    # Keyboard
    # ------------------------------------------------------------------ #

    @staticmethod
    def _press_key(
        virtual_key: int,
    ) -> None:

        user32 = ctypes.windll.user32

        user32.keybd_event(
            virtual_key,
            0,
            0,
            0,
        )

        user32.keybd_event(
            virtual_key,
            0,
            KEYEVENTF_KEYUP,
            0,
        )

    # ------------------------------------------------------------------ #
    # Slides
    # ------------------------------------------------------------------ #

    def next_slide(self) -> None:

        self._press_key(
            VK_RIGHT
        )

        logger.info(
            "Next slide"
        )

    def previous_slide(self) -> None:

        self._press_key(
            VK_LEFT
        )

        logger.info(
            "Previous slide"
        )

    # ------------------------------------------------------------------ #
    # Media
    # ------------------------------------------------------------------ #

    def next_track(self) -> None:

        self._press_key(
            VK_MEDIA_NEXT_TRACK
        )

        logger.info(
            "Next track"
        )

    def previous_track(self) -> None:

        self._press_key(
            VK_MEDIA_PREV_TRACK
        )

        logger.info(
            "Previous track"
        )

    def pause_media(self) -> None:

        self._press_key(
            VK_MEDIA_PLAY_PAUSE
        )

        logger.info(
            "Play/Pause"
        )

    # ------------------------------------------------------------------ #
    # Volume
    # ------------------------------------------------------------------ #

    def volume_up(self) -> None:

        self._press_key(
            VK_VOLUME_UP
        )

        logger.info(
            "Volume up"
        )

    def volume_down(self) -> None:

        self._press_key(
            VK_VOLUME_DOWN
        )

        logger.info(
            "Volume down"
        )

    # ------------------------------------------------------------------ #
    # Screenshot
    # ------------------------------------------------------------------ #

    def take_screenshot(self) -> None:
        """Capture the primary Windows display."""

        screenshots_dir = Path(
            "screenshots"
        )

        screenshots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        output_file = (
            screenshots_dir
            / f"screenshot_{timestamp}.png"
        )

        try:

            from PIL import ImageGrab

            image = ImageGrab.grab()

            image.save(
                output_file,
                "PNG",
            )

            logger.info(
                "Screenshot saved: %s",
                output_file,
            )

        except ImportError:

            logger.error(
                "Pillow is not installed. "
                "Install it with: pip install pillow"
            )

        except Exception as exc:

            logger.exception(
                "Screenshot failed: %s",
                exc,
            )