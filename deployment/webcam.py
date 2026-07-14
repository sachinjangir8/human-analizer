"""
deployment/webcam.py
=====================
Real-time webcam capture and prediction loop, designed to be driven from
Streamlit's main thread via repeated calls to ``WebcamSession.read_and_predict()``
inside a loop bound to a placeholder (Streamlit doesn't support true
background threads well across reruns, so a simple pull-based session
object is used instead of a push-based callback).
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from models.inference import ActivityPredictor
from models.utils import get_logger

logger = get_logger(__name__)


class WebcamSession:
    """Manages an OpenCV video capture device plus an ``ActivityPredictor``
    for continuous real-time activity recognition, tracking FPS as it
    goes.

    Attributes:
        predictor: The ``ActivityPredictor`` used for inference.
        confidence_threshold: Minimum confidence for a prediction to be
            considered "confident" (surfaced to the UI; low-confidence
            predictions are still returned but flagged).
    """

    def __init__(self, predictor: ActivityPredictor, camera_index: int = 0, confidence_threshold: float = 0.6) -> None:
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Could not open webcam at index {camera_index}")

        self.predictor.reset_buffer()
        self._last_frame_time = time.time()
        self._fps = 0.0

    def read_and_predict(self) -> Tuple[Optional[np.ndarray], Optional[Dict], float]:
        """Read a single frame, update FPS, run pose extraction + rolling
        prediction, and draw the skeleton overlay.

        Returns:
            Tuple of:
              - ``annotated_frame`` (RGB, ready for ``st.image``) or
                ``None`` if the read failed.
              - ``prediction`` dict (see ``ActivityPredictor._predict_sequence``)
                or ``None`` if the rolling buffer is not yet full.
              - ``fps``: current smoothed frames-per-second estimate.
        """
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from webcam")
            return None, None, self._fps

        now = time.time()
        instant_fps = 1.0 / max(now - self._last_frame_time, 1e-6)
        self._fps = 0.9 * self._fps + 0.1 * instant_fps if self._fps > 0 else instant_fps
        self._last_frame_time = now

        landmarks, results = self.predictor.pose_extractor.extract_frame_landmarks(frame)
        annotated = self.predictor.pose_extractor.draw_landmarks(frame, results)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        prediction = self.predictor.push_frame(frame)

        return annotated_rgb, prediction, self._fps

    def release(self) -> None:
        """Release the webcam device."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Webcam released")

    def __enter__(self) -> "WebcamSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
