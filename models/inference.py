"""
models/inference.py
====================
High-level prediction API used by both the Streamlit app and any external
script. Wraps a trained Keras model, the label encoder, and the pose
extractor into a single ``ActivityPredictor`` with a rolling-buffer mode
for real-time webcam inference and a batch mode for uploaded videos.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import config
from models.mediapipe_extractor import PoseExtractor
from models.utils import get_logger, load_pickle

logger = get_logger(__name__)


class ActivityPredictor:
    """Runs activity classification given a trained model and label encoder.

    Attributes:
        model: Loaded Keras model.
        classes: Ordered list of class names matching the model's output
            layer indices.
        sequence_length: Number of frames expected per prediction.
    """

    def __init__(
        self,
        model_path: str = config.TRAINING_CONFIG.checkpoint_path,
        label_encoder_path: str = config.TRAINING_CONFIG.label_encoder_path,
        sequence_length: int = config.SEQUENCE_LENGTH,
    ) -> None:
        logger.info("Loading model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path)

        try:
            self.label_encoder = load_pickle(label_encoder_path)
            self.classes: List[str] = list(self.label_encoder.classes_)
        except FileNotFoundError:
            logger.warning(
                "Label encoder not found at %s; falling back to config.ACTIVITY_CLASSES",
                label_encoder_path,
            )
            self.label_encoder = None
            self.classes = config.ACTIVITY_CLASSES

        self.sequence_length = sequence_length
        self.pose_extractor = PoseExtractor()

        # Rolling buffer used for real-time / streaming inference.
        self._frame_buffer: deque = deque(maxlen=sequence_length)

    # ------------------------------------------------------------------ #
    # Batch (uploaded video) inference
    # ------------------------------------------------------------------ #
    def predict_video(self, video_path: str) -> Dict:
        """Run end-to-end prediction on an uploaded video file.

        Args:
            video_path: Path to a video file on disk.

        Returns:
            Dictionary with keys ``activity``, ``confidence``,
            ``probabilities`` (dict of class -> probability), and
            ``inference_time_sec``.
        """
        start = time.time()
        sequence = self.pose_extractor.extract_video_sequence(
            video_path, sequence_length=self.sequence_length
        )
        result = self._predict_sequence(sequence)
        result["inference_time_sec"] = round(time.time() - start, 4)
        return result

    def predict_video_timeline(
        self, video_path: str, window_stride: int = 15
    ) -> List[Dict]:
        """Slide a window across a longer video to produce an activity
        timeline instead of a single prediction.

        Args:
            video_path: Path to a video file on disk.
            window_stride: Number of frames to advance the window each
                step.

        Returns:
            List of per-window prediction dicts, each including a
            ``start_frame`` key.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < self.sequence_length:
            return [self.predict_video(video_path)]

        timeline = []
        for start_idx in range(0, len(frames) - self.sequence_length + 1, window_stride):
            window = frames[start_idx : start_idx + self.sequence_length]
            sequence = np.array(
                [self.pose_extractor.extract_normalized_landmarks(f) for f in window],
                dtype=np.float32,
            )
            result = self._predict_sequence(sequence)
            result["start_frame"] = start_idx
            timeline.append(result)
        return timeline

    # ------------------------------------------------------------------ #
    # Streaming (webcam) inference
    # ------------------------------------------------------------------ #
    def push_frame(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """Feed a single live frame into the rolling buffer; once the
        buffer is full, return a prediction (recomputed each call once
        warmed up, giving continuous real-time updates).

        Args:
            frame_bgr: A single BGR frame from the webcam.

        Returns:
            A prediction dict (see :meth:`_predict_sequence`) once the
            buffer has ``sequence_length`` frames, otherwise ``None``.
        """
        landmarks = self.pose_extractor.extract_normalized_landmarks(frame_bgr)
        self._frame_buffer.append(landmarks)

        if len(self._frame_buffer) < self.sequence_length:
            return None

        sequence = np.array(self._frame_buffer, dtype=np.float32)
        return self._predict_sequence(sequence)

    def reset_buffer(self) -> None:
        """Clear the rolling frame buffer (call when restarting the webcam)."""
        self._frame_buffer.clear()

    # ------------------------------------------------------------------ #
    # Core prediction
    # ------------------------------------------------------------------ #
    def _predict_sequence(self, sequence: np.ndarray) -> Dict:
        """Run the model on a single ``(sequence_length, num_features)``
        array and package the result.

        Args:
            sequence: Landmark sequence array.

        Returns:
            Dict with ``activity``, ``confidence``, and ``probabilities``.
        """
        batch = np.expand_dims(sequence, axis=0)
        probs = self.model.predict(batch, verbose=0)[0]

        top_idx = int(np.argmax(probs))
        activity = self.classes[top_idx] if top_idx < len(self.classes) else "Unknown"
        probabilities = {cls: float(p) for cls, p in zip(self.classes, probs)}

        return {
            "activity": activity,
            "confidence": float(probs[top_idx]),
            "probabilities": probabilities,
        }

    def close(self) -> None:
        """Release underlying resources (MediaPipe pose instance)."""
        self.pose_extractor.close()

    def __enter__(self) -> "ActivityPredictor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
