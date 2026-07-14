"""
models/mediapipe_extractor.py
==============================
Wraps Google's MediaPipe Pose solution to extract 33-point body landmarks
from individual video frames, with utilities for handling missing
detections and drawing the skeleton for visualization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python and mediapipe are required. Install via "
        "`pip install -r requirements.txt`."
    ) from exc

import config
from models.utils import get_logger, normalize_landmarks

logger = get_logger(__name__)


class PoseExtractor:
    """Extracts and tracks human pose landmarks from video frames using
    MediaPipe Pose.

    Attributes:
        min_detection_confidence: Minimum confidence for the initial
            detection to be considered successful.
        min_tracking_confidence: Minimum confidence for landmarks to be
            considered tracked between frames.
        model_complexity: 0 (lite), 1 (full), or 2 (heavy) — trades
            accuracy for speed.
    """

    def __init__(
        self,
        min_detection_confidence: float = config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.POSE_MIN_TRACKING_CONFIDENCE,
        model_complexity: int = config.POSE_MODEL_COMPLEXITY,
        static_image_mode: bool = False,
    ) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_valid_landmarks: Optional[np.ndarray] = None
        logger.info("PoseExtractor initialized (complexity=%d)", model_complexity)

    def extract_frame_landmarks(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Optional["mp.solutions.pose.PoseLandmark"]]:
        """Run pose detection on a single BGR frame.

        Args:
            frame_bgr: Frame as read by OpenCV (BGR channel order).

        Returns:
            A tuple of:
              - ``landmarks``: array of shape ``(33, 4)`` with
                ``(x, y, z, visibility)`` per landmark, normalized to
                ``[0, 1]`` image coordinates. If no person is detected,
                the last valid landmarks are reused (if available) or a
                zero array is returned.
              - ``raw_results``: the raw MediaPipe results object (useful
                for drawing), or ``None`` if detection failed entirely.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks is None:
            logger.debug("No pose detected in frame; using fallback landmarks")
            if self._last_valid_landmarks is not None:
                return self._last_valid_landmarks.copy(), None
            return np.zeros((config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK)), None

        landmarks = np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ],
            dtype=np.float32,
        )
        self._last_valid_landmarks = landmarks
        return landmarks, results

    def extract_normalized_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract landmarks and apply scale/translation normalization.

        Args:
            frame_bgr: Frame as read by OpenCV (BGR channel order).

        Returns:
            Flattened, normalized landmark vector of length
            ``config.FEATURES_PER_FRAME``.
        """
        landmarks, _ = self.extract_frame_landmarks(frame_bgr)
        normalized = normalize_landmarks(landmarks)
        return normalized.flatten()

    def draw_landmarks(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """Draw the detected pose skeleton onto a copy of the frame.

        Args:
            frame_bgr: Original BGR frame.
            results: Raw MediaPipe results object from
                :meth:`extract_frame_landmarks`. If ``None``, the frame is
                returned unmodified.

        Returns:
            A new BGR frame with the skeleton overlay drawn, if available.
        """
        annotated = frame_bgr.copy()
        if results is not None and results.pose_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated

    def extract_video_sequence(
        self,
        video_path: str,
        sequence_length: int = config.SEQUENCE_LENGTH,
        frame_skip: int = config.FRAME_SKIP,
    ) -> np.ndarray:
        """Extract a fixed-length landmark sequence from a video file.

        Frames are sampled evenly across the video to produce exactly
        ``sequence_length`` frames, so the resulting sequence has a
        consistent shape regardless of the source video's length or FPS.

        Args:
            video_path: Path to the video file.
            sequence_length: Number of frames the output sequence should
                contain.
            frame_skip: Step used when reading frames before sampling
                (a coarse pre-filter for very long videos).

        Returns:
            Array of shape ``(sequence_length, config.FEATURES_PER_FRAME)``.

        Raises:
            IOError: If the video file cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        frames: List[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_skip == 0:
                frames.append(frame)
            idx += 1
        cap.release()

        if len(frames) == 0:
            logger.warning("No frames read from %s", video_path)
            return np.zeros((sequence_length, config.FEATURES_PER_FRAME), dtype=np.float32)

        sample_indices = np.linspace(0, len(frames) - 1, sequence_length).astype(int)
        sequence = np.array(
            [self.extract_normalized_landmarks(frames[i]) for i in sample_indices],
            dtype=np.float32,
        )
        return sequence

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
