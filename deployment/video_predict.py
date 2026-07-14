"""
deployment/video_predict.py
============================
Handles the "upload a video and predict" flow: saving the uploaded file,
running the predictor, optionally producing a sliding-window timeline, and
optionally writing an annotated output video with the skeleton overlay and
predicted label burned in.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from models.inference import ActivityPredictor
from models.mediapipe_extractor import PoseExtractor
from models.utils import get_logger

logger = get_logger(__name__)


def save_uploaded_file(uploaded_file) -> str:
    """Persist a Streamlit ``UploadedFile`` to a temporary path on disk so
    OpenCV can open it by path.

    Args:
        uploaded_file: The object returned by ``st.file_uploader``.

    Returns:
        Path to the temporary video file.
    """
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def predict_uploaded_video(
    predictor: ActivityPredictor, video_path: str, timeline: bool = False
) -> Tuple[Dict, Optional[List[Dict]]]:
    """Run prediction on an uploaded video, optionally with a sliding
    window for an activity timeline.

    Args:
        predictor: A loaded ``ActivityPredictor``.
        video_path: Path to the (temporary) video file.
        timeline: Whether to also compute a per-window timeline.

    Returns:
        Tuple of ``(overall_result, timeline_or_none)``.
    """
    overall = predictor.predict_video(video_path)
    timeline_result = predictor.predict_video_timeline(video_path) if timeline else None
    return overall, timeline_result


def render_annotated_video(
    video_path: str, output_path: str, label: str, confidence: float
) -> str:
    """Write a copy of the video with the pose skeleton and predicted
    label/confidence burned into every frame — used for the "Save
    Prediction Video" bonus feature.

    Args:
        video_path: Source video path.
        output_path: Destination path for the annotated video (``.mp4``).
        label: Predicted activity label to overlay.
        confidence: Confidence score to overlay.

    Returns:
        The output path, for convenience.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with PoseExtractor() as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, results = extractor.extract_frame_landmarks(frame)
            annotated = extractor.draw_landmarks(frame, results)

            text = f"{label} ({confidence:.0%})"
            cv2.rectangle(annotated, (0, 0), (width, 40), (0, 0, 0), -1)
            cv2.putText(
                annotated, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
            writer.write(annotated)

    cap.release()
    writer.release()
    logger.info("Saved annotated video -> %s", output_path)
    return output_path


def cleanup_temp_file(path: str) -> None:
    """Remove a temporary file if it exists, ignoring errors.

    Args:
        path: File path to remove.
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        logger.warning("Could not remove temp file %s: %s", path, exc)
