"""
models/utils.py
================
Reusable helper utilities shared across preprocessing, training, and
inference modules: logging setup, reproducibility, and pickle I/O.
"""

from __future__ import annotations

import logging
import os
import pickle
import random
from typing import Any

import numpy as np

import config


_LOGGERS: dict = {}


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, creating it once per name.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance that writes to both
        stdout and the shared project log file.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    if not logger.handlers:
        formatter = logging.Formatter(config.LOG_FORMAT)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow for reproducibility.

    Args:
        seed: The seed value to apply everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_pickle(obj: Any, path: str) -> None:
    """Serialize an object to disk using pickle.

    Args:
        obj: Any picklable Python object.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load a pickled object from disk.

    Args:
        path: Path to a pickle file.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The same path, for chaining.
    """
    os.makedirs(path, exist_ok=True)
    return path


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a single frame's landmark array to be translation- and
    scale-invariant, using the hip midpoint as the origin and torso length
    as the scale reference.

    Args:
        landmarks: Array of shape ``(num_landmarks, 4)`` holding
            ``(x, y, z, visibility)`` per landmark, in MediaPipe order.

    Returns:
        Normalized array of the same shape.
    """
    if landmarks.size == 0:
        return landmarks

    # MediaPipe Pose indices: 23 = left hip, 24 = right hip,
    # 11 = left shoulder, 12 = right shoulder.
    left_hip, right_hip = landmarks[23, :3], landmarks[24, :3]
    left_shoulder, right_shoulder = landmarks[11, :3], landmarks[12, :3]

    hip_center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    torso_length = np.linalg.norm(shoulder_center - hip_center)
    scale = torso_length if torso_length > 1e-6 else 1.0

    normalized = landmarks.copy()
    normalized[:, :3] = (landmarks[:, :3] - hip_center) / scale
    return normalized
