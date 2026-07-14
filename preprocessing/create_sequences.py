"""
preprocessing/create_sequences.py
==================================
Converts variable-length per-video landmark arrays (from
``generate_landmarks.py``) into fixed-length ``(sequence_length,
num_features)`` samples, applies optional data augmentation, and saves the
final ``X`` (samples) and ``y`` (labels) NumPy arrays plus a fitted label
encoder.

Augmentations implemented (each applied independently with its own
probability, per ``config.AUGMENTATION_CONFIG``):
    - Random frame skipping (temporal jitter before resampling)
    - Horizontal flip (mirror x-coordinates and swap left/right landmark pairs)
    - Rotation (2D rotation of x,y around the hip center)
    - Gaussian noise on coordinates

Usage:
    python -m preprocessing.create_sequences
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import config
from models.utils import ensure_dir, get_logger, save_pickle, set_global_seed

logger = get_logger(__name__)

# MediaPipe left/right landmark index pairs, used to swap sides on flip.
_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def resample_to_length(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a variable-length sequence to a fixed number of frames via
    linear index sampling (upsamples by repetition, downsamples by
    dropping frames).

    Args:
        sequence: Array of shape ``(num_frames, num_features)``.
        target_length: Desired output frame count.

    Returns:
        Array of shape ``(target_length, num_features)``.
    """
    if sequence.shape[0] == 0:
        return np.zeros((target_length, sequence.shape[-1]), dtype=np.float32)
    indices = np.linspace(0, sequence.shape[0] - 1, target_length).astype(int)
    return sequence[indices]


def augment_random_frame_skip(sequence: np.ndarray, skip_range: Tuple[int, int]) -> np.ndarray:
    """Drop every Nth frame (N random within ``skip_range``) before the
    caller resamples back to fixed length, simulating variable playback
    speed.

    Args:
        sequence: Array of shape ``(num_frames, num_features)``.
        skip_range: Inclusive ``(min, max)`` skip stride.

    Returns:
        A shorter sequence with some frames removed.
    """
    stride = np.random.randint(skip_range[0], skip_range[1] + 1)
    if stride <= 1 or sequence.shape[0] <= stride:
        return sequence
    return sequence[::stride]


def augment_horizontal_flip(sequence: np.ndarray) -> np.ndarray:
    """Mirror the x-coordinate of every landmark and swap left/right
    landmark pairs so the sequence represents the same motion viewed as
    if flipped horizontally.

    Args:
        sequence: Array of shape ``(num_frames, num_landmarks * 4)``.

    Returns:
        Flipped sequence of the same shape.
    """
    seq = sequence.reshape(sequence.shape[0], config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK).copy()
    seq[:, :, 0] *= -1.0  # flip x
    for left, right in _LR_PAIRS:
        seq[:, [left, right], :] = seq[:, [right, left], :]
    return seq.reshape(sequence.shape[0], -1)


def augment_rotation(sequence: np.ndarray, max_degrees: float) -> np.ndarray:
    """Apply a random small 2D rotation (around the origin, which is the
    hip center after normalization) to the x,y coordinates of every
    landmark in the sequence.

    Args:
        sequence: Array of shape ``(num_frames, num_landmarks * 4)``.
        max_degrees: Maximum absolute rotation angle in degrees.

    Returns:
        Rotated sequence of the same shape.
    """
    angle = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    seq = sequence.reshape(sequence.shape[0], config.NUM_LANDMARKS, config.COORDS_PER_LANDMARK).copy()
    xy = seq[:, :, :2]
    seq[:, :, :2] = xy @ rotation_matrix.T
    return seq.reshape(sequence.shape[0], -1)


def augment_noise(sequence: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to landmark coordinates to improve robustness.

    Args:
        sequence: Array of shape ``(num_frames, num_features)``.
        std: Standard deviation of the noise.

    Returns:
        Noisy sequence of the same shape.
    """
    noise = np.random.normal(0, std, size=sequence.shape).astype(np.float32)
    return sequence + noise


def apply_augmentations(sequence: np.ndarray) -> np.ndarray:
    """Probabilistically apply all configured augmentations to a sequence.

    Args:
        sequence: Array of shape ``(num_frames, num_features)``.

    Returns:
        Augmented sequence (same or different frame count until resampled
        by the caller).
    """
    aug_cfg = config.AUGMENTATION_CONFIG
    if not aug_cfg.enabled:
        return sequence

    if np.random.rand() < aug_cfg.random_frame_skip_prob:
        sequence = augment_random_frame_skip(sequence, aug_cfg.random_frame_skip_range)
    if np.random.rand() < aug_cfg.horizontal_flip_prob:
        sequence = augment_horizontal_flip(sequence)
    if np.random.rand() < aug_cfg.rotation_prob:
        sequence = augment_rotation(sequence, aug_cfg.rotation_max_degrees)
    if np.random.rand() < aug_cfg.noise_prob:
        sequence = augment_noise(sequence, aug_cfg.noise_std)

    return sequence


def build_dataset(
    landmarks_root: Path,
    sequence_length: int,
    augmentations_per_sample: int = 2,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load every per-video landmark ``.npy`` file, resample to fixed
    length, generate augmented copies, and assemble the final training
    arrays.

    Args:
        landmarks_root: Directory containing ``<class>/<video_id>.npy``
            files (output of ``generate_landmarks.py``).
        sequence_length: Fixed output sequence length.
        augmentations_per_sample: Number of extra augmented copies to
            generate per original sample (0 disables augmentation copies;
            the original, un-augmented sample is always kept).

    Returns:
        Tuple of ``(X, y_encoded, label_encoder)`` where ``X`` has shape
        ``(num_samples, sequence_length, num_features)`` and ``y_encoded``
        has shape ``(num_samples,)``.
    """
    class_dirs = sorted([d for d in landmarks_root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {landmarks_root}")

    X: List[np.ndarray] = []
    y: List[str] = []

    for class_dir in class_dirs:
        npy_files = sorted(class_dir.glob("*.npy"))
        logger.info("Class '%s': %d landmark files", class_dir.name, len(npy_files))

        for npy_path in tqdm(npy_files, desc=f"Sequencing {class_dir.name}"):
            raw_sequence = np.load(npy_path)
            fixed = resample_to_length(raw_sequence, sequence_length)
            X.append(fixed)
            y.append(class_dir.name)

            for _ in range(augmentations_per_sample):
                augmented = apply_augmentations(raw_sequence)
                fixed_aug = resample_to_length(augmented, sequence_length)
                X.append(fixed_aug)
                y.append(class_dir.name)

    X_arr = np.array(X, dtype=np.float32)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X_arr, y_encoded, encoder


def run(augmentations_per_sample: int = 2, seed: int = config.TRAINING_CONFIG.random_seed) -> None:
    """Build and persist the final sequence dataset (``X.npy``, ``y.npy``,
    and the fitted label encoder) to ``config.PROCESSED_DATA_DIR``.

    Args:
        augmentations_per_sample: Extra augmented copies per original
            sample.
        seed: Random seed for reproducible augmentation.
    """
    set_global_seed(seed)

    landmarks_root = Path(config.PROCESSED_DATA_DIR) / "landmarks"
    if not landmarks_root.exists():
        logger.error(
            "No landmarks found at %s. Run generate_landmarks.py first.", landmarks_root
        )
        return

    X, y_encoded, encoder = build_dataset(
        landmarks_root, config.SEQUENCE_LENGTH, augmentations_per_sample
    )

    out_dir = ensure_dir(str(Path(config.PROCESSED_DATA_DIR) / "sequences"))
    np.save(Path(out_dir) / "X.npy", X)
    np.save(Path(out_dir) / "y.npy", y_encoded)
    save_pickle(encoder, config.TRAINING_CONFIG.label_encoder_path)

    logger.info("Saved sequence dataset: X=%s, y=%s -> %s", X.shape, y_encoded.shape, out_dir)
    logger.info("Classes: %s", list(encoder.classes_))


if __name__ == "__main__":
    run()
