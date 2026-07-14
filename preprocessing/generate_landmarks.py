"""
preprocessing/generate_landmarks.py
====================================
Runs MediaPipe Pose over every extracted-frame folder (produced by
``extract_frames.py``) and saves one ``.npy`` landmark-sequence file per
video into ``config.PROCESSED_DATA_DIR/landmarks/<class>/<video_id>.npy``.

Supports resuming:
- If a landmark .npy file already exists for a video, it is skipped.
- This allows restarting the script without reprocessing completed videos.

Usage:
    python -m preprocessing.generate_landmarks
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from tqdm import tqdm

import config
from models.mediapipe_extractor import PoseExtractor
from models.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def process_video_folder(extractor: PoseExtractor, frame_dir: Path) -> np.ndarray:
    """
    Run pose extraction over every JPEG frame in a folder.

    Args:
        extractor: Shared PoseExtractor instance.
        frame_dir: Folder containing extracted frames.

    Returns:
        Landmark sequence of shape:
        (num_frames, config.FEATURES_PER_FRAME)
    """
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))

    if not frame_paths:
        return np.zeros((0, config.FEATURES_PER_FRAME), dtype=np.float32)

    sequence = []

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))

        if frame is None:
            continue

        landmarks = extractor.extract_normalized_landmarks(frame)
        sequence.append(landmarks)

    return np.asarray(sequence, dtype=np.float32)


def run() -> None:
    """
    Convert extracted frames into MediaPipe landmark sequences.

    Resume support:
    Existing landmark files (.npy) are skipped automatically.
    """

    frames_root = Path(config.PROCESSED_DATA_DIR) / "frames"
    landmarks_root = Path(config.PROCESSED_DATA_DIR) / "landmarks"

    if not frames_root.exists():
        logger.error(
            "No extracted frames found at %s. "
            "Run extract_frames.py first.",
            frames_root,
        )
        return

    class_dirs = sorted(
        [d for d in frames_root.iterdir() if d.is_dir()]
    )

    total_processed = 0
    total_skipped = 0

    with PoseExtractor() as extractor:

        for class_dir in class_dirs:

            out_class_dir = Path(
                ensure_dir(str(landmarks_root / class_dir.name))
            )

            video_dirs = sorted(
                [d for d in class_dir.iterdir() if d.is_dir()]
            )

            progress = tqdm(
                video_dirs,
                desc=f"Landmarks: {class_dir.name}",
                unit="video",
            )

            for video_dir in progress:

                out_path = out_class_dir / f"{video_dir.name}.npy"

                # Resume support
                if out_path.exists():
                    total_skipped += 1
                    progress.set_postfix(skipped=total_skipped)
                    continue

                sequence = process_video_folder(extractor, video_dir)

                if sequence.shape[0] == 0:
                    logger.warning("Skipping empty sequence: %s", video_dir)
                    continue

                np.save(out_path, sequence)

                total_processed += 1

                progress.set_postfix(
                    processed=total_processed,
                    skipped=total_skipped,
                )

    logger.info("=" * 60)
    logger.info("Landmark generation completed.")
    logger.info("Processed : %d videos", total_processed)
    logger.info("Skipped   : %d videos", total_skipped)
    logger.info("Output    : %s", landmarks_root)
    logger.info("=" * 60)


if __name__ == "__main__":
    run()