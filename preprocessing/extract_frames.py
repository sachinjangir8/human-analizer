"""
preprocessing/extract_frames.py
================================
Walks the raw dataset directory (organized as one subfolder per activity
class, each containing video files) and extracts evenly-sampled frames
from every video into config.PROCESSED_DATA_DIR/frames/<class>/<video_id>/.

Usage:
python preprocessing/extract_frames.py --input data/raw/UCF-101 --frames-per-video 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# -------------------------------------------------------
# Add project root to Python path (FIXES import errors)
# -------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from tqdm import tqdm

import config
from models.utils import ensure_dir, get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def list_videos(class_dir: Path) -> List[Path]:
    """Return all video files under a class directory."""
    return sorted(
        p
        for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frames_per_video: int,
) -> int:
    """
    Extract evenly spaced frames from one video.
    """

    ensure_dir(str(output_dir))

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.warning("Cannot open %s", video_path)
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        logger.warning("No frames in %s", video_path)
        cap.release()
        return 0

    sample_indices = np.linspace(
        0,
        total_frames - 1,
        min(frames_per_video, total_frames),
        dtype=int,
    )

    sample_indices = set(sample_indices.tolist())

    written = 0
    current = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current in sample_indices:
            frame_path = output_dir / f"frame_{written:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            written += 1

        current += 1

    cap.release()

    return written


def run(
    input_dir: str,
    frames_per_video: int,
):
    input_root = Path(input_dir)

    if not input_root.exists():
        logger.error("Dataset not found: %s", input_root)
        return

    output_root = Path(config.PROCESSED_DATA_DIR) / "frames"

    ensure_dir(str(output_root))

    class_dirs = sorted(
        d for d in input_root.iterdir() if d.is_dir()
    )

    if len(class_dirs) == 0:
        logger.error("No activity folders found.")
        return

    logger.info("Found %d activity classes", len(class_dirs))

    for class_dir in class_dirs:

        videos = list_videos(class_dir)

        logger.info(
            "%s : %d videos",
            class_dir.name,
            len(videos),
        )

        for video in tqdm(
            videos,
            desc=class_dir.name,
        ):

            out_dir = (
                output_root
                / class_dir.name
                / video.stem
            )

            extract_frames_from_video(
                video,
                out_dir,
                frames_per_video,
            )

    logger.info("Finished extracting all frames.")
    logger.info("Saved to %s", output_root)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Dataset folder",
    )

    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=60,
    )

    args = parser.parse_args()

    run(
        args.input,
        args.frames_per_video,
    )