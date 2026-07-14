"""
preprocessing/download_dataset.py
==================================
Downloads UCF101 or HMDB51, extracts the archive, and filters it down to
only the class folders relevant to ``config.ACTIVITY_CLASSES`` (via the
mapping in ``config.DATASET_CONFIG``).

Usage:
    python -m preprocessing.download_dataset --dataset ucf101
    python -m preprocessing.download_dataset --dataset hmdb51 --classes Walking Running

Notes:
    - These archives are several GB (UCF101 ~6.9GB, HMDB51 ~2GB) and the
      source servers are occasionally slow/unavailable. Run this on a
      machine with a real internet connection and disk space; it is not
      executed automatically by any other script.
    - Extracting .rar files requires ``unrar`` or ``rarfile`` + a
      system rar binary to be installed.
    - If you already have the dataset locally, skip this script and point
      ``config.RAW_DATA_DIR`` at your existing folder instead.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm

import config
from models.utils import get_logger

logger = get_logger(__name__)


def download_file(url: str, dest_path: str, chunk_size: int = 1 << 20) -> str:
    """Stream-download a large file with a progress bar, resuming is not
    supported (re-run from scratch if interrupted).

    Args:
        url: Direct download URL.
        dest_path: Local file path to save to.
        chunk_size: Bytes per read chunk.

    Returns:
        The path the file was saved to.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        logger.info("Downloading %s (%.1f MB) -> %s", url, total / 1e6, dest_path)

        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return dest_path


def extract_archive(archive_path: str, extract_to: str) -> None:
    """Extract a .rar or .zip archive to a directory.

    Args:
        archive_path: Path to the downloaded archive.
        extract_to: Destination directory.

    Raises:
        RuntimeError: If the archive format is unsupported or extraction
            tooling is missing.
    """
    os.makedirs(extract_to, exist_ok=True)
    suffix = Path(archive_path).suffix.lower()

    if suffix == ".zip":
        shutil.unpack_archive(archive_path, extract_to)
        return

    if suffix == ".rar":
        try:
            import rarfile
        except ImportError as exc:
            raise RuntimeError(
                "Extracting .rar files requires `pip install rarfile` plus a "
                "system `unrar` binary. Install both and re-run."
            ) from exc
        with rarfile.RarFile(archive_path) as rf:
            rf.extractall(extract_to)
        return

    raise RuntimeError(f"Unsupported archive format: {suffix}")


def filter_classes(
    extracted_dir: str, target_classes: List[str], class_map: dict
) -> None:
    """Remove class folders from the extracted dataset that are not needed,
    keeping disk usage down to only what ``config.ACTIVITY_CLASSES``
    requires.

    Args:
        extracted_dir: Root directory of the extracted dataset (contains
            one subfolder per source-dataset class).
        target_classes: Our friendly activity class names.
        class_map: Mapping from friendly class name -> source dataset
            class folder name (see ``config.DATASET_CONFIG``).
    """
    wanted_folders = {class_map.get(c, c) for c in target_classes}
    root = Path(extracted_dir)

    if not root.exists():
        logger.warning("Extracted directory does not exist: %s", extracted_dir)
        return

    for child in root.iterdir():
        if child.is_dir() and child.name not in wanted_folders:
            logger.info("Removing unused class folder: %s", child)
            shutil.rmtree(child, ignore_errors=True)


def main(dataset: str, classes: Optional[List[str]], keep_archive: bool) -> None:
    """Entry point: download, extract, and filter the requested dataset.

    Args:
        dataset: ``"ucf101"`` or ``"hmdb51"``.
        classes: Subset of ``config.ACTIVITY_CLASSES`` to keep. Defaults
            to all configured classes.
        keep_archive: If False, delete the downloaded archive after
            extraction to save disk space.
    """
    ds_cfg = config.DATASET_CONFIG
    classes = classes or config.ACTIVITY_CLASSES

    if dataset.lower() == "ucf101":
        url = ds_cfg.ucf101_url
        class_map = ds_cfg.ucf101_class_map
    elif dataset.lower() == "hmdb51":
        url = ds_cfg.hmdb51_url
        class_map = {}  # populate as needed for HMDB51 naming
    else:
        raise ValueError("dataset must be 'ucf101' or 'hmdb51'")

    archive_path = os.path.join(ds_cfg.download_dir, os.path.basename(url))
    extract_dir = os.path.join(ds_cfg.download_dir, dataset.lower())

    logger.info("Starting dataset download: %s", dataset)
    download_file(url, archive_path)
    extract_archive(archive_path, extract_dir)
    filter_classes(extract_dir, classes, class_map)

    if not keep_archive:
        os.remove(archive_path)
        logger.info("Removed archive to save space: %s", archive_path)

    logger.info("Dataset ready at: %s", extract_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare UCF101/HMDB51 subsets")
    parser.add_argument("--dataset", choices=["ucf101", "hmdb51"], default="ucf101")
    parser.add_argument("--classes", nargs="*", default=None, help="Subset of activity classes to keep")
    parser.add_argument("--keep-archive", action="store_true", help="Do not delete the archive after extraction")
    args = parser.parse_args()

    main(args.dataset, args.classes, args.keep_archive)
