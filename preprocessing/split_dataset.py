"""
preprocessing/split_dataset.py
===============================
Splits the assembled sequence dataset (``X.npy``, ``y.npy`` produced by
``create_sequences.py``) into stratified train/validation/test sets and
saves each split as ``.npy`` files under ``config.TRAIN_DATA_DIR``,
``config.VAL_DATA_DIR``, and ``config.TEST_DATA_DIR``.

Usage:
    python -m preprocessing.split_dataset
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import config
from models.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def run(
    val_split: float = config.TRAINING_CONFIG.validation_split,
    test_split: float = config.TRAINING_CONFIG.test_split,
    seed: int = config.TRAINING_CONFIG.random_seed,
) -> None:
    """Load the full sequence dataset and write stratified train/val/test
    splits to disk.

    Args:
        val_split: Fraction of the full dataset reserved for validation.
        test_split: Fraction of the full dataset reserved for testing.
        seed: Random seed for reproducible splitting.
    """
    seq_dir = Path(config.PROCESSED_DATA_DIR) / "sequences"
    X_path, y_path = seq_dir / "X.npy", seq_dir / "y.npy"

    if not (X_path.exists() and y_path.exists()):
        logger.error("Sequence files not found at %s. Run create_sequences.py first.", seq_dir)
        return

    X = np.load(X_path)
    y = np.load(y_path)

    # First split off the test set, then split remaining into train/val.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    relative_val = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val, random_state=seed, stratify=y_trainval
    )

    _save_split(config.TRAIN_DATA_DIR, X_train, y_train, "train")
    _save_split(config.VAL_DATA_DIR, X_val, y_val, "val")
    _save_split(config.TEST_DATA_DIR, X_test, y_test, "test")

    logger.info(
        "Split complete -> train=%d, val=%d, test=%d",
        len(X_train), len(X_val), len(X_test),
    )


def _save_split(directory: str, X: np.ndarray, y: np.ndarray, name: str) -> None:
    """Save one split's ``X``/``y`` arrays to a directory.

    Args:
        directory: Destination directory.
        X: Feature array for this split.
        y: Label array for this split.
        name: Split name, used only for logging.
    """
    ensure_dir(directory)
    np.save(Path(directory) / "X.npy", X)
    np.save(Path(directory) / "y.npy", y)
    logger.info("Saved %s split (%d samples) -> %s", name, len(X), directory)


if __name__ == "__main__":
    run()
