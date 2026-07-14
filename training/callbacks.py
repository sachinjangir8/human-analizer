"""
training/callbacks.py
======================
Builds the list of Keras callbacks used during training: early stopping,
LR reduction on plateau, model checkpointing, TensorBoard logging, and a
custom epoch-time logger.
"""

from __future__ import annotations

import time
from typing import List

import tensorflow as tf

import config
from config import TrainingConfig
from models.utils import get_logger

logger = get_logger(__name__)


class EpochTimeLogger(tf.keras.callbacks.Callback):
    """Logs wall-clock duration of each training epoch via the project
    logger, useful for spotting I/O or augmentation bottlenecks.
    """

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        duration = time.time() - self._epoch_start
        logs = logs or {}
        logger.info(
            "Epoch %d finished in %.1fs | loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1, duration,
            logs.get("loss", float("nan")), logs.get("accuracy", float("nan")),
            logs.get("val_loss", float("nan")), logs.get("val_accuracy", float("nan")),
        )


def build_callbacks(cfg: TrainingConfig = config.TRAINING_CONFIG) -> List[tf.keras.callbacks.Callback]:
    """Assemble the standard callback list for training.

    Args:
        cfg: Training configuration providing paths/patience values.

    Returns:
        List of instantiated Keras callbacks:
        ``EarlyStopping``, ``ReduceLROnPlateau``, ``ModelCheckpoint``,
        optionally ``TensorBoard``, and ``EpochTimeLogger``.
    """
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=cfg.checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EpochTimeLogger(),
    ]

    if cfg.use_tensorboard:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=cfg.tensorboard_log_dir, histogram_freq=1)
        )

    return callbacks
