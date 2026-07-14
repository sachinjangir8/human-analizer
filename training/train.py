"""
training/train.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from models.lstm_model import build_model
from models.utils import get_logger, set_global_seed
from training.callbacks import build_callbacks

logger = get_logger(__name__)


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        logger.info("No GPU detected; training on CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    logger.info("Detected %d GPU(s).", len(gpus))


def load_split(split_dir):
    X = np.load(Path(split_dir) / "X.npy")
    y = np.load(Path(split_dir) / "y.npy")
    return X, y


def run(epochs, batch_size, learning_rate, resume_path):

    set_global_seed(config.TRAINING_CONFIG.random_seed)
    configure_gpu()

    logger.info("Loading dataset...")

    X_train, y_train = load_split(config.TRAIN_DATA_DIR)
    X_val, y_val = load_split(config.VAL_DATA_DIR)

    logger.info("Train shape : %s", X_train.shape)
    logger.info("Val shape   : %s", X_val.shape)

    logger.info("Train labels min=%d max=%d", y_train.min(), y_train.max())
    logger.info("Val labels   min=%d max=%d", y_val.min(), y_val.max())

    num_classes = len(np.unique(y_train))

    logger.info("Detected %d classes", num_classes)

    if y_train.max() >= num_classes:
        raise ValueError(
            f"Invalid labels detected. "
            f"Max label={y_train.max()} "
            f"Detected classes={num_classes}"
        )

    # Keep model configuration synchronized
    config.MODEL_CONFIG.num_classes = num_classes

    y_train = tf.keras.utils.to_categorical(
        y_train,
        num_classes=num_classes,
    )

    y_val = tf.keras.utils.to_categorical(
        y_val,
        num_classes=num_classes,
    )

    if resume_path:

        logger.info("Loading checkpoint %s", resume_path)

        model = tf.keras.models.load_model(resume_path)

        tf.keras.backend.set_value(
            model.optimizer.learning_rate,
            learning_rate,
        )

    else:

        config.MODEL_CONFIG.learning_rate = learning_rate

        model = build_model()

    model.summary(print_fn=logger.info)

    callbacks = build_callbacks()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    with open(config.TRAINING_CONFIG.history_path, "w") as f:
        json.dump(history.history, f, indent=4)

    logger.info("History saved.")

    from training.metrics import plot_training_curves

    plot_training_curves(
        history.history,
        str(config.OUTPUTS_DIR),
    )

    logger.info("Training Finished.")
    logger.info("Best model -> %s", config.TRAINING_CONFIG.checkpoint_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.TRAINING_CONFIG.epochs,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.TRAINING_CONFIG.batch_size,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=config.TRAINING_CONFIG.learning_rate,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    run(
        args.epochs,
        args.batch_size,
        args.lr,
        args.resume,
    )