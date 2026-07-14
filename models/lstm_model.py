"""
models/lstm_model.py
=====================
Model architecture factory. The default and primary architecture is
MediaPipe Pose landmarks -> LSTM -> Softmax, but the factory also exposes
CNN+LSTM, 3D CNN, and MoveNet+LSTM variants so the backbone can be swapped
without touching training, inference, or the Streamlit app — every
architecture consumes/produces the same tensor shapes:

    Input:  (batch, sequence_length, num_features)   [or raw frames for conv3d]
    Output: (batch, num_classes) softmax probabilities
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import config
from config import ModelConfig
from models.utils import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Builds a compiled Keras model for a chosen architecture.

    Every ``build_*`` method returns a compiled ``tf.keras.Model`` whose
    input/output contract matches what ``training/train.py`` and
    ``models/inference.py`` expect, so swapping architectures is a one-line
    config change (``config.MODEL_CONFIG.architecture``).
    """

    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        self.cfg = cfg or config.MODEL_CONFIG

    def build(self) -> tf.keras.Model:
        """Dispatch to the architecture-specific builder named in config.

        Returns:
            A compiled ``tf.keras.Model``.

        Raises:
            ValueError: If ``self.cfg.architecture`` is not recognized.
        """
        builders = {
            "lstm": self.build_lstm,
            "cnn_lstm": self.build_cnn_lstm,
            "conv3d": self.build_conv3d,
            "movenet_lstm": self.build_movenet_lstm,
        }
        arch = self.cfg.architecture.lower()
        if arch not in builders:
            raise ValueError(
                f"Unknown architecture '{arch}'. Choose from {list(builders)}."
            )
        logger.info("Building model architecture: %s", arch)
        model = builders[arch]()
        self._compile(model)
        return model

    # ------------------------------------------------------------------ #
    # Architectures
    # ------------------------------------------------------------------ #
    def build_lstm(self) -> tf.keras.Model:
        """Primary architecture: Pose landmarks -> stacked LSTM -> Softmax.

        Matches the spec: LSTM(128) -> Dropout -> LSTM(64) -> Dense(64)
        -> Dropout -> Dense(num_classes).
        """
        cfg = self.cfg
        inputs = layers.Input(
            shape=(cfg.sequence_length, cfg.num_features), name="landmark_sequence"
        )
        x = layers.Masking(mask_value=0.0)(inputs)
        x = layers.LSTM(cfg.lstm_units_1, return_sequences=True, name="lstm_1")(x)
        x = layers.Dropout(cfg.dropout_rate, name="dropout_1")(x)
        x = layers.LSTM(cfg.lstm_units_2, return_sequences=False, name="lstm_2")(x)
        x = layers.Dense(cfg.dense_units, activation="relu", name="dense_1")(x)
        x = layers.Dropout(cfg.dropout_rate, name="dropout_2")(x)
        outputs = layers.Dense(cfg.num_classes, activation="softmax", name="predictions")(x)
        return models.Model(inputs, outputs, name="pose_lstm")

    def build_cnn_lstm(self) -> tf.keras.Model:
        """Alternative architecture: 1D-CNN feature extractor over the
        landmark sequence, feeding into an LSTM temporal head.
        """
        cfg = self.cfg
        inputs = layers.Input(
            shape=(cfg.sequence_length, cfg.num_features), name="landmark_sequence"
        )
        x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
        x = layers.LSTM(cfg.lstm_units_1, return_sequences=True)(x)
        x = layers.Dropout(cfg.dropout_rate)(x)
        x = layers.LSTM(cfg.lstm_units_2)(x)
        x = layers.Dense(cfg.dense_units, activation="relu")(x)
        x = layers.Dropout(cfg.dropout_rate)(x)
        outputs = layers.Dense(cfg.num_classes, activation="softmax")(x)
        return models.Model(inputs, outputs, name="cnn_lstm")

    def build_conv3d(self, frame_height: int = 64, frame_width: int = 64) -> tf.keras.Model:
        """Alternative architecture: 3D CNN operating directly on raw
        video frame stacks instead of pose landmarks.

        Note: this architecture expects raw frame tensors
        ``(sequence_length, H, W, 3)`` rather than landmark vectors, so it
        requires a different preprocessing path (skip pose extraction and
        stack resized RGB frames instead).
        """
        cfg = self.cfg
        inputs = layers.Input(
            shape=(cfg.sequence_length, frame_height, frame_width, 3), name="frame_sequence"
        )
        x = layers.Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu")(inputs)
        x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
        x = layers.Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = layers.Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(cfg.dense_units, activation="relu")(x)
        x = layers.Dropout(cfg.dropout_rate)(x)
        outputs = layers.Dense(cfg.num_classes, activation="softmax")(x)
        return models.Model(inputs, outputs, name="conv3d")

    def build_movenet_lstm(self) -> tf.keras.Model:
        """Alternative architecture: identical head to ``build_lstm`` but
        intended to consume MoveNet keypoints (17 points x 3 values)
        instead of MediaPipe's 33 points x 4 values. Swap the extractor in
        ``models/mediapipe_extractor.py`` for a MoveNet-based one and point
        ``config.MODEL_CONFIG.num_features`` at ``17 * 3`` to use this.
        """
        cfg = self.cfg
        inputs = layers.Input(
            shape=(cfg.sequence_length, cfg.num_features), name="keypoint_sequence"
        )
        x = layers.Masking(mask_value=0.0)(inputs)
        x = layers.Bidirectional(layers.LSTM(cfg.lstm_units_1, return_sequences=True))(x)
        x = layers.Dropout(cfg.dropout_rate)(x)
        x = layers.LSTM(cfg.lstm_units_2)(x)
        x = layers.Dense(cfg.dense_units, activation="relu")(x)
        x = layers.Dropout(cfg.dropout_rate)(x)
        outputs = layers.Dense(cfg.num_classes, activation="softmax")(x)
        return models.Model(inputs, outputs, name="movenet_lstm")

    # ------------------------------------------------------------------ #
    # Compilation
    # ------------------------------------------------------------------ #
    def _compile(self, model: tf.keras.Model) -> None:
        cfg = self.cfg
        if cfg.optimizer.lower() == "adam":
            optimizer = optimizers.Adam(learning_rate=cfg.learning_rate)
        elif cfg.optimizer.lower() == "sgd":
            optimizer = optimizers.SGD(learning_rate=cfg.learning_rate, momentum=0.9)
        elif cfg.optimizer.lower() == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=cfg.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

        model.compile(
            optimizer=optimizer,
            loss=cfg.loss,
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
        )


def build_model(cfg: Optional[ModelConfig] = None) -> tf.keras.Model:
    """Convenience function: build and compile a model from config.

    Args:
        cfg: Optional model configuration. Defaults to
            ``config.MODEL_CONFIG``.

    Returns:
        A compiled ``tf.keras.Model``.
    """
    return ModelFactory(cfg).build()
