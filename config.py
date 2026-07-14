"""
config.py
=========
Central configuration for the Human Activity Recognition (HAR) system.

All paths, hyperparameters, and class definitions live here so that every
other module (preprocessing, training, inference, deployment) stays in sync
by importing a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# --------------------------------------------------------------------------- #
# Project root & directory layout
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = DATA_DIR / "train"
VAL_DATA_DIR = DATA_DIR / "val"
TEST_DATA_DIR = DATA_DIR / "test"

MODELS_DIR = ROOT_DIR / "models"
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"

for _d in (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR,
    SAVED_MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, LOGS_DIR, TENSORBOARD_DIR,
):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------- #
# Activity classes
# --------------------------------------------------------------------------- #
# Extend this list to add new activities. The rest of the pipeline (model
# output layer, label encoder, UI) reads from this list dynamically, so
# adding a class here + re-running preprocessing/training is enough.
# --------------------------------------------------------------------------- #
# Activity classes (UCF101)
# --------------------------------------------------------------------------- #

ACTIVITY_CLASSES: List[str] = [f"class_{i}" for i in range(101)]

NUM_CLASSES: int = len(ACTIVITY_CLASSES)


# --------------------------------------------------------------------------- #
# Pose / landmark configuration (MediaPipe Pose)
# --------------------------------------------------------------------------- #
NUM_LANDMARKS: int = 33          # MediaPipe Pose landmark count
COORDS_PER_LANDMARK: int = 4     # x, y, z, visibility
FEATURES_PER_FRAME: int = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 132

POSE_MIN_DETECTION_CONFIDENCE: float = 0.5
POSE_MIN_TRACKING_CONFIDENCE: float = 0.5
POSE_MODEL_COMPLEXITY: int = 1  # 0=lite, 1=full, 2=heavy


# --------------------------------------------------------------------------- #
# Sequence configuration
# --------------------------------------------------------------------------- #
SEQUENCE_LENGTH: int = 30        # number of frames per sample sequence
FRAME_SKIP: int = 1              # sample every Nth frame during extraction


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    architecture: str = "lstm"          # one of: lstm, cnn_lstm, conv3d, movenet_lstm
    sequence_length: int = SEQUENCE_LENGTH
    num_features: int = FEATURES_PER_FRAME
    num_classes: int = NUM_CLASSES

    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dense_units: int = 64
    dropout_rate: float = 0.3

    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss: str = "categorical_crossentropy"


MODEL_CONFIG = ModelConfig()


# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    validation_split: float = 0.15
    test_split: float = 0.15

    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6

    checkpoint_path: str = str(SAVED_MODELS_DIR / "best_model.keras")
    label_encoder_path: str = str(SAVED_MODELS_DIR / "label_encoder.pkl")
    history_path: str = str(REPORTS_DIR / "history.json")

    use_tensorboard: bool = True
    tensorboard_log_dir: str = str(TENSORBOARD_DIR)

    random_seed: int = 42


TRAINING_CONFIG = TrainingConfig()


# --------------------------------------------------------------------------- #
# Data augmentation configuration
# --------------------------------------------------------------------------- #
@dataclass
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_prob: float = 0.3
    rotation_max_degrees: float = 15.0
    noise_prob: float = 0.3
    noise_std: float = 0.01
    random_frame_skip_prob: float = 0.3
    random_frame_skip_range: tuple = (1, 3)


AUGMENTATION_CONFIG = AugmentationConfig()


# --------------------------------------------------------------------------- #
# Dataset download configuration
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    name: str = "UCF101"  # or "HMDB51"
    ucf101_url: str = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    hmdb51_url: str = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    download_dir: str = str(RAW_DATA_DIR)
    # Map our activity classes to source-dataset class names. UCF101 / HMDB51
    # naming differs from our friendly class names, so we keep a lookup
    # table that preprocessing/download_dataset.py uses to filter classes.
    ucf101_class_map: dict = field(default_factory=lambda: {
        "Walking": "WalkingWithDog",
        "Running": "Running",  # not native to UCF101; combine/relabel as needed
        "Jumping": "JumpRope",
        "Clapping": "HandstandPushups",  # placeholder — replace with a true match
        "Waving": "Punch",  # placeholder — replace with a true match
    })


DATASET_CONFIG = DatasetConfig()


# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE: str = str(LOGS_DIR / "har.log")


# --------------------------------------------------------------------------- #
# Streamlit / deployment configuration
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    page_title: str = "AI Human Activity Recognition"
    page_icon: str = "🏃"
    layout: str = "wide"
    default_confidence_threshold: float = 0.6
    webcam_fps_target: int = 15
    max_upload_size_mb: int = 200


APP_CONFIG = AppConfig()
