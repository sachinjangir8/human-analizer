"""
training/evaluate.py
=====================
Evaluates a trained model on the held-out test split: computes accuracy,
precision, recall, F1, confusion matrix, classification report, and ROC
curves, saving all plots/reports to ``config.OUTPUTS_DIR``.

Usage:
    python -m training.evaluate
    python -m training.evaluate --model saved_models/best_model.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
from models.utils import get_logger, load_pickle
from training.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    save_metrics_report,
)
from training.train import load_split

logger = get_logger(__name__)


def run(model_path: str, label_encoder_path: str) -> None:
    """Load the test split and a trained model, run predictions, and
    generate the full evaluation report.

    Args:
        model_path: Path to a saved ``.keras`` model.
        label_encoder_path: Path to the fitted ``LabelEncoder`` pickle.
    """
    logger.info("Loading test split...")
    X_test, y_test = load_split(config.TEST_DATA_DIR)

    logger.info("Loading model from %s", model_path)
    model = tf.keras.models.load_model(model_path)

    encoder = load_pickle(label_encoder_path)
    class_names = list(encoder.classes_)

    logger.info("Running predictions on %d test samples...", len(X_test))
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    metrics = compute_metrics(y_test, y_pred, class_names)
    logger.info("Test accuracy: %.4f", metrics["accuracy"])
    logger.info("\n%s", metrics["classification_report"])

    save_metrics_report(metrics, str(Path(config.REPORTS_DIR) / "test_metrics.json"))
    plot_confusion_matrix(
        y_test, y_pred, class_names, str(Path(config.OUTPUTS_DIR) / "confusion_matrix.png")
    )
    plot_roc_curves(
        y_test, y_probs, class_names, str(Path(config.REPORTS_DIR) / "roc_curves.png")
    )

    logger.info("Evaluation complete. Reports saved under %s", config.OUTPUTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained HAR model on the test split")
    parser.add_argument("--model", type=str, default=config.TRAINING_CONFIG.checkpoint_path)
    parser.add_argument("--label-encoder", type=str, default=config.TRAINING_CONFIG.label_encoder_path)
    args = parser.parse_args()

    run(args.model, args.label_encoder)
