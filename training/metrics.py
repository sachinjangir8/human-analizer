"""
training/metrics.py
====================
Computes and plots classification metrics: accuracy, precision, recall,
F1, confusion matrix, classification report, and ROC curves (one-vs-rest).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

import config
from models.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    """Compute the standard classification metric set.

    Args:
        y_true: Ground-truth integer class labels, shape ``(n_samples,)``.
        y_pred: Predicted integer class labels, shape ``(n_samples,)``.
        class_names: Ordered class name list matching label indices.

    Returns:
        Dictionary with ``accuracy``, ``precision``, ``recall``, ``f1``
        (all macro-averaged), and the full ``classification_report`` text.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: str
) -> None:
    """Render and save a heatmap confusion matrix.

    Args:
        y_true: Ground-truth integer class labels.
        y_pred: Predicted integer class labels.
        class_names: Ordered class name list.
        save_path: PNG output path.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    ensure_dir(str(Path(save_path).parent))
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix -> %s", save_path)


def plot_roc_curves(
    y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str], save_path: str
) -> None:
    """Render and save one-vs-rest ROC curves for every class.

    Args:
        y_true: Ground-truth integer class labels.
        y_probs: Predicted probability matrix, shape
            ``(n_samples, n_classes)``.
        class_names: Ordered class name list.
        save_path: PNG output path.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(9, 7))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    ensure_dir(str(Path(save_path).parent))
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved ROC curves -> %s", save_path)


def plot_training_curves(history: dict, output_dir: str) -> None:
    """Plot and save accuracy and loss curves from a Keras training
    ``history.history`` dict.

    Args:
        history: The ``.history`` dict from a fitted Keras model.
        output_dir: Directory to save ``accuracy.png`` and ``loss.png``
            into.
    """
    ensure_dir(output_dir)

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("accuracy", []), label="Train Accuracy")
    plt.plot(history.get("val_accuracy", []), label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.get("loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "loss.png", dpi=150)
    plt.close()

    logger.info("Saved training curve plots -> %s", output_dir)


def save_metrics_report(metrics: Dict, save_path: str) -> None:
    """Persist a metrics dict to a JSON report file.

    Args:
        metrics: Dict returned by :func:`compute_metrics`.
        save_path: JSON output path.
    """
    ensure_dir(str(Path(save_path).parent))
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics report -> %s", save_path)
