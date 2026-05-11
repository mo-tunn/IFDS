"""
@file metrics.py
@brief Metric helpers for classification and localization evaluation.
"""

from __future__ import annotations

import numpy as np


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    @brief Compute binary accuracy for probabilities.
    @param y_true Ground-truth labels.
    @param y_pred Predicted probabilities.
    @param threshold Positive-class threshold.
    @return Accuracy in [0, 1].
    """
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred) >= threshold
    if truth.size == 0:
        return 0.0
    return float(np.mean(truth == pred))


def f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    @brief Compute binary F1 score without external dependencies.
    @param y_true Ground-truth labels.
    @param y_pred Predicted probabilities.
    @param threshold Positive-class threshold.
    @return F1 score in [0, 1].
    """
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred) >= threshold
    true_positive = np.sum(truth & pred)
    false_positive = np.sum(~truth & pred)
    false_negative = np.sum(truth & ~pred)
    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return float((2 * true_positive) / denominator)


def mask_iou(mask_true: np.ndarray, mask_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    @brief Compute intersection-over-union for binary masks.
    @param mask_true Ground-truth mask.
    @param mask_pred Predicted mask or heatmap.
    @param threshold Threshold for predicted mask.
    @return IoU in [0, 1].
    """
    truth = np.asarray(mask_true).astype(bool)
    pred = np.asarray(mask_pred) >= threshold
    intersection = np.logical_and(truth, pred).sum()
    union = np.logical_or(truth, pred).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)
