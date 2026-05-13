"""
@file test_utils_and_augmentation.py
@brief Coverage tests for utility metrics, visualization, and augmentation helpers.
"""

from __future__ import annotations

import numpy as np

from src.preprocessing.augmentation import (
    adjust_brightness,
    augment_for_training,
    random_horizontal_flip,
    rotate,
)
from src.utils.metrics import binary_accuracy, f1_score_binary, mask_iou
from src.utils.visualization import heatmap_to_overlay, overlay_mask


class FixedRng:
    def __init__(self, random_value: float, uniform_values: list[float] | None = None) -> None:
        self.random_value = random_value
        self.uniform_values = uniform_values or []

    def random(self) -> float:
        return self.random_value

    def uniform(self, _low: float, _high: float) -> float:
        return self.uniform_values.pop(0)


def test_metrics_handle_normal_and_empty_cases() -> None:
    assert binary_accuracy(np.array([0, 1, 1]), np.array([0.1, 0.7, 0.2])) == 2 / 3
    assert binary_accuracy(np.array([]), np.array([])) == 0.0
    assert f1_score_binary(np.array([1, 0, 1]), np.array([0.9, 0.7, 0.1])) == 0.5
    assert f1_score_binary(np.array([0, 0]), np.array([0.1, 0.2])) == 0.0
    assert mask_iou(np.array([[1, 0], [1, 0]]), np.array([[0.8, 0.2], [0.8, 0.9]])) == 2 / 3
    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0


def test_visualization_helpers_resize_and_overlay() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    masked = overlay_mask(image, mask, color=(10, 20, 30), alpha=0.5)
    assert masked.shape == image.shape
    assert masked.max() > 0

    heatmap = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    overlay = heatmap_to_overlay(image.astype(np.float32), heatmap)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_augmentation_helpers_cover_flip_rotate_and_brightness() -> None:
    image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)

    flipped = random_horizontal_flip(image, FixedRng(0.1))
    assert np.array_equal(flipped, image[:, ::-1])
    assert random_horizontal_flip(image, FixedRng(0.9)) is image

    rotated = rotate(image, 10.0)
    assert rotated.shape == image.shape

    brighter_uint8 = adjust_brightness(image, 1.5)
    assert brighter_uint8.dtype == np.uint8
    assert brighter_uint8.max() >= image.max()

    float_image = image.astype(np.float32) / 255.0
    darker_float = adjust_brightness(float_image, 0.5)
    assert darker_float.dtype == np.float32
    assert darker_float.max() <= float_image.max()

    augmented = augment_for_training(image, FixedRng(0.1, [0.0, 1.0]))
    assert augmented.shape == image.shape
