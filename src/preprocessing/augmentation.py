"""
@file augmentation.py
@brief Lightweight image augmentation helpers for optional training workflows.
"""

from __future__ import annotations

import cv2
import numpy as np


def random_horizontal_flip(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    @brief Randomly flip an image horizontally.
    @param image RGB image.
    @param rng Optional numpy random generator.
    @return Original or flipped image.
    """
    generator = rng or np.random.default_rng()
    if generator.random() < 0.5:
        return np.ascontiguousarray(image[:, ::-1])
    return image


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    """
    @brief Rotate an image around its center.
    @param image RGB image.
    @param degrees Rotation angle in degrees.
    @return Rotated image with original dimensions.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    @brief Adjust image brightness by multiplication.
    @param image RGB image in uint8 or normalized float.
    @param factor Brightness factor where 1.0 preserves the image.
    @return Brightness-adjusted image with the same dtype.
    """
    if image.dtype == np.uint8:
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return np.clip(image * factor, 0.0, 1.0).astype(image.dtype)


def augment_for_training(
    image: np.ndarray,
    rng: np.random.Generator | None = None,
    max_rotation_degrees: float = 15.0,
    brightness_delta: float = 0.20,
) -> np.ndarray:
    """
    @brief Apply the documented flip, rotation, and brightness augmentation.
    @param image RGB image.
    @param rng Optional numpy random generator.
    @param max_rotation_degrees Maximum absolute rotation angle.
    @param brightness_delta Maximum brightness change ratio.
    @return Augmented image.
    """
    generator = rng or np.random.default_rng()
    output = random_horizontal_flip(image, generator)
    angle = float(generator.uniform(-max_rotation_degrees, max_rotation_degrees))
    output = rotate(output, angle)
    factor = float(generator.uniform(1.0 - brightness_delta, 1.0 + brightness_delta))
    return adjust_brightness(output, factor)
