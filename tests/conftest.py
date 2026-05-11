"""
@file conftest.py
@brief Shared pytest fixtures.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """
    @brief Build a textured RGB test image.
    @return RGB uint8 image.
    """
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.rectangle(image, (24, 24), (108, 108), (230, 230, 230), -1)
    cv2.circle(image, (66, 66), 28, (20, 90, 180), -1)
    cv2.line(image, (20, 180), (220, 40), (255, 255, 255), 3)
    noise = np.random.default_rng(42).integers(0, 60, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


@pytest.fixture
def blank_rgb_image() -> np.ndarray:
    """
    @brief Build a low-feature RGB image.
    @return RGB uint8 image.
    """
    return np.full((128, 128, 3), 127, dtype=np.uint8)


@pytest.fixture
def copy_move_image(sample_rgb_image: np.ndarray) -> np.ndarray:
    """
    @brief Build a simple synthetic copy-move image.
    @return RGB uint8 image.
    """
    image = sample_rgb_image.copy()
    patch = image[24:108, 24:108].copy()
    image[140:224, 140:224] = patch
    return image


@pytest.fixture
def png_bytes(sample_rgb_image: np.ndarray) -> bytes:
    """
    @brief Encode an RGB image as PNG bytes.
    @return PNG payload.
    """
    buffer = io.BytesIO()
    Image.fromarray(sample_rgb_image).save(buffer, format="PNG")
    return buffer.getvalue()
