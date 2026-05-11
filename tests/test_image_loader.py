"""
@file test_image_loader.py
@brief Tests for image loading and preprocessing.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

import src.preprocessing.image_loader as image_loader_module
from src.preprocessing.image_loader import ImageLoader


def test_load_from_bytes_png_returns_rgb_and_metadata(png_bytes: bytes) -> None:
    image, metadata = ImageLoader.load_from_bytes(png_bytes, "sample.png")

    assert image.shape == (256, 256, 3)
    assert image.dtype == np.uint8
    assert metadata["format"] == "PNG"
    assert metadata["width"] == 256
    assert metadata["height"] == 256


def test_load_from_bytes_rejects_unsupported_extension(png_bytes: bytes) -> None:
    with pytest.raises(ValueError, match="Desteklenmeyen format"):
        ImageLoader.load_from_bytes(png_bytes, "sample.txt")


def test_load_from_bytes_rejects_oversized_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_loader_module, "MAX_FILE_SIZE_BYTES", 4)
    with pytest.raises(ValueError, match="Dosya çok büyük"):
        ImageLoader.load_from_bytes(b"12345", "sample.png")


def test_load_from_bytes_rejects_corrupt_image() -> None:
    with pytest.raises(ValueError, match="Görüntü okunamadı"):
        ImageLoader.load_from_bytes(b"not-an-image", "sample.png")


def test_gif_uses_first_frame() -> None:
    first = Image.new("RGB", (12, 10), (255, 0, 0))
    second = Image.new("RGB", (12, 10), (0, 255, 0))
    buffer = io.BytesIO()
    first.save(buffer, format="GIF", save_all=True, append_images=[second], duration=100)

    image, metadata = ImageLoader.load_from_bytes(buffer.getvalue(), "animated.gif")

    assert image.shape == (10, 12, 3)
    assert metadata["format"] == "GIF"
    assert np.all(image[0, 0] == np.array([255, 0, 0]))


def test_preprocess_shapes_and_ranges(png_bytes: bytes) -> None:
    image, _ = ImageLoader.load_from_bytes(png_bytes, "sample.png")

    xception = ImageLoader.preprocess_for_xception(image)
    efficientnet = ImageLoader.preprocess_for_efficientnet(image)
    legacy_second_model = ImageLoader.preprocess_for_lstm(image)

    assert xception.shape == (224, 224, 3)
    assert efficientnet.shape == (224, 224, 3)
    assert legacy_second_model.shape == efficientnet.shape
    assert xception.dtype == np.float32
    assert efficientnet.dtype == np.float32
    assert 0.0 <= float(xception.min()) <= float(xception.max()) <= 1.0
