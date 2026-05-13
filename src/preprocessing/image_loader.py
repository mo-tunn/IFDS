"""
@file image_loader.py
@brief Multi-format image loading, validation, and preprocessing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config.settings import (
    EFFICIENTNET_INPUT_SIZE,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    SUPPORTED_FORMATS,
    XCEPTION_INPUT_SIZE,
)


class ImageLoader:
    """
    @class ImageLoader
    @brief Loads supported image formats into RGB numpy arrays.
    """

    @staticmethod
    def load_from_path(file_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
        """
        @brief Load an image from disk and return RGB pixels plus metadata.
        @param file_path Image path.
        @return Tuple of RGB image array and metadata.
        @raises FileNotFoundError When the path does not exist.
        @raises ValueError When the file is unsupported, oversized, or unreadable.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File was not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a valid file: {path}")

        ImageLoader._validate_suffix(path.suffix)
        size_bytes = path.stat().st_size
        ImageLoader._validate_size(size_bytes)

        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"File could not be read: {path}") from exc

        image = ImageLoader._decode_bytes(payload, path.suffix)
        metadata = ImageLoader._metadata(path.name, path.suffix, size_bytes, image)
        return image, metadata

    @staticmethod
    def load_from_bytes(file_bytes: bytes, filename: str) -> tuple[np.ndarray, dict[str, Any]]:
        """
        @brief Load an uploaded image from bytes without persisting it.
        @param file_bytes Image bytes from an uploader.
        @param filename Original filename used for extension and metadata.
        @return Tuple of RGB image array and metadata.
        @raises ValueError When the file is unsupported, oversized, or unreadable.
        """
        suffix = Path(filename).suffix.lower()
        ImageLoader._validate_suffix(suffix)
        ImageLoader._validate_size(len(file_bytes))

        image = ImageLoader._decode_bytes(file_bytes, suffix)
        metadata = ImageLoader._metadata(filename, suffix, len(file_bytes), image)
        return image, metadata

    @staticmethod
    def preprocess_for_xception(image: np.ndarray) -> np.ndarray:
        """
        @brief Resize and normalize an RGB image for Xception inference.
        @param image RGB uint8 image.
        @return Float32 image with shape (224, 224, 3) and range [0, 1].
        """
        return ImageLoader._resize_and_normalize(image, XCEPTION_INPUT_SIZE)

    @staticmethod
    def preprocess_for_lstm(image: np.ndarray) -> np.ndarray:
        """
        @brief Backward-compatible alias for EfficientNet preprocessing.
        @param image RGB uint8 image.
        @return Float32 image with shape (224, 224, 3) and range [0, 1].
        """
        return ImageLoader._resize_and_normalize(image, EFFICIENTNET_INPUT_SIZE)

    @staticmethod
    def preprocess_for_efficientnet(image: np.ndarray) -> np.ndarray:
        """
        @brief Resize and normalize an RGB image for EfficientNet inference.
        @param image RGB uint8 image.
        @return Float32 image with shape (224, 224, 3) and range [0, 1].
        """
        return ImageLoader._resize_and_normalize(image, EFFICIENTNET_INPUT_SIZE)

    @staticmethod
    def preprocess_for_model(
        image: np.ndarray,
        target_size: tuple[int, int] = XCEPTION_INPUT_SIZE,
    ) -> np.ndarray:
        """
        @brief Backward-compatible generic preprocessing helper.
        @param image RGB image.
        @param target_size Target width and height.
        @return Normalized float32 image.
        """
        return ImageLoader._resize_and_normalize(image, target_size)

    @staticmethod
    def _validate_suffix(suffix: str) -> None:
        normalized = suffix.lower()
        if normalized not in SUPPORTED_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_FORMATS))
            raise ValueError(f"Unsupported format: {suffix or '(none)'}. Supported formats: {supported}")

    @staticmethod
    def _validate_size(size_bytes: int) -> None:
        if size_bytes <= 0:
            raise ValueError("File is empty or does not contain readable image data.")
        if size_bytes > MAX_FILE_SIZE_BYTES:
            size_mb = size_bytes / (1024 * 1024)
            raise ValueError(f"File is too large: {size_mb:.1f} MB (maximum {MAX_FILE_SIZE_MB} MB).")

    @staticmethod
    def _decode_bytes(file_bytes: bytes, suffix: str) -> np.ndarray:
        if suffix.lower() == ".gif":
            return ImageLoader._decode_with_pillow(file_bytes)

        buffer = np.frombuffer(file_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image_bgr is not None:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return ImageLoader._decode_with_pillow(file_bytes)

    @staticmethod
    def _decode_with_pillow(file_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(file_bytes)) as pil_image:
                if getattr(pil_image, "is_animated", False):
                    pil_image.seek(0)
                return np.asarray(pil_image.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Image could not be read or the file is corrupted.") from exc

    @staticmethod
    def _resize_and_normalize(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected image format must be RGB with 3 channels.")
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    @staticmethod
    def _metadata(
        filename: str,
        suffix: str,
        size_bytes: int,
        image: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "filename": filename,
            "format": suffix.upper().replace(".", ""),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "channels": int(image.shape[2]) if image.ndim == 3 else 1,
            "size_mb": round(size_bytes / (1024 * 1024), 3),
        }
