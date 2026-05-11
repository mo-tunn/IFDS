"""
@file test_ai_models.py
@brief Tests for AI wrappers without real TensorFlow weights.
"""

from __future__ import annotations

import numpy as np

from src.ai_models.efficientnet_model import EfficientNetForensicModel
from src.ai_models.xception_model import XceptionForensicModel


class DummyXceptionBackend:
    """@brief Minimal backend exposing Keras-like predict."""

    def predict(self, batch, verbose: int = 0):
        return np.array([[0.82]], dtype=np.float32)


class DummyEfficientNetBackend:
    """@brief Minimal backend exposing Keras-like predict."""

    def predict(self, batch, verbose: int = 0):
        return np.array([[0.91]], dtype=np.float32)


def test_xception_predict_with_mock_backend() -> None:
    model = XceptionForensicModel()
    model.model = DummyXceptionBackend()
    image = np.zeros((224, 224, 3), dtype=np.float32)

    result = model.predict(image)

    assert result.model_name == "Xception"
    assert result.is_forged is True
    assert result.class_label == "Tampered"
    assert result.confidence > 0.8


def test_efficientnet_predict_with_mock_backend() -> None:
    model = EfficientNetForensicModel()
    model.model = DummyEfficientNetBackend()
    image = np.zeros((224, 224, 3), dtype=np.float32)

    result = model.predict(image)

    assert result.model_name == "EfficientNet CNN"
    assert result.is_forged is True
    assert result.class_label == "Tampered"
    assert result.confidence > 0.9
