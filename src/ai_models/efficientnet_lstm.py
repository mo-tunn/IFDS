"""
@file efficientnet_lstm.py
@brief Backward-compatible import shim for the former EfficientNet+LSTM module.
"""

from __future__ import annotations

from src.ai_models.efficientnet_model import EfficientNetForensicModel


EfficientNetLSTMModel = EfficientNetForensicModel
