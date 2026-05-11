"""
@file base_model.py
@brief Shared AI inference result structures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AIDetectionResult:
    """
    @class AIDetectionResult
    @brief Result object returned by AI inference models.
    """

    model_name: str
    is_forged: bool
    confidence: float
    class_label: str
    processing_time: float
    heatmap: np.ndarray | None = None
    overlay_image: np.ndarray | None = None
    error_message: str | None = None
