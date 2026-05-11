"""
@file base_detector.py
@brief Shared contracts for classical forgery detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    """
    @class DetectionResult
    @brief Result object returned by every classical detector.
    """

    algorithm: str
    is_forged: bool
    confidence: float
    match_count: int
    total_keypoints: int
    processing_time: float
    annotated_image: np.ndarray | None = None
    forge_mask: np.ndarray | None = None
    error_message: str | None = None


class BaseDetector(ABC):
    """
    @class BaseDetector
    @brief Abstract base class for OpenCV feature-based detectors.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        @brief Analyze an RGB image for signs of forgery.
        @param image RGB image as numpy array.
        @return DetectionResult containing verdict and diagnostics.
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        @brief Return a user-facing algorithm name.
        @return Algorithm name.
        """
