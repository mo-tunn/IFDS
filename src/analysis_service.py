"""
@file analysis_service.py
@brief Orchestrates classical and AI analysis without letting one failure stop the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.ai_models.base_model import AIDetectionResult
from src.ai_models.efficientnet_model import EfficientNetForensicModel
from src.ai_models.gradcam import GradCAM
from src.ai_models.xception_model import XceptionForensicModel
from src.classical.akaze_detector import AKAZEDetector
from src.classical.base_detector import BaseDetector, DetectionResult
from src.classical.orb_detector import ORBDetector
from src.classical.sift_detector import SIFTDetector
from src.classical.surf_detector import SURFDetector
from src.preprocessing.image_loader import ImageLoader


@dataclass
class AnalysisSelection:
    """
    @class AnalysisSelection
    @brief User-selected algorithms for one analysis run.
    """

    run_sift: bool = True
    run_surf: bool = True
    run_akaze: bool = True
    run_orb: bool = True
    run_xception: bool = True
    run_efficientnet: bool = True
    run_gradcam: bool = True


@dataclass
class AnalysisBundle:
    """
    @class AnalysisBundle
    @brief Aggregated outputs from classical and AI analysis.
    """

    classical: dict[str, DetectionResult] = field(default_factory=dict)
    ai: dict[str, AIDetectionResult] = field(default_factory=dict)
    processing_time: float = 0.0


class AnalysisService:
    """
    @class AnalysisService
    @brief Runs selected IFDS detectors and models for a single image.
    """

    def __init__(
        self,
        xception_model: XceptionForensicModel | None = None,
        efficientnet_model: EfficientNetForensicModel | None = None,
    ) -> None:
        """
        @brief Initialize service with optional preloaded AI models.
        @param xception_model Optional Xception model wrapper.
        @param efficientnet_model Optional EfficientNet model wrapper.
        """
        self.xception_model = xception_model
        self.efficientnet_model = efficientnet_model

    def run(self, image: np.ndarray, selection: AnalysisSelection) -> AnalysisBundle:
        """
        @brief Run selected detectors and AI models against an RGB image.
        @param image RGB uint8 image.
        @param selection Selection flags.
        @return AnalysisBundle with non-fatal errors embedded as results.
        """
        start = time.perf_counter()
        bundle = AnalysisBundle()
        bundle.classical = self.run_classical(image, selection)
        bundle.ai = self.run_ai(image, selection)
        bundle.processing_time = time.perf_counter() - start
        return bundle

    def run_classical(
        self,
        image: np.ndarray,
        selection: AnalysisSelection,
    ) -> dict[str, DetectionResult]:
        """
        @brief Run selected classical detectors.
        @param image RGB image.
        @param selection Selection flags.
        @return Mapping of algorithm name to DetectionResult.
        """
        detectors: list[BaseDetector] = []
        results: dict[str, DetectionResult] = {}

        for enabled, factory, name in (
            (selection.run_sift, SIFTDetector, "SIFT"),
            (selection.run_surf, SURFDetector, "SURF"),
            (selection.run_akaze, AKAZEDetector, "AKAZE"),
            (selection.run_orb, ORBDetector, "ORB"),
        ):
            if not enabled:
                continue
            try:
                detectors.append(factory())
            except Exception as exc:
                results[name] = DetectionResult(
                    algorithm=name,
                    is_forged=False,
                    confidence=0.0,
                    match_count=0,
                    total_keypoints=0,
                    processing_time=0.0,
                    error_message=str(exc),
                )

        for detector in detectors:
            name = detector.get_name()
            try:
                results[name] = detector.detect(image)
            except Exception as exc:
                results[name] = DetectionResult(
                    algorithm=name,
                    is_forged=False,
                    confidence=0.0,
                    match_count=0,
                    total_keypoints=0,
                    processing_time=0.0,
                    error_message=str(exc),
                )
        return results

    def run_ai(
        self,
        image: np.ndarray,
        selection: AnalysisSelection,
    ) -> dict[str, AIDetectionResult]:
        """
        @brief Run selected AI models.
        @param image RGB uint8 image.
        @param selection Selection flags.
        @return Mapping of model name to AIDetectionResult.
        """
        results: dict[str, AIDetectionResult] = {}

        if selection.run_xception:
            xception = self.xception_model or XceptionForensicModel()
            try:
                if xception.model is None:
                    xception.load_weights()
                xception_input = ImageLoader.preprocess_for_xception(image)
                result = xception.predict(xception_input)
                if selection.run_gradcam and xception.model is not None:
                    try:
                        heatmap, overlay = GradCAM(xception.model).generate(np.expand_dims(xception_input, axis=0))
                        result.heatmap = heatmap
                        result.overlay_image = overlay
                    except Exception as exc:
                        result.error_message = f"Grad-CAM üretilemedi: {self._short_error(exc)}"
                results["Xception"] = result
            except Exception as exc:
                results["Xception"] = XceptionForensicModel.unavailable_result(str(exc))

        if selection.run_efficientnet:
            efficientnet = self.efficientnet_model or EfficientNetForensicModel()
            try:
                if efficientnet.model is None:
                    efficientnet.load_weights()
                efficientnet_input = ImageLoader.preprocess_for_efficientnet(image)
                results["EfficientNet CNN"] = efficientnet.predict(efficientnet_input)
            except Exception as exc:
                results["EfficientNet CNN"] = EfficientNetForensicModel.unavailable_result(str(exc))

        return results

    @staticmethod
    def _short_error(exc: Exception, max_length: int = 240) -> str:
        """
        @brief Convert verbose framework exceptions into UI-sized messages.
        @param exc Exception instance.
        @param max_length Maximum message length.
        @return Compact single-line message.
        """
        message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return message[:max_length]

    @staticmethod
    def to_dict(bundle: AnalysisBundle) -> dict[str, Any]:
        """
        @brief Convert an AnalysisBundle to a dictionary for UI/reporting compatibility.
        @param bundle Analysis output bundle.
        @return Dictionary with classical, ai, and processing time.
        """
        return {
            "classical": bundle.classical,
            "ai": bundle.ai,
            "processing_time": bundle.processing_time,
        }
