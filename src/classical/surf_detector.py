"""
@file surf_detector.py
@brief SURF based copy-move forgery detection with graceful availability checks.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from config.settings import LOWE_RATIO_FLOAT, SURF_MIN_MATCHES
from src.classical.base_detector import BaseDetector, DetectionResult
from src.classical.feature_utils import (
    blank_result_image,
    compute_confidence,
    estimate_inlier_ratio,
    filter_self_matches,
    make_annotated_image,
    make_forge_mask,
)


class SURFDetector(BaseDetector):
    """
    @class SURFDetector
    @brief Detects repeated local features using SURF descriptors.
    """

    def __init__(self, hessian_threshold: float = 400.0, ratio: float = LOWE_RATIO_FLOAT) -> None:
        """
        @brief Create a SURF detector.
        @param hessian_threshold Hessian threshold for SURF keypoints.
        @param ratio Lowe ratio threshold.
        @raises ImportError When SURF is unavailable in the current OpenCV build.
        """
        try:
            self._detector = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian_threshold)
        except (AttributeError, cv2.error) as exc:
            raise ImportError("Can't decide") from exc
        self._ratio = ratio
        self._min_matches = SURF_MIN_MATCHES

    def get_name(self) -> str:
        """
        @brief Return algorithm name.
        @return SURF.
        """
        return "SURF"

    @staticmethod
    def unavailable_result(message: str) -> DetectionResult:
        """
        @brief Build a non-fatal result for unavailable SURF support.
        @param message Explanation to show in the UI and report.
        @return DetectionResult with error_message populated.
        """
        return DetectionResult(
            algorithm="SURF",
            is_forged=False,
            confidence=0.0,
            match_count=0,
            total_keypoints=0,
            processing_time=0.0,
            error_message=message,
        )

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        @brief Run SURF self-matching analysis.
        @param image RGB image.
        @return DetectionResult.
        """
        start = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        keypoints = keypoints or []

        if descriptors is None or len(keypoints) < self._min_matches:
            return DetectionResult(
                algorithm=self.get_name(),
                is_forged=False,
                confidence=0.0,
                match_count=0,
                total_keypoints=len(keypoints),
                processing_time=time.perf_counter() - start,
                annotated_image=blank_result_image(image, keypoints),
            )

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(descriptors.astype(np.float32), descriptors.astype(np.float32), k=4)
        good = filter_self_matches(keypoints, matches, self._ratio)
        inlier_ratio, inliers = estimate_inlier_ratio(keypoints, good)
        evidence_matches = inliers if len(inliers) >= 4 else good
        confidence = compute_confidence(len(evidence_matches), len(keypoints), inlier_ratio, self._min_matches)
        is_forged = len(evidence_matches) >= self._min_matches and confidence >= 0.35
        mask = make_forge_mask(image.shape, keypoints, evidence_matches)

        return DetectionResult(
            algorithm=self.get_name(),
            is_forged=is_forged,
            confidence=confidence,
            match_count=len(evidence_matches),
            total_keypoints=len(keypoints),
            processing_time=time.perf_counter() - start,
            annotated_image=make_annotated_image(image, keypoints, evidence_matches, mask),
            forge_mask=mask,
        )
