"""
@file akaze_detector.py
@brief AKAZE based copy-move forgery detection.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from config.settings import AKAZE_MIN_MATCHES, LOWE_RATIO_BINARY
from src.classical.base_detector import BaseDetector, DetectionResult
from src.classical.feature_utils import (
    blank_result_image,
    compute_confidence,
    estimate_inlier_ratio,
    filter_self_matches,
    make_annotated_image,
    make_forge_mask,
)


class AKAZEDetector(BaseDetector):
    """
    @class AKAZEDetector
    @brief Detects repeated local features using AKAZE binary descriptors.
    """

    def __init__(self, ratio: float = LOWE_RATIO_BINARY) -> None:
        """
        @brief Create an AKAZE detector.
        @param ratio Lowe ratio threshold for binary descriptors.
        """
        self._detector = cv2.AKAZE_create()
        self._ratio = ratio
        self._min_matches = AKAZE_MIN_MATCHES

    def get_name(self) -> str:
        """
        @brief Return algorithm name.
        @return AKAZE.
        """
        return "AKAZE"

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        @brief Run AKAZE self-matching analysis.
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

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(descriptors, descriptors, k=4)
        good = filter_self_matches(keypoints, matches, self._ratio)
        inlier_ratio, inliers = estimate_inlier_ratio(keypoints, good)
        evidence_matches = inliers if len(inliers) >= 4 else good
        confidence = compute_confidence(len(evidence_matches), len(keypoints), inlier_ratio, self._min_matches)
        is_forged = len(evidence_matches) >= self._min_matches and confidence >= 0.30
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
