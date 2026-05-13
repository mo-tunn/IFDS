"""
@file test_classical_detectors.py
@brief Tests for classical OpenCV detectors.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.analysis_service import AnalysisSelection, AnalysisService
from src.classical.akaze_detector import AKAZEDetector
from src.classical.base_detector import DetectionResult
from src.classical.feature_utils import filter_self_matches
from src.classical.orb_detector import ORBDetector
from src.classical.sift_detector import SIFTDetector
from src.classical.surf_detector import SURFDetector


@pytest.mark.parametrize("detector_cls", [SIFTDetector, AKAZEDetector, ORBDetector])
def test_detector_returns_detection_result(detector_cls, sample_rgb_image) -> None:
    detector = detector_cls()
    result = detector.detect(sample_rgb_image)

    assert isinstance(result, DetectionResult)
    assert result.algorithm == detector.get_name()
    assert 0.0 <= result.confidence <= 1.0
    assert result.processing_time >= 0.0


@pytest.mark.parametrize("detector_cls", [SIFTDetector, AKAZEDetector, ORBDetector])
def test_detector_handles_low_feature_image(detector_cls, blank_rgb_image) -> None:
    detector = detector_cls()
    result = detector.detect(blank_rgb_image)

    assert isinstance(result, DetectionResult)
    assert result.is_forged is False
    assert result.match_count == 0


def test_copy_move_image_runs_without_crash(copy_move_image) -> None:
    result = ORBDetector().detect(copy_move_image)

    assert isinstance(result, DetectionResult)
    assert result.match_count >= 0
    assert result.total_keypoints >= 0


def test_surf_unavailable_is_returned_as_error(monkeypatch: pytest.MonkeyPatch, sample_rgb_image) -> None:
    def raising_factory():
        raise ImportError("SURF unavailable")

    monkeypatch.setattr("src.analysis_service.SURFDetector", raising_factory)
    service = AnalysisService()
    results = service.run_classical(
        sample_rgb_image,
        AnalysisSelection(run_sift=False, run_surf=True, run_akaze=False, run_orb=False),
    )

    assert "SURF" in results
    assert results["SURF"].error_message == "SURF unavailable"


def test_surf_unavailable_result_factory() -> None:
    result = SURFDetector.unavailable_result("missing nonfree support")

    assert result.algorithm == "SURF"
    assert result.error_message == "missing nonfree support"


class NoDescriptorDetector:
    def detectAndCompute(self, _gray, _mask):
        return [cv2.KeyPoint(5.0, 5.0, 3.0)], None


class DescriptorDetector:
    def detectAndCompute(self, _gray, _mask):
        keypoints = [cv2.KeyPoint(float(index * 10), 5.0, 3.0) for index in range(12)]
        descriptors = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
        return keypoints, descriptors


def _surf_with_detector(detector, min_matches: int = 1) -> SURFDetector:
    surf = object.__new__(SURFDetector)
    surf._detector = detector
    surf._ratio = 0.95
    surf._min_matches = min_matches
    return surf


def test_surf_detect_handles_missing_descriptors(sample_rgb_image) -> None:
    result = _surf_with_detector(NoDescriptorDetector()).detect(sample_rgb_image)

    assert result.algorithm == "SURF"
    assert result.is_forged is False
    assert result.total_keypoints == 1
    assert result.annotated_image is not None


def test_surf_detect_uses_match_pipeline(monkeypatch: pytest.MonkeyPatch, sample_rgb_image) -> None:
    matches = [
        [
            cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.0),
            cv2.DMatch(_queryIdx=0, _trainIdx=2, _distance=0.1),
            cv2.DMatch(_queryIdx=0, _trainIdx=3, _distance=0.3),
        ],
        [
            cv2.DMatch(_queryIdx=1, _trainIdx=1, _distance=0.0),
            cv2.DMatch(_queryIdx=1, _trainIdx=4, _distance=0.1),
            cv2.DMatch(_queryIdx=1, _trainIdx=5, _distance=0.3),
        ],
    ]

    class FakeMatcher:
        def __init__(self, *_args, **_kwargs):
            pass

        def knnMatch(self, *_args, **_kwargs):
            return matches

    monkeypatch.setattr("src.classical.surf_detector.cv2.BFMatcher", FakeMatcher)
    monkeypatch.setattr(
        "src.classical.surf_detector.estimate_inlier_ratio",
        lambda _keypoints, good: (0.75, list(good)),
    )

    result = _surf_with_detector(DescriptorDetector()).detect(sample_rgb_image)

    assert result.match_count == 2
    assert result.confidence > 0
    assert result.forge_mask is not None


def test_filter_self_matches_skips_short_self_close_and_duplicate_groups() -> None:
    keypoints = [cv2.KeyPoint(float(index * 2), 0.0, 1.0) for index in range(5)]
    groups = [
        [cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.0)],
        [
            cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.0),
            cv2.DMatch(_queryIdx=0, _trainIdx=1, _distance=0.1),
            cv2.DMatch(_queryIdx=0, _trainIdx=2, _distance=0.3),
        ],
        [
            cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.0),
            cv2.DMatch(_queryIdx=0, _trainIdx=4, _distance=0.1),
            cv2.DMatch(_queryIdx=0, _trainIdx=3, _distance=0.3),
        ],
        [
            cv2.DMatch(_queryIdx=4, _trainIdx=4, _distance=0.0),
            cv2.DMatch(_queryIdx=4, _trainIdx=0, _distance=0.1),
            cv2.DMatch(_queryIdx=4, _trainIdx=3, _distance=0.3),
        ],
    ]

    good = filter_self_matches(keypoints, groups, ratio=0.8, min_displacement_px=4.0)

    assert len(good) == 1
    assert (good[0].queryIdx, good[0].trainIdx) == (0, 4)
