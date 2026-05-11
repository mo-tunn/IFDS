"""
@file test_classical_detectors.py
@brief Tests for classical OpenCV detectors.
"""

from __future__ import annotations

import pytest

from src.analysis_service import AnalysisSelection, AnalysisService
from src.classical.akaze_detector import AKAZEDetector
from src.classical.base_detector import DetectionResult
from src.classical.orb_detector import ORBDetector
from src.classical.sift_detector import SIFTDetector


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
