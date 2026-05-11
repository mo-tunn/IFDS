"""
@file test_analysis_service.py
@brief Integration tests for the analysis orchestration service.
"""

from __future__ import annotations

from src.analysis_service import AnalysisSelection, AnalysisService
from src.verdict import VerdictService


def test_analysis_service_combines_selected_results(sample_rgb_image) -> None:
    service = AnalysisService()
    selection = AnalysisSelection(
        run_sift=False,
        run_surf=False,
        run_akaze=True,
        run_orb=False,
        run_xception=True,
        run_efficientnet=True,
        run_gradcam=False,
    )

    bundle = service.run(sample_rgb_image, selection)

    assert "AKAZE" in bundle.classical
    assert "Xception" in bundle.ai
    assert "EfficientNet CNN" in bundle.ai
    assert bundle.ai["Xception"].error_message is not None
    assert bundle.ai["EfficientNet CNN"].error_message is not None
    assert bundle.processing_time >= 0.0

    verdict = VerdictService.evaluate(AnalysisService.to_dict(bundle))

    assert verdict.label in {"Authentic", "Tampered", "Review needed"}
    assert 0.0 <= verdict.score <= 1.0
