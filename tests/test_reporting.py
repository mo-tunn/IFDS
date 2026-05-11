"""
@file test_reporting.py
@brief Tests for report generation.
"""

from __future__ import annotations

from src.analysis_service import AnalysisSelection, AnalysisService
from src.reporting.report_generator import ReportGenerator


def test_generate_html_contains_metadata(sample_rgb_image) -> None:
    results = AnalysisService.to_dict(
        AnalysisService().run(
            sample_rgb_image,
            AnalysisSelection(run_sift=False, run_surf=False, run_akaze=False, run_orb=False),
        )
    )
    metadata = {
        "filename": "sample.png",
        "format": "PNG",
        "width": 256,
        "height": 256,
        "channels": 3,
        "size_mb": 0.1,
    }

    html = ReportGenerator().generate_html(sample_rgb_image, metadata, results)

    assert "IFDS Image Forgery Detection Report" in html
    assert "sample.png" in html
