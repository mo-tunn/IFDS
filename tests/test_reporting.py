"""
@file test_reporting.py
@brief Tests for report generation.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from src.ai_models.base_model import AIDetectionResult
from src.analysis_service import AnalysisSelection, AnalysisService
from src.classical.base_detector import DetectionResult
from src.reporting.report_generator import ReportGenerator
from src.verdict import FinalVerdict


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


def test_generate_pdf_includes_results_and_visuals(sample_rgb_image) -> None:
    metadata = {
        "filename": "sample.png",
        "format": "PNG",
        "width": 256,
        "height": 256,
        "channels": 3,
        "size_mb": 0.1,
    }
    visual = np.zeros_like(sample_rgb_image)
    results = {
        "classical": {
            "SIFT": DetectionResult(
                algorithm="SIFT",
                is_forged=True,
                confidence=0.82,
                match_count=18,
                total_keypoints=120,
                processing_time=0.12,
                annotated_image=visual,
            )
        },
        "ai": {
            "Xception": AIDetectionResult(
                model_name="Xception",
                is_forged=False,
                confidence=0.66,
                class_label="Authentic",
                processing_time=0.2,
                overlay_image=np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4),
            )
        },
        "processing_time": 0.4,
        "final_verdict": FinalVerdict(
            label="Tampered",
            level="High suspicion",
            score=0.9,
            tampered_score=1.0,
            authentic_score=0.1,
            summary="Strong tampering evidence.",
        ),
    }

    pdf_bytes = ReportGenerator().generate_pdf(sample_rgb_image, metadata, results)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_generate_alias_and_empty_rows(sample_rgb_image) -> None:
    metadata = {"filename": "empty.png", "format": "PNG"}
    results = {
        "classical": {"raw": "plain value"},
        "ai": {},
        "final_verdict": FinalVerdict(
            label="Review needed",
            level="Insufficient decisive evidence",
            score=0.0,
            tampered_score=0.0,
            authentic_score=0.0,
            summary="No usable result.",
        ),
    }
    generator = ReportGenerator()

    pdf_bytes = generator.generate(sample_rgb_image, metadata, results)
    normalized = generator._normalize_results(results["classical"])
    fake_pdf = type(
        "FakePdf",
        (),
        {"set_x": lambda *a, **k: None, "set_font": lambda *a, **k: None, "cell": lambda *a, **k: None},
    )()
    generator._write_result_rows(fake_pdf, [])

    assert pdf_bytes.startswith(b"%PDF")
    assert normalized == [{"value": "plain value"}]


def test_generate_pdf_reports_missing_fpdf(monkeypatch: pytest.MonkeyPatch, sample_rgb_image) -> None:
    real_import = builtins.__import__

    def raising_import(name, *args, **kwargs):
        if name == "fpdf":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)

    with pytest.raises(ImportError, match="fpdf2 must be installed"):
        ReportGenerator().generate_pdf(sample_rgb_image, {"filename": "sample.png"}, {"classical": {}, "ai": {}})
