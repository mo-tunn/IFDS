"""
@file test_analysis_service.py
@brief Integration tests for the analysis orchestration service.
"""

from __future__ import annotations

import numpy as np

from src.ai_models.base_model import AIDetectionResult
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


class LoadedXception:
    def __init__(self) -> None:
        self.model = object()
        self.loaded = False

    def load_weights(self) -> None:
        self.loaded = True

    def predict(self, image) -> AIDetectionResult:
        return AIDetectionResult(
            model_name="Xception",
            is_forged=True,
            confidence=0.9,
            class_label="Tampered",
            processing_time=0.01,
        )


class LoadedEfficientNet:
    def __init__(self) -> None:
        self.model = object()

    def load_weights(self) -> None:
        raise AssertionError("already loaded")

    def predict(self, image) -> AIDetectionResult:
        return AIDetectionResult(
            model_name="EfficientNet CNN",
            is_forged=False,
            confidence=0.8,
            class_label="Authentic",
            processing_time=0.01,
        )


def test_run_ai_attaches_gradcam_outputs(monkeypatch, sample_rgb_image) -> None:
    class FakeGradCAM:
        def __init__(self, model) -> None:
            self.model = model

        def generate(self, batch):
            assert batch.shape[0] == 1
            return np.ones((4, 4), dtype=np.float32), np.zeros((224, 224, 3), dtype=np.uint8)

    monkeypatch.setattr("src.analysis_service.GradCAM", FakeGradCAM)
    service = AnalysisService(LoadedXception(), LoadedEfficientNet())

    results = service.run_ai(sample_rgb_image, AnalysisSelection(run_xception=True, run_efficientnet=True))

    assert results["Xception"].heatmap is not None
    assert results["Xception"].overlay_image is not None
    assert results["EfficientNet CNN"].class_label == "Authentic"


def test_run_ai_captures_gradcam_error(monkeypatch, sample_rgb_image) -> None:
    class FailingGradCAM:
        def __init__(self, model) -> None:
            self.model = model

        def generate(self, _batch):
            raise RuntimeError("first line\nsecond line")

    monkeypatch.setattr("src.analysis_service.GradCAM", FailingGradCAM)

    results = AnalysisService(LoadedXception(), None).run_ai(
        sample_rgb_image,
        AnalysisSelection(run_xception=True, run_efficientnet=False, run_gradcam=True),
    )

    assert results["Xception"].error_message == "Grad-CAM üretilemedi: first line"
    assert AnalysisService._short_error(RuntimeError("x" * 300), max_length=10) == "x" * 10
