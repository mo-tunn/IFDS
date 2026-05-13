"""
@file test_verdict.py
@brief Tests for weighted final verdict calculation.
"""

from __future__ import annotations

from src.ai_models.base_model import AIDetectionResult
from src.classical.base_detector import DetectionResult
from src.verdict import VerdictService


def test_weighted_verdict_prioritizes_strong_sift_over_weak_efficientnet() -> None:
    results = {
        "classical": {
            "SIFT": DetectionResult(
                algorithm="SIFT",
                is_forged=True,
                confidence=0.70,
                match_count=68,
                total_keypoints=320,
                processing_time=0.1,
            ),
            "ORB": DetectionResult(
                algorithm="ORB",
                is_forged=False,
                confidence=0.23,
                match_count=5,
                total_keypoints=180,
                processing_time=0.1,
            ),
        },
        "ai": {
            "Xception": AIDetectionResult(
                model_name="Xception",
                is_forged=True,
                confidence=0.52,
                class_label="Tampered",
                processing_time=0.1,
            ),
            "EfficientNet CNN": AIDetectionResult(
                model_name="EfficientNet CNN",
                is_forged=False,
                confidence=0.71,
                class_label="Authentic",
                processing_time=0.1,
            ),
        },
    }

    verdict = VerdictService.evaluate(results)

    assert verdict.label == "Tampered"
    assert verdict.tampered_score > verdict.authentic_score
    assert any(signal.name == "EfficientNet CNN" and signal.weight < 1.0 for signal in verdict.signals)
    assert any(signal.name == "Xception" and signal.verdict == "Uncertain" for signal in verdict.signals)
    assert any(signal.name == "EfficientNet CNN" and signal.verdict == "Uncertain" for signal in verdict.signals)


def test_unavailable_results_do_not_contribute_to_verdict() -> None:
    results = {
        "classical": {
            "SURF": DetectionResult(
                algorithm="SURF",
                is_forged=False,
                confidence=0.0,
                match_count=0,
                total_keypoints=0,
                processing_time=0.0,
                error_message="SURF unavailable",
            )
        },
        "ai": {},
    }

    verdict = VerdictService.evaluate(results)

    assert verdict.label == "Review needed"
    assert verdict.signals == []


def test_low_confidence_orb_does_not_create_authentic_evidence() -> None:
    results = {
        "classical": {
            "ORB": DetectionResult(
                algorithm="ORB",
                is_forged=False,
                confidence=0.35,
                match_count=6,
                total_keypoints=250,
                processing_time=0.1,
            )
        },
        "ai": {},
    }

    verdict = VerdictService.evaluate(results)

    assert verdict.label == "Review needed"
    assert verdict.authentic_score == 0.0
    assert verdict.signals[0].verdict == "Uncertain"


def test_verdict_can_return_authentic_and_medium_suspicion() -> None:
    authentic = VerdictService.evaluate(
        {
            "classical": {
                "AKAZE": DetectionResult(
                    algorithm="AKAZE",
                    is_forged=False,
                    confidence=0.80,
                    match_count=40,
                    total_keypoints=200,
                    processing_time=0.1,
                )
            },
            "ai": {},
        }
    )
    medium = VerdictService.evaluate(
        {
            "classical": {
                "SIFT": DetectionResult(
                    algorithm="SIFT",
                    is_forged=True,
                    confidence=0.60,
                    match_count=40,
                    total_keypoints=200,
                    processing_time=0.1,
                )
            },
            "ai": {
                "Xception": AIDetectionResult(
                    model_name="Xception",
                    is_forged=False,
                    confidence=0.58,
                    class_label="Authentic",
                    processing_time=0.1,
                )
            },
        }
    )

    assert authentic.label == "Authentic"
    assert "No strong tampering signal" in authentic.summary
    assert medium.label == "Tampered"
    assert medium.level == "Medium suspicion"
