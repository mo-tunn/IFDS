"""
@file verdict.py
@brief Weighted final verdict calculation for combined IFDS outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.ai_models.base_model import AIDetectionResult
from src.classical.base_detector import DetectionResult


ALGORITHM_WEIGHTS = {
    "SIFT": 1.35,
    "SURF": 1.10,
    "AKAZE": 0.95,
    "ORB": 0.25,
    "Xception": 0.85,
    "EfficientNet CNN": 0.20,
}

CLASSICAL_MIN_CONFIDENCE = {
    "SIFT": 0.45,
    "SURF": 0.45,
    "AKAZE": 0.35,
    "ORB": 0.55,
}

AI_MIN_CONFIDENCE = {
    "Xception": 0.58,
    "EfficientNet CNN": 0.75,
}

MIN_DECISIVE_EVIDENCE = 0.45


@dataclass
class VerdictSignal:
    """
    @class VerdictSignal
    @brief One weighted signal contributing to the final verdict.
    """

    name: str
    source_type: str
    verdict: str
    confidence: float
    weight: float
    contribution: float
    note: str = ""


@dataclass
class FinalVerdict:
    """
    @class FinalVerdict
    @brief Final ensemble decision over classical and AI outputs.
    """

    label: str
    level: str
    score: float
    tampered_score: float
    authentic_score: float
    summary: str
    signals: list[VerdictSignal] = field(default_factory=list)


class VerdictService:
    """
    @class VerdictService
    @brief Computes a weighted ensemble verdict without treating all models equally.
    """

    @staticmethod
    def evaluate(results: dict[str, Any]) -> FinalVerdict:
        """
        @brief Compute the final weighted verdict.
        @param results AnalysisService.to_dict output.
        @return FinalVerdict.
        """
        signals: list[VerdictSignal] = []
        tampered_score = 0.0
        authentic_score = 0.0

        for result in results.get("classical", {}).values():
            signal = VerdictService._classical_signal(result)
            if signal is None:
                continue
            signals.append(signal)
            if signal.verdict == "Tampered":
                tampered_score += signal.contribution
            elif signal.verdict == "Authentic":
                authentic_score += signal.contribution

        for result in results.get("ai", {}).values():
            signal = VerdictService._ai_signal(result)
            if signal is None:
                continue
            signals.append(signal)
            if signal.verdict == "Tampered":
                tampered_score += signal.contribution
            elif signal.verdict == "Authentic":
                authentic_score += signal.contribution

        total = tampered_score + authentic_score
        score = tampered_score / total if total > 0 else 0.0
        label, level = VerdictService._label(score, tampered_score, authentic_score)
        summary = VerdictService._summary(label, level, score, signals)
        return FinalVerdict(
            label=label,
            level=level,
            score=score,
            tampered_score=tampered_score,
            authentic_score=authentic_score,
            summary=summary,
            signals=signals,
        )

    @staticmethod
    def _classical_signal(result: DetectionResult) -> VerdictSignal | None:
        if result.error_message:
            return None
        name = result.algorithm
        confidence = float(result.confidence or 0.0)
        threshold = CLASSICAL_MIN_CONFIDENCE.get(name, 0.45)
        weight = ALGORITHM_WEIGHTS.get(name, 0.7)
        evidence_factor = min(1.0, max(0.35, (result.match_count or 0) / 40.0))
        if confidence < threshold:
            verdict = "Uncertain"
            contribution = 0.0
            note = f"{result.match_count} matches; below decision threshold"
        else:
            verdict = "Tampered" if result.is_forged else "Authentic"
            contribution = weight * confidence * evidence_factor
            note = f"{result.match_count} matches"
        return VerdictSignal(
            name=name,
            source_type="Classical",
            verdict=verdict,
            confidence=confidence,
            weight=weight,
            contribution=contribution,
            note=note,
        )

    @staticmethod
    def _ai_signal(result: AIDetectionResult) -> VerdictSignal | None:
        if result.error_message and result.class_label == "Unavailable":
            return None
        name = result.model_name
        confidence = float(result.confidence or 0.0)
        threshold = AI_MIN_CONFIDENCE.get(name, 0.55)
        weight = ALGORITHM_WEIGHTS.get(name, 0.6)
        if confidence < threshold:
            verdict = "Uncertain"
            contribution = 0.0
            note = "below decision threshold"
        else:
            verdict = "Tampered" if result.is_forged else "Authentic"
            contribution = weight * confidence
            note = ""
        return VerdictSignal(
            name=name,
            source_type="AI",
            verdict=verdict,
            confidence=confidence,
            weight=weight,
            contribution=contribution,
            note=note,
        )

    @staticmethod
    def _label(score: float, tampered_score: float, authentic_score: float) -> tuple[str, str]:
        if tampered_score + authentic_score < MIN_DECISIVE_EVIDENCE:
            return "Review needed", "Insufficient decisive evidence"
        if score >= 0.68:
            return "Tampered", "High suspicion"
        if score >= 0.55:
            return "Tampered", "Medium suspicion"
        if score >= 0.42:
            return "Review needed", "Mixed evidence"
        return "Authentic", "Low suspicion"

    @staticmethod
    def _summary(label: str, level: str, score: float, signals: list[VerdictSignal]) -> str:
        tampered = [signal for signal in signals if signal.verdict == "Tampered"]
        if not signals:
            return "Final verdict could not be computed because no usable detector result was produced."
        if tampered:
            names = ", ".join(signal.name for signal in sorted(tampered, key=lambda item: item.contribution, reverse=True)[:3])
            return f"{level}: weighted tampering score is {score * 100:.1f}%. Main tampering signals: {names}."
        return f"{level}: weighted tampering score is {score * 100:.1f}%. No strong tampering signal dominated."
