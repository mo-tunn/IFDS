"""
@file report_generator.py
@brief PDF and HTML report generation for IFDS analysis results.
"""

from __future__ import annotations

import base64
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Template
from PIL import Image

from config.settings import PRIMARY_COLOR, SECONDARY_COLOR
from src.verdict import FinalVerdict, VerdictService


class ReportGenerator:
    """
    @class ReportGenerator
    @brief Generates downloadable HTML and PDF analysis reports.
    """

    def generate_html(
        self,
        image: np.ndarray,
        metadata: dict[str, Any],
        results: dict[str, Any],
    ) -> str:
        """
        @brief Render an HTML report.
        @param image Uploaded RGB image.
        @param metadata Image metadata.
        @param results Analysis results dictionary.
        @return HTML string.
        """
        template = Template(self._html_template())
        final_verdict = self._final_verdict(results)
        return template.render(
            image_data=self._image_to_data_uri(image),
            metadata=metadata,
            final_verdict=final_verdict,
            classical=self._normalize_results(results.get("classical", {})),
            ai=self._normalize_results(results.get("ai", {})),
            visuals=self._collect_visuals(results),
            processing_time=results.get("processing_time", 0.0),
            primary_color=PRIMARY_COLOR,
            secondary_color=SECONDARY_COLOR,
        )

    def generate_pdf(
        self,
        image: np.ndarray,
        metadata: dict[str, Any],
        results: dict[str, Any],
    ) -> bytes:
        """
        @brief Generate a compact PDF report.
        @param image Uploaded RGB image.
        @param metadata Image metadata.
        @param results Analysis results dictionary.
        @return PDF bytes.
        @raises ImportError When fpdf2 is unavailable.
        """
        try:
            from fpdf import FPDF
        except ImportError as exc:
            raise ImportError("PDF raporu için fpdf2 kurulmalı.") from exc

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(31, 56, 100)
        pdf.cell(0, 10, "IFDS Image Forgery Detection Report", ln=True)
        pdf.set_text_color(0, 0, 0)
        final_verdict = self._final_verdict(results)

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "uploaded.png"
            Image.fromarray(image.astype(np.uint8)).save(image_path)
            pdf.image(str(image_path), x=10, y=25, w=65)

            pdf.set_xy(82, 25)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, "Image Metadata", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for key, value in metadata.items():
                pdf.set_x(82)
                pdf.multi_cell(0, 6, f"{key}: {value}")

        pdf.ln(12)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Final Verdict", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(
            0,
            6,
            f"{final_verdict.label} - {final_verdict.level} | "
            f"Weighted tampering score: {final_verdict.score * 100:.1f}%",
        )
        pdf.multi_cell(0, 6, final_verdict.summary)

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Classical Analysis", ln=True)
        self._write_result_rows(pdf, self._normalize_results(results.get("classical", {})))

        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI Analysis", ln=True)
        self._write_result_rows(pdf, self._normalize_results(results.get("ai", {})))

        pdf.ln(3)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 6, f"Total processing time: {results.get('processing_time', 0.0):.3f} seconds")

        visuals = self._collect_visual_arrays(results)
        if visuals:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Visual Evidence", ln=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                x_pos = 10
                y_pos = 25
                for index, (title, visual) in enumerate(visuals[:6]):
                    if index and index % 2 == 0:
                        pdf.add_page()
                        x_pos = 10
                        y_pos = 25
                    image_path = Path(tmp_dir) / f"visual_{index}.png"
                    Image.fromarray(visual.astype(np.uint8)).save(image_path)
                    pdf.set_xy(x_pos, y_pos - 6)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.cell(85, 5, title)
                    pdf.image(str(image_path), x=x_pos, y=y_pos, w=85)
                    x_pos = 105 if x_pos == 10 else 10
                    if x_pos == 10:
                        y_pos += 80

        output = pdf.output(dest="S")
        if isinstance(output, bytes):
            return output
        return output.encode("latin-1")

    def generate(
        self,
        image: np.ndarray,
        metadata: dict[str, Any],
        results: dict[str, Any],
    ) -> bytes:
        """
        @brief Backward-compatible PDF generation alias.
        @param image Uploaded RGB image.
        @param metadata Image metadata.
        @param results Analysis results dictionary.
        @return PDF bytes.
        """
        return self.generate_pdf(image, metadata, results)

    def _write_result_rows(self, pdf: Any, rows: list[dict[str, Any]]) -> None:
        if not rows:
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 6, "No result produced.", ln=True)
            return

        for row in rows:
            name = row.get("algorithm") or row.get("model_name") or "Unknown"
            label = row.get("class_label") or ("Tampered" if row.get("is_forged") else "Authentic")
            confidence = float(row.get("confidence") or 0.0)
            duration = float(row.get("processing_time") or 0.0)
            error = row.get("error_message")
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, str(name), ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 6, f"Verdict: {label} | Confidence: {confidence * 100:.1f}% | Time: {duration:.3f}s")
            if "match_count" in row:
                pdf.multi_cell(
                    0,
                    6,
                    f"Matches: {row.get('match_count', 0)} | Keypoints: {row.get('total_keypoints', 0)}",
                )
            if error:
                pdf.set_text_color(180, 35, 24)
                pdf.multi_cell(0, 6, f"Note: {error}")
                pdf.set_text_color(0, 0, 0)

    def _normalize_results(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for _, value in results.items():
            if is_dataclass(value):
                row = asdict(value)
            elif isinstance(value, dict):
                row = dict(value)
            else:
                row = {"value": str(value)}
            row.pop("annotated_image", None)
            row.pop("forge_mask", None)
            row.pop("heatmap", None)
            row.pop("overlay_image", None)
            normalized.append(row)
        return normalized

    def _collect_visuals(self, results: dict[str, Any]) -> list[dict[str, str]]:
        visuals = []
        for title, image in self._collect_visual_arrays(results):
            visuals.append({"title": title, "data_uri": self._image_to_data_uri(image)})
        return visuals

    def _collect_visual_arrays(self, results: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
        visuals: list[tuple[str, np.ndarray]] = []
        for name, result in results.get("classical", {}).items():
            image = getattr(result, "annotated_image", None)
            if image is not None:
                visuals.append((f"{name} annotated result", self._to_uint8_rgb(image)))
        for name, result in results.get("ai", {}).items():
            image = getattr(result, "overlay_image", None)
            if image is not None:
                visuals.append((f"{name} heatmap overlay", self._to_uint8_rgb(image)))
        return visuals

    @staticmethod
    def _final_verdict(results: dict[str, Any]) -> FinalVerdict:
        value = results.get("final_verdict")
        if isinstance(value, FinalVerdict):
            return value
        return VerdictService.evaluate(results)

    def _image_to_data_uri(self, image: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = Path(handle.name)
        try:
            Image.fromarray(image.astype(np.uint8)).save(path)
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{payload}"
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            import cv2

            colored = cv2.applyColorMap(np.uint8(np.clip(image, 0, 1) * 255), cv2.COLORMAP_JET)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        if image.dtype == np.uint8:
            return image
        return np.clip(image * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def _html_template() -> str:
        return """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>IFDS Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; color: #162033; }
    h1 { color: {{ primary_color }}; }
    h2 { border-bottom: 2px solid {{ secondary_color }}; padding-bottom: 4px; }
    img { max-width: 360px; border: 1px solid #d9e2ef; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }
    th, td { border: 1px solid #d9e2ef; padding: 8px; text-align: left; }
    th { background: #edf4fb; color: {{ primary_color }}; }
    .error { color: #B42318; }
    .verdict { border: 1px solid #d9e2ef; padding: 12px; margin: 16px 0; background: #f8fbff; }
  </style>
</head>
<body>
  <h1>IFDS Image Forgery Detection Report</h1>
  <img src="{{ image_data }}" alt="Analyzed image">
  <h2>Final Verdict</h2>
  <div class="verdict">
    <strong>{{ final_verdict.label }} - {{ final_verdict.level }}</strong><br>
    Weighted tampering score: {{ "%.1f"|format(final_verdict.score * 100) }}%<br>
    {{ final_verdict.summary }}
  </div>
  <h2>Metadata</h2>
  <table>
    {% for key, value in metadata.items() %}
    <tr><th>{{ key }}</th><td>{{ value }}</td></tr>
    {% endfor %}
  </table>
  <h2>Classical Analysis</h2>
  <table>
    <tr><th>Algorithm</th><th>Verdict</th><th>Confidence</th><th>Matches</th><th>Keypoints</th><th>Time</th><th>Note</th></tr>
    {% for row in classical %}
    <tr>
      <td>{{ row.algorithm }}</td>
      <td>{{ "Tampered" if row.is_forged else "Authentic" }}</td>
      <td>{{ "%.1f"|format(row.confidence * 100) }}%</td>
      <td>{{ row.match_count }}</td>
      <td>{{ row.total_keypoints }}</td>
      <td>{{ "%.3f"|format(row.processing_time) }}s</td>
      <td class="error">{{ row.error_message or "" }}</td>
    </tr>
    {% endfor %}
  </table>
  <h2>AI Analysis</h2>
  <table>
    <tr><th>Model</th><th>Verdict</th><th>Confidence</th><th>Time</th><th>Note</th></tr>
    {% for row in ai %}
    <tr>
      <td>{{ row.model_name }}</td>
      <td>{{ row.class_label }}</td>
      <td>{{ "%.1f"|format(row.confidence * 100) }}%</td>
      <td>{{ "%.3f"|format(row.processing_time) }}s</td>
      <td class="error">{{ row.error_message or "" }}</td>
    </tr>
    {% endfor %}
  </table>
  {% if visuals %}
  <h2>Visual Evidence</h2>
  {% for visual in visuals %}
    <h3>{{ visual.title }}</h3>
    <img src="{{ visual.data_uri }}" alt="{{ visual.title }}">
  {% endfor %}
  {% endif %}
  <p>Total processing time: {{ "%.3f"|format(processing_time) }}s</p>
</body>
</html>
"""
