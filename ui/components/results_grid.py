"""
@file results_grid.py
@brief Streamlit grid for classical detector results.
"""

from __future__ import annotations

import html

from src.classical.base_detector import DetectionResult


def _status_class(result: DetectionResult) -> str:
    if result.error_message:
        return "is-warning"
    return "is-danger" if result.is_forged else "is-success"


def render_results_grid(results: dict[str, DetectionResult]) -> None:
    """
    @brief Render classical result metric cards and visual overlays.
    @param results Mapping of algorithm name to DetectionResult.
    """
    import streamlit as st

    if not results:
        st.info("No classical algorithm results available.")
        return

    columns = st.columns(min(4, max(1, len(results))))
    for index, (name, result) in enumerate(results.items()):
        with columns[index % len(columns)]:
            verdict = "Tampered" if result.is_forged else "Authentic"
            if result.error_message:
                verdict = "can't decide"
            st.markdown(
                f"""
                <div class="ifds-result-card {_status_class(result)}">
                  <div class="ifds-card-topline">
                    <span>{html.escape(name)}</span>
                    <strong>{result.confidence * 100:.1f}%</strong>
                  </div>
                  <div class="ifds-card-verdict">{html.escape(verdict)}</div>
                  <div class="ifds-card-meta">
                    <span>{result.match_count} matches</span>
                    <span>{result.total_keypoints} keypoints</span>
                    <span>{result.processing_time:.3f}s</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result.error_message:
                st.warning(f"{name}: {result.error_message}")
            if result.annotated_image is not None:
                st.image(result.annotated_image, caption=f"{name} overlay", use_column_width=True)
