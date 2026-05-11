"""
@file results_grid.py
@brief Streamlit grid for classical detector results.
"""

from __future__ import annotations

from src.classical.base_detector import DetectionResult


def render_results_grid(results: dict[str, DetectionResult]) -> None:
    """
    @brief Render classical result metric cards and visual overlays.
    @param results Mapping of algorithm name to DetectionResult.
    """
    import streamlit as st

    if not results:
        st.info("Klasik algoritma sonucu yok.")
        return

    columns = st.columns(min(4, max(1, len(results))))
    for index, (name, result) in enumerate(results.items()):
        with columns[index % len(columns)]:
            if result.error_message:
                st.warning(f"{name}: {result.error_message}")
            verdict = "Tampered" if result.is_forged else "Authentic"
            st.metric(name, verdict, f"{result.confidence * 100:.1f}%")
            st.caption(
                f"Matches: {result.match_count} | Keypoints: {result.total_keypoints} | "
                f"{result.processing_time:.3f}s"
            )
            if result.annotated_image is not None:
                st.image(result.annotated_image, caption=f"{name} overlay", use_column_width=True)
