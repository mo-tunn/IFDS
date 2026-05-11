"""
@file heatmap_viewer.py
@brief Streamlit helper for AI heatmap rendering.
"""

from __future__ import annotations

from src.ai_models.base_model import AIDetectionResult


def render_heatmap_viewer(result: AIDetectionResult) -> None:
    """
    @brief Render an AI result heatmap or overlay when available.
    @param result AI detection result.
    """
    import streamlit as st

    if result.overlay_image is None:
        return
    st.image(result.overlay_image, caption=f"{result.model_name} heatmap", use_column_width=True)
