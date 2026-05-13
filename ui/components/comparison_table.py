"""
@file comparison_table.py
@brief Streamlit comparison table for all analysis results.
"""

from __future__ import annotations

from typing import Any


def render_comparison_table(results: dict[str, Any]) -> None:
    """
    @brief Render a combined table for classical and AI outputs.
    @param results Analysis results dictionary.
    """
    import pandas as pd
    import streamlit as st

    rows: list[dict[str, Any]] = []
    for result in results.get("classical", {}).values():
        rows.append(
            {
                "Type": "Classical",
                "Name": result.algorithm,
                "Verdict": "Tampered" if result.is_forged else "Authentic",
                "Confidence": f"{result.confidence * 100:.1f}%",
                "Time (s)": f"{result.processing_time:.3f}",
                "Evidence": f"{result.match_count} matches",
                "Note": result.error_message or "",
            }
        )
    for result in results.get("ai", {}).values():
        rows.append(
            {
                "Type": "AI",
                "Name": result.model_name,
                "Verdict": result.class_label,
                "Confidence": f"{result.confidence * 100:.1f}%",
                "Time (s)": f"{result.processing_time:.3f}",
                "Evidence": "heatmap" if result.heatmap is not None else "",
                "Note": result.error_message or "",
            }
        )

    if not rows:
        st.info("Karşılaştırma için sonuç yok.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Verdict": st.column_config.TextColumn("Verdict", width="medium"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            "Time (s)": st.column_config.TextColumn("Time (s)", width="small"),
            "Evidence": st.column_config.TextColumn("Evidence", width="medium"),
            "Note": st.column_config.TextColumn("Note", width="large"),
        },
    )
