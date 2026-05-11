"""
@file final_verdict.py
@brief Streamlit final verdict summary component.
"""

from __future__ import annotations

from src.verdict import FinalVerdict


def render_final_verdict(verdict: FinalVerdict) -> None:
    """
    @brief Render weighted final verdict.
    @param verdict FinalVerdict instance.
    """
    import pandas as pd
    import streamlit as st

    if verdict.label == "Tampered":
        st.error(f"Genel Karar: {verdict.label} - {verdict.level}")
    elif verdict.label == "Review needed":
        st.warning(f"Genel Karar: {verdict.label} - {verdict.level}")
    else:
        st.success(f"Genel Karar: {verdict.label} - {verdict.level}")

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Weighted Tampering Score", f"{verdict.score * 100:.1f}%")
    col_2.metric("Tampered Evidence", f"{verdict.tampered_score:.2f}")
    col_3.metric("Authentic Evidence", f"{verdict.authentic_score:.2f}")
    st.caption(verdict.summary)

    if verdict.signals:
        rows = [
            {
                "Type": signal.source_type,
                "Name": signal.name,
                "Verdict": signal.verdict,
                "Confidence": f"{signal.confidence * 100:.1f}%",
                "Weight": f"{signal.weight:.2f}",
                "Contribution": f"{signal.contribution:.3f}",
                "Note": signal.note,
            }
            for signal in verdict.signals
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
