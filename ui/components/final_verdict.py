"""
@file final_verdict.py
@brief Streamlit final verdict summary component.
"""

from __future__ import annotations

import html

from src.verdict import FinalVerdict


def _verdict_class(label: str) -> str:
    if label == "Tampered":
        return "is-danger"
    if label == "Review needed":
        return "is-warning"
    return "is-success"


def render_final_verdict(verdict: FinalVerdict) -> None:
    """
    @brief Render weighted final verdict.
    @param verdict FinalVerdict instance.
    """
    import pandas as pd
    import streamlit as st

    st.markdown(
        f"""
        <section class="ifds-verdict-panel {_verdict_class(verdict.label)}">
          <div>
            <span class="ifds-eyebrow">Genel karar</span>
            <h2>{html.escape(verdict.label)}</h2>
            <p>{html.escape(verdict.level)}</p>
          </div>
          <div class="ifds-score-ring">
            <strong>{verdict.score * 100:.1f}%</strong>
            <span>Tampering score</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col_1, col_2, col_3 = st.columns(3)
    col_1.metric("Tampered Evidence", f"{verdict.tampered_score:.2f}")
    col_2.metric("Authentic Evidence", f"{verdict.authentic_score:.2f}")
    col_3.metric("Signal Count", str(len(verdict.signals)))
    st.markdown(f"<p class='ifds-summary'>{html.escape(verdict.summary)}</p>", unsafe_allow_html=True)

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
