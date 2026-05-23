"""
@file app.py
@brief Streamlit entrypoint for the Image Forgery Detection System.
@usage streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import EFFICIENTNET_MODEL_PATH, XCEPTION_MODEL_PATH
from src.analysis_service import AnalysisSelection, AnalysisService
from src.preprocessing.image_loader import ImageLoader
from src.reporting.report_generator import ReportGenerator
from src.verdict import VerdictService
from ui.components.comparison_table import render_comparison_table
from ui.components.final_verdict import render_final_verdict
from ui.components.heatmap_viewer import render_heatmap_viewer
from ui.components.results_grid import render_results_grid
from ui.components.upload_section import render_upload_section


st.set_page_config(
    page_title="IFDS - Image Forgery Detection",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_STATE_KEY = "ifds_last_analysis"


def load_theme() -> None:
    """
    @brief Load custom Streamlit CSS when the theme file exists.
    """
    theme_path = Path("ui/styles/theme.css")
    if theme_path.exists():
        st.markdown(f"<style>{theme_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _selection_count(selection: AnalysisSelection) -> int:
    return sum(
        [
            selection.run_sift,
            selection.run_surf,
            selection.run_akaze,
            selection.run_orb,
            selection.run_xception,
            selection.run_efficientnet,
        ]
    )


def _file_digest(file_bytes: bytes, filename: str) -> str:
    digest = hashlib.sha256()
    digest.update(filename.encode("utf-8", errors="ignore"))
    digest.update(file_bytes)
    return digest.hexdigest()


def _selection_signature(selection: AnalysisSelection) -> tuple[bool, ...]:
    return (
        selection.run_sift,
        selection.run_surf,
        selection.run_akaze,
        selection.run_orb,
        selection.run_xception,
        selection.run_efficientnet,
        selection.run_gradcam,
    )


def _report_stem(filename: object) -> str:
    stem = Path(str(filename)).stem or "image"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe[:80] or "image"


def _render_page_header() -> None:
    st.markdown(
        """
        <section class="ifds-page-header">
          <div>
            <span class="ifds-eyebrow">Image Forgery Detection System</span>
            <h1>IFDS Analysis Console</h1>
            <p>Image manipulation analysis workspace powered by SIFT, SURF, AKAZE, ORB, and optional AI models.</p>
          </div>
          <div class="ifds-header-badges">
            <span>Classical</span>
            <span>AI</span>
            <span>Report</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_readiness(selection: AnalysisSelection) -> None:
    selected = _selection_count(selection)
    model_count = int(XCEPTION_MODEL_PATH.exists()) + int(EFFICIENTNET_MODEL_PATH.exists())
    classical_count = sum([selection.run_sift, selection.run_surf, selection.run_akaze, selection.run_orb])
    st.markdown(
        f"""
        <div class="ifds-status-strip">
          <div><strong>{selected}</strong><span>Selected analyzers</span></div>
          <div><strong>{classical_count}</strong><span>Classical methods</span></div>
          <div><strong>{model_count}/2</strong><span>Model files found</span></div>
          <div><strong>50 MB</strong><span>Upload limit</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_upload_shell() -> Any:
    st.markdown(
        """
        <div class="ifds-section-label">
          <span>1</span>
          <div>
            <strong>Upload image</strong>
            <p>Choose a JPG, PNG, GIF, BMP, or TIFF file.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return render_upload_section()


def _render_analysis_prompt() -> None:
    st.markdown(
        """
        <div class="ifds-empty-state">
          <strong>Waiting for an image</strong>
          <span>Upload a file to reveal the preview, metadata, and analysis controls.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metadata_cards(metadata: dict[str, object]) -> None:
    items = [
        ("Format", metadata["format"]),
        ("Size", f"{metadata['size_mb']:.3f} MB"),
        ("Resolution", f"{metadata['width']} x {metadata['height']}"),
        ("Channels", metadata["channels"]),
    ]
    cards = "\n".join(
        f"<div><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>"
        for label, value in items
    )
    st.markdown(
        f"""
        <div class="ifds-metadata-grid">
          {cards}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> AnalysisSelection:
    """
    @brief Render analysis controls in the sidebar.
    @return AnalysisSelection from user choices.
    """
    with st.sidebar:
        st.markdown("<div class='ifds-sidebar-brand'><strong>IFDS</strong><span>Analysis setup</span></div>", unsafe_allow_html=True)

        st.subheader("Classical Algorithms")
        run_sift = st.toggle("SIFT", value=True, help="Local feature analysis that is robust to scale and rotation.")
        run_surf = st.toggle("SURF", value=True, help="Runs when OpenCV nonfree support is available; otherwise it is skipped.")
        run_akaze = st.toggle("AKAZE", value=True, help="Fast analysis based on binary descriptors.")
        run_orb = st.toggle("ORB", value=True, help="Fast, lightweight feature matching.")

        st.subheader("AI Models")
        run_xception = st.toggle("Xception CNN", value=True, help="Runs AI classification when the model file is available.")
        run_efficientnet = st.toggle("EfficientNet CNN + LSTM", value=True, help="Provides a second AI opinion.")
        run_gradcam = st.toggle("Grad-CAM", value=True, disabled=not run_xception, help="Generates a heatmap for the Xception result.")

        st.divider()
        model_rows = [
            ("Xception", XCEPTION_MODEL_PATH),
            ("EfficientNet + LSTM", EFFICIENTNET_MODEL_PATH),
        ]
        for label, path in model_rows:
            if path.exists():
                st.success(f"{label} model ready")
            else:
                st.warning(f"{label} weights missing")
        st.caption("If model files are missing, classical analysis and reporting still continue.")

    return AnalysisSelection(
        run_sift=run_sift,
        run_surf=run_surf,
        run_akaze=run_akaze,
        run_orb=run_orb,
        run_xception=run_xception,
        run_efficientnet=run_efficientnet,
        run_gradcam=run_gradcam,
    )


def render_metadata(metadata: dict[str, object]) -> None:
    """
    @brief Render image metadata in a compact table.
    @param metadata Image metadata dictionary.
    """
    st.dataframe(
        [
            {"Property": "Filename", "Value": str(metadata["filename"])},
            {"Property": "Format", "Value": str(metadata["format"])},
            {"Property": "Size", "Value": f"{metadata['size_mb']:.3f} MB"},
            {"Property": "Resolution", "Value": f"{metadata['width']} x {metadata['height']} px"},
            {"Property": "Channels", "Value": str(metadata["channels"])},
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """
    @brief Run the Streamlit application.
    """
    load_theme()
    selection = render_sidebar()

    _render_page_header()
    _render_readiness(selection)

    uploaded_file = _render_upload_shell()
    if uploaded_file is None:
        _render_analysis_prompt()
        return

    try:
        file_bytes = uploaded_file.getvalue()
        image, metadata = ImageLoader.load_from_bytes(file_bytes, uploaded_file.name)
    except ValueError as exc:
        st.error(str(exc))
        return

    file_key = _file_digest(file_bytes, uploaded_file.name)
    selection_key = _selection_signature(selection)

    preview_col, metadata_col = st.columns([1.05, 1], gap="large")
    with preview_col:
        st.markdown(
            """
            <div class="ifds-section-label">
              <span>2</span>
              <div><strong>Preview</strong><p>Review the file before running analysis.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(image, caption=f"Uploaded image: {metadata['filename']}", use_column_width=True)
    with metadata_col:
        st.markdown(
            """
            <div class="ifds-section-label">
              <span>3</span>
              <div><strong>Image details</strong><p>File properties and technical summary.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_metadata_cards(metadata)
        render_metadata(metadata)

    st.markdown("<div class='ifds-runbar'>", unsafe_allow_html=True)
    run_disabled = _selection_count(selection) == 0
    run_clicked = st.button("Start Analysis", type="primary", use_container_width=True, disabled=run_disabled)
    st.markdown("</div>", unsafe_allow_html=True)
    if run_disabled:
        st.warning("Select at least one analysis method.")

    cached = st.session_state.get(APP_STATE_KEY)
    can_reuse = bool(cached and cached.get("file_key") == file_key and cached.get("selection_key") == selection_key)

    if run_clicked and not run_disabled:
        progress = st.progress(0, text="Starting analysis...")
        service = AnalysisService()
        progress.progress(15, text="Preparing analyzers...")
        bundle = service.run(image, selection)
        results = AnalysisService.to_dict(bundle)
        final_verdict = VerdictService.evaluate(results)
        results["final_verdict"] = final_verdict
        progress.progress(100, text="Analysis completed.")
        progress.empty()
        st.session_state[APP_STATE_KEY] = {
            "file_key": file_key,
            "selection_key": selection_key,
            "metadata": metadata,
            "results": results,
        }
    elif can_reuse:
        results = cached["results"]
    else:
        st.markdown(
            """
            <div class="ifds-empty-state is-compact">
              <strong>Ready to analyze</strong>
              <span>Press Start Analysis to generate results with the selected methods.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    final_verdict = results["final_verdict"]

    st.subheader("Overall Verdict")
    render_final_verdict(final_verdict)

    if results["classical"]:
        st.subheader("Classical Algorithm Results")
        render_results_grid(results["classical"])

    if results["ai"]:
        st.subheader("AI Model Results")
        ai_columns = st.columns(min(2, max(1, len(results["ai"]))))
        for index, result in enumerate(results["ai"].values()):
            with ai_columns[index % len(ai_columns)]:
                if result.error_message and result.class_label == "Unavailable":
                    st.warning(f"{result.model_name}: {result.error_message}")
                else:
                    st.markdown(
                        f"""
                        <div class="ifds-result-card {'is-danger' if result.is_forged else 'is-success'}">
                          <div class="ifds-card-topline">
                            <span>{html.escape(result.model_name)}</span>
                            <strong>{result.confidence * 100:.1f}%</strong>
                          </div>
                          <div class="ifds-card-verdict">{html.escape(result.class_label)}</div>
                          <div class="ifds-card-meta"><span>{result.processing_time:.3f}s</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if result.error_message:
                        st.caption(result.error_message)
                    render_heatmap_viewer(result)

    st.subheader("Comparative Analysis")
    render_comparison_table(results)

    st.divider()
    report = ReportGenerator()
    report_col_1, report_col_2 = st.columns(2)
    report_stem = _report_stem(metadata["filename"])
    with report_col_1:
        try:
            pdf_bytes = report.generate_pdf(image, metadata, results)
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"ifds_report_{report_stem}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"PDF report could not be generated: {exc}")
    with report_col_2:
        html_report = report.generate_html(image, metadata, results)
        st.download_button(
            "Download HTML Report",
            data=html_report,
            file_name=f"ifds_report_{report_stem}.html",
            mime="text/html",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
