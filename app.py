"""
@file app.py
@brief Streamlit entrypoint for the Image Forgery Detection System.
@usage streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

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


def load_theme() -> None:
    """
    @brief Load custom Streamlit CSS when the theme file exists.
    """
    theme_path = Path("ui/styles/theme.css")
    if theme_path.exists():
        st.markdown(f"<style>{theme_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_sidebar() -> AnalysisSelection:
    """
    @brief Render analysis controls in the sidebar.
    @return AnalysisSelection from user choices.
    """
    with st.sidebar:
        st.title("IFDS")
        st.caption("Image Forgery Detection System")

        st.subheader("Klasik Algoritmalar")
        run_sift = st.checkbox("SIFT", value=True)
        run_surf = st.checkbox("SURF", value=True)
        run_akaze = st.checkbox("AKAZE", value=True)
        run_orb = st.checkbox("ORB", value=True)

        st.subheader("AI Modelleri")
        run_xception = st.checkbox("Xception CNN", value=True)
        run_efficientnet = st.checkbox("EfficientNet CNN", value=True)
        run_gradcam = st.checkbox("Grad-CAM", value=True)

        st.divider()
        if not XCEPTION_MODEL_PATH.exists():
            st.warning(f"Xception ağırlığı bulunamadı: {XCEPTION_MODEL_PATH}")
        if not EFFICIENTNET_MODEL_PATH.exists():
            st.warning(f"EfficientNet ağırlığı bulunamadı: {EFFICIENTNET_MODEL_PATH}")
        st.info("Desteklenen formatlar: GIF, JPG/JPEG, PNG, BMP, TIFF. Maksimum 50 MB.")

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
            {"Özellik": "Dosya Adı", "Değer": metadata["filename"]},
            {"Özellik": "Format", "Değer": metadata["format"]},
            {"Özellik": "Boyut", "Değer": f"{metadata['size_mb']:.3f} MB"},
            {"Özellik": "Çözünürlük", "Değer": f"{metadata['width']} x {metadata['height']} px"},
            {"Özellik": "Kanal", "Değer": metadata["channels"]},
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

    st.title("Image Forgery Detection System")
    st.caption("SIFT | SURF | AKAZE | ORB | Xception CNN | EfficientNet CNN")

    uploaded_file = render_upload_section()
    if uploaded_file is None:
        st.info("Başlamak için bir görüntü yükleyin.")
        return

    try:
        file_bytes = uploaded_file.getvalue()
        image, metadata = ImageLoader.load_from_bytes(file_bytes, uploaded_file.name)
    except ValueError as exc:
        st.error(str(exc))
        return

    preview_col, metadata_col = st.columns([1.1, 1])
    with preview_col:
        st.image(image, caption="Yüklenen görüntü", use_column_width=True)
    with metadata_col:
        st.subheader("Görüntü Bilgileri")
        render_metadata(metadata)

    st.divider()
    if not st.button("Analizi Başlat", type="primary", use_container_width=True):
        return

    progress = st.progress(0, text="Analiz başlıyor...")
    service = AnalysisService()
    progress.progress(15, text="Algoritmalar hazırlanıyor...")
    bundle = service.run(image, selection)
    results = AnalysisService.to_dict(bundle)
    final_verdict = VerdictService.evaluate(results)
    results["final_verdict"] = final_verdict
    progress.progress(100, text="Analiz tamamlandı.")
    progress.empty()

    st.subheader("Genel Karar")
    render_final_verdict(final_verdict)

    if results["classical"]:
        st.subheader("Klasik Algoritma Sonuçları")
        render_results_grid(results["classical"])

    if results["ai"]:
        st.subheader("AI Model Sonuçları")
        ai_columns = st.columns(min(2, max(1, len(results["ai"]))))
        for index, result in enumerate(results["ai"].values()):
            with ai_columns[index % len(ai_columns)]:
                if result.error_message and result.class_label == "Unavailable":
                    st.warning(f"{result.model_name}: {result.error_message}")
                else:
                    st.metric(result.model_name, result.class_label, f"{result.confidence * 100:.1f}%")
                    if result.error_message:
                        st.caption(result.error_message)
                    render_heatmap_viewer(result)

    st.subheader("Karşılaştırmalı Analiz")
    render_comparison_table(results)

    st.divider()
    report = ReportGenerator()
    report_col_1, report_col_2 = st.columns(2)
    with report_col_1:
        try:
            pdf_bytes = report.generate_pdf(image, metadata, results)
            st.download_button(
                "PDF Rapor İndir",
                data=pdf_bytes,
                file_name=f"ifds_report_{metadata['filename']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"PDF raporu üretilemedi: {exc}")
    with report_col_2:
        html_report = report.generate_html(image, metadata, results)
        st.download_button(
            "HTML Rapor İndir",
            data=html_report,
            file_name=f"ifds_report_{metadata['filename']}.html",
            mime="text/html",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
