"""
@file upload_section.py
@brief Streamlit upload component for IFDS.
"""

from __future__ import annotations

from config.settings import STREAMLIT_UPLOAD_EXTENSIONS


def render_upload_section():
    """
    @brief Render a Streamlit file uploader.
    @return Uploaded file object or None.
    """
    import streamlit as st

    return st.file_uploader(
        "Analiz edilecek görüntüyü yükleyin",
        type=STREAMLIT_UPLOAD_EXTENSIONS,
        help="GIF, JPG/JPEG, PNG, BMP ve TIFF desteklenir. Maksimum 50 MB.",
    )
