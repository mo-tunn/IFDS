"""
@file upload_section.py
@brief Streamlit upload component for IFDS.
"""

from __future__ import annotations

from config.settings import MAX_FILE_SIZE_MB, STREAMLIT_UPLOAD_EXTENSIONS


def render_upload_section():
    """
    @brief Render a Streamlit file uploader.
    @return Uploaded file object or None.
    """
    import streamlit as st

    return st.file_uploader(
        "Görüntü dosyası",
        type=STREAMLIT_UPLOAD_EXTENSIONS,
        help=f"GIF, JPG/JPEG, PNG, BMP ve TIFF desteklenir. Maksimum {MAX_FILE_SIZE_MB} MB.",
        label_visibility="collapsed",
    )
