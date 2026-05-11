"""
@file visualization.py
@brief Image visualization helpers.
"""

from __future__ import annotations

import cv2
import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 70, 70),
    alpha: float = 0.35,
) -> np.ndarray:
    """
    @brief Overlay a binary mask on an RGB image.
    @param image RGB source image.
    @param mask Binary mask.
    @param color RGB overlay color.
    @param alpha Overlay opacity.
    @return RGB annotated image.
    """
    output = image.copy()
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = output.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(output, 1.0 - alpha, overlay, alpha, 0)


def heatmap_to_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    """
    @brief Convert a heatmap to a colored RGB overlay.
    @param image RGB image in uint8 or normalized float.
    @param heatmap Heatmap with values in [0, 1].
    @param alpha Heatmap opacity.
    @return RGB overlay.
    """
    base = image
    if base.dtype != np.uint8:
        base = np.clip(base * 255, 0, 255).astype(np.uint8)
    resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1.0 - alpha, colored_rgb, alpha, 0)
