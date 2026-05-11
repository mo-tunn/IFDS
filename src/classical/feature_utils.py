"""
@file feature_utils.py
@brief Shared feature matching, scoring, and visualization helpers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from config.settings import (
    MIN_DISPLACEMENT_PX,
    RANSAC_REPROJECTION_THRESHOLD,
)


def filter_self_matches(
    keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[Sequence[cv2.DMatch]],
    ratio: float,
    min_displacement_px: float = MIN_DISPLACEMENT_PX,
) -> list[cv2.DMatch]:
    """
    @brief Apply self-match exclusion, displacement filtering, and Lowe ratio.
    @param keypoints Keypoints used for descriptor matching.
    @param matches KNN match groups from an OpenCV matcher.
    @param ratio Lowe ratio threshold.
    @param min_displacement_px Minimum pixel distance between matched points.
    @return Filtered good matches.
    """
    good_matches: list[cv2.DMatch] = []
    seen: set[tuple[int, int]] = set()

    for group in matches:
        if len(group) < 2:
            continue
        first_non_self: cv2.DMatch | None = None
        second_candidate: cv2.DMatch | None = None
        for candidate in group:
            if candidate.queryIdx == candidate.trainIdx:
                continue
            if first_non_self is None:
                first_non_self = candidate
            elif second_candidate is None:
                second_candidate = candidate
                break
        if first_non_self is None or second_candidate is None:
            continue
        if first_non_self.distance > ratio * max(second_candidate.distance, 1e-8):
            continue

        query_point = np.array(keypoints[first_non_self.queryIdx].pt)
        train_point = np.array(keypoints[first_non_self.trainIdx].pt)
        displacement = float(np.linalg.norm(query_point - train_point))
        if displacement < min_displacement_px:
            continue

        pair = tuple(sorted((first_non_self.queryIdx, first_non_self.trainIdx)))
        if pair in seen:
            continue
        seen.add(pair)
        good_matches.append(first_non_self)

    return good_matches


def estimate_inlier_ratio(
    keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
) -> tuple[float, list[cv2.DMatch]]:
    """
    @brief Estimate geometric consistency with RANSAC homography.
    @param keypoints Keypoints used by the match list.
    @param matches Filtered candidate matches.
    @return Tuple of inlier ratio and inlier match list.
    """
    if len(matches) < 4:
        return 0.0, list(matches)

    src_pts = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    try:
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJECTION_THRESHOLD)
    except cv2.error:
        return 0.0, list(matches)
    if mask is None:
        return 0.0, list(matches)

    flags = mask.ravel().astype(bool)
    inliers = [match for match, keep in zip(matches, flags) if keep]
    return float(np.mean(flags)) if len(flags) else 0.0, inliers


def compute_confidence(
    match_count: int,
    total_keypoints: int,
    inlier_ratio: float,
    min_matches: int,
) -> float:
    """
    @brief Convert feature evidence into a normalized confidence score.
    @param match_count Number of filtered matches.
    @param total_keypoints Number of detected keypoints.
    @param inlier_ratio RANSAC inlier ratio in [0, 1].
    @param min_matches Detector-specific minimum match threshold.
    @return Confidence in [0, 1].
    """
    if total_keypoints <= 0 or match_count <= 0:
        return 0.0
    match_strength = min(match_count / max(min_matches * 2.0, 1.0), 1.0)
    density_strength = min(match_count / max(total_keypoints * 0.08, 1.0), 1.0)
    geometry_strength = min(max(inlier_ratio, 0.0), 1.0)
    confidence = 0.50 * match_strength + 0.30 * density_strength + 0.20 * geometry_strength
    return float(min(max(confidence, 0.0), 1.0))


def make_forge_mask(
    shape: tuple[int, ...],
    keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    radius: int = 18,
) -> np.ndarray:
    """
    @brief Build a binary mask from matched keypoint pairs.
    @param shape Source image shape.
    @param keypoints Keypoints used by matches.
    @param matches Filtered or inlier matches.
    @param radius Circle radius for each keypoint.
    @return Binary uint8 mask.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for match in matches:
        for index in (match.queryIdx, match.trainIdx):
            x_coord, y_coord = keypoints[index].pt
            cv2.circle(mask, (int(round(x_coord)), int(round(y_coord))), radius, 255, -1)
    if np.any(mask):
        kernel_size = max(9, int(math.sqrt(shape[0] * shape[1]) * 0.025))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def make_annotated_image(
    image: np.ndarray,
    keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    mask: np.ndarray | None,
) -> np.ndarray:
    """
    @brief Overlay mask and matched keypoint pairs on an RGB image.
    @param image Original RGB image.
    @param keypoints Keypoints used by matches.
    @param matches Matches to draw.
    @param mask Optional binary forge mask.
    @return Annotated RGB image.
    """
    annotated = image.copy()
    if mask is not None and np.any(mask):
        overlay = annotated.copy()
        overlay[mask > 0] = (255, 70, 70)
        annotated = cv2.addWeighted(annotated, 0.72, overlay, 0.28, 0)

    for match in matches[:75]:
        qx, qy = keypoints[match.queryIdx].pt
        tx, ty = keypoints[match.trainIdx].pt
        query_point = (int(round(qx)), int(round(qy)))
        train_point = (int(round(tx)), int(round(ty)))
        cv2.circle(annotated, query_point, 4, (46, 117, 182), -1)
        cv2.circle(annotated, train_point, 4, (31, 56, 100), -1)
        cv2.line(annotated, query_point, train_point, (31, 56, 100), 1)

    return annotated


def blank_result_image(image: np.ndarray, keypoints: Sequence[cv2.KeyPoint]) -> np.ndarray:
    """
    @brief Draw detected keypoints when no suspicious match evidence exists.
    @param image Original RGB image.
    @param keypoints Detected keypoints.
    @return RGB image with keypoints.
    """
    annotated = image.copy()
    return cv2.drawKeypoints(
        image,
        list(keypoints)[:150],
        annotated,
        color=(46, 117, 182),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
