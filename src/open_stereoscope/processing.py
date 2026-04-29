from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


class RegistrationError(RuntimeError):
    """Raised when a usable overlap cannot be found."""


@dataclass(frozen=True)
class ImageAdjustments:
    brightness: int = 0
    contrast: float = 1.0


@dataclass(frozen=True)
class RegistrationResult:
    fixed_crop: np.ndarray
    moving_crop: np.ndarray
    transform: np.ndarray
    overlap_box: tuple[int, int, int, int]
    method: str
    match_count: int
    confidence: float
    registration_adjustments: ImageAdjustments

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.fixed_crop.shape[:2]
        return width, height


def load_image(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    data = np.frombuffer(file_path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RegistrationError(f"Could not read image: {file_path}")
    return image


def register_pair(
    fixed: np.ndarray, moving: np.ndarray, feature_method: str = "ORB"
) -> RegistrationResult:
    if fixed.ndim != 3 or moving.ndim != 3:
        raise RegistrationError("Images must be color images.")

    fixed_mask = _content_mask(fixed)
    moving_mask = _content_mask(moving)
    registration_moving, registration_adjustments = _normalize_for_registration(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
    )
    transform, method, match_count, confidence = _estimate_transform(
        fixed,
        registration_moving,
        fixed_mask,
        moving_mask,
        feature_method,
    )
    warped, overlap_mask = _warp_to_fixed_space(
        moving,
        moving_mask,
        fixed_mask,
        transform,
        fixed.shape[:2],
    )

    box = _mask_box(overlap_mask)
    if box is None:
        raise RegistrationError("No overlapping area was found after registration.")

    x, y, width, height = box
    min_side = min(width, height)
    if min_side < 32:
        raise RegistrationError("The detected overlap is too small to export.")

    fixed_crop = fixed[y : y + height, x : x + width].copy()
    moving_crop = warped[y : y + height, x : x + width].copy()

    return RegistrationResult(
        fixed_crop=fixed_crop,
        moving_crop=moving_crop,
        transform=transform,
        overlap_box=box,
        method=method,
        match_count=match_count,
        confidence=confidence,
        registration_adjustments=registration_adjustments,
    )


def build_wiggle_frames(
    result: RegistrationResult,
    fixed_adjustments: ImageAdjustments | None = None,
    moving_adjustments: ImageAdjustments | None = None,
) -> list[np.ndarray]:
    fixed_crop = apply_adjustments(
        result.fixed_crop,
        fixed_adjustments or ImageAdjustments(),
    )
    moving_crop = apply_adjustments(
        result.moving_crop,
        moving_adjustments or ImageAdjustments(),
    )
    return [
        _bgr_to_rgb(fixed_crop),
        _bgr_to_rgb(moving_crop),
    ]


def build_animation_frames(
    result: RegistrationResult,
    fixed_adjustments: ImageAdjustments | None = None,
    moving_adjustments: ImageAdjustments | None = None,
    animation_mode: str = "wiggle",
) -> list[np.ndarray]:
    normalized_mode = animation_mode.strip().lower()
    if normalized_mode == "smooth":
        return build_smooth_interpolation_frames(
            result,
            fixed_adjustments,
            moving_adjustments,
        )
    return build_wiggle_frames(result, fixed_adjustments, moving_adjustments)


def build_smooth_interpolation_frames(
    result: RegistrationResult,
    fixed_adjustments: ImageAdjustments | None = None,
    moving_adjustments: ImageAdjustments | None = None,
    steps: int = 8,
) -> list[np.ndarray]:
    fixed_crop = apply_adjustments(
        result.fixed_crop,
        fixed_adjustments or ImageAdjustments(),
    )
    moving_crop = apply_adjustments(
        result.moving_crop,
        moving_adjustments or ImageAdjustments(),
    )

    flow_forward = _dense_flow(fixed_crop, moving_crop)
    flow_backward = _dense_flow(moving_crop, fixed_crop)

    forward_values = np.linspace(0.0, 1.0, steps + 1)
    backward_values = np.linspace(1.0, 0.0, steps + 1)[1:-1]
    frames = []
    for amount in np.concatenate([forward_values, backward_values]):
        frame = _interpolate_with_flow(
            fixed_crop,
            moving_crop,
            flow_forward,
            flow_backward,
            float(amount),
        )
        frames.append(_bgr_to_rgb(frame))
    return frames


def apply_adjustments(image: np.ndarray, adjustments: ImageAdjustments) -> np.ndarray:
    contrast = float(np.clip(adjustments.contrast, 0.0, 3.0))
    brightness = int(np.clip(adjustments.brightness, -255, 255))
    if brightness == 0 and contrast == 1.0:
        return image.copy()

    adjusted = image.astype(np.float32) * contrast + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def estimate_adjustments_to_match(
    reference: np.ndarray,
    target: np.ndarray,
    min_brightness: int = -100,
    max_brightness: int = 100,
    min_contrast: float = 0.0,
    max_contrast: float = 2.0,
) -> ImageAdjustments:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)

    reference_mean, reference_std = cv2.meanStdDev(reference_gray)
    target_mean, target_std = cv2.meanStdDev(target_gray)

    reference_mean_value = float(reference_mean[0][0])
    reference_std_value = max(float(reference_std[0][0]), 1.0)
    target_mean_value = float(target_mean[0][0])
    target_std_value = max(float(target_std[0][0]), 1.0)

    contrast = reference_std_value / target_std_value
    contrast = float(np.clip(contrast, min_contrast, max_contrast))
    brightness = reference_mean_value - target_mean_value * contrast
    brightness = int(round(np.clip(brightness, min_brightness, max_brightness)))

    return ImageAdjustments(brightness=brightness, contrast=contrast)


def export_gif(
    result: RegistrationResult,
    output_path: str | Path,
    delay_ms: int,
    fixed_adjustments: ImageAdjustments | None = None,
    moving_adjustments: ImageAdjustments | None = None,
    animation_mode: str = "wiggle",
) -> None:
    frames = build_animation_frames(
        result,
        fixed_adjustments,
        moving_adjustments,
        animation_mode,
    )
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(20, int(delay_ms)),
        loop=0,
        disposal=2,
    )


def export_mp4(
    result: RegistrationResult,
    output_path: str | Path,
    delay_ms: int,
    fixed_adjustments: ImageAdjustments | None = None,
    moving_adjustments: ImageAdjustments | None = None,
    animation_mode: str = "wiggle",
) -> None:
    frames = _make_even_sized(
        build_animation_frames(
            result,
            fixed_adjustments,
            moving_adjustments,
            animation_mode,
        )
    )
    fps = max(1.0, 1000.0 / float(delay_ms))
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=8) as writer:
        cycle_count = 8 if animation_mode.strip().lower() == "wiggle" else 4
        for _ in range(cycle_count):
            for frame in frames:
                writer.append_data(frame)


def _estimate_transform(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    feature_method: str,
) -> tuple[np.ndarray, str, int, float]:
    fixed_gray = _prepare_gray(fixed, fixed_mask)
    moving_gray = _prepare_gray(moving, moving_mask)

    try:
        normalized_method = feature_method.strip().upper()
        if normalized_method == "SIFT":
            return _estimate_affine_with_sift(
                fixed_gray,
                moving_gray,
                fixed_mask,
                moving_mask,
            )
        return _estimate_affine_with_orb(
            fixed_gray,
            moving_gray,
            fixed_mask,
            moving_mask,
        )
    except RegistrationError:
        return _estimate_translation_with_phase_correlation(
            fixed_gray,
            moving_gray,
            fixed_mask,
            moving_mask,
        )


def _estimate_affine_with_orb(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[np.ndarray, str, int, float]:
    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    fixed_keypoints, fixed_descriptors = orb.detectAndCompute(fixed_gray, fixed_mask)
    moving_keypoints, moving_descriptors = orb.detectAndCompute(moving_gray, moving_mask)

    if fixed_descriptors is None or moving_descriptors is None:
        raise RegistrationError("Not enough feature detail was found for ORB registration.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(moving_descriptors, fixed_descriptors, k=2)
    good_matches = []
    for candidate in matches:
        if len(candidate) != 2:
            continue
        best, next_best = candidate
        if best.distance < 0.76 * next_best.distance:
            good_matches.append(best)

    if len(good_matches) < 8:
        raise RegistrationError("Not enough matching features were found.")

    source_points = np.float32(
        [moving_keypoints[match.queryIdx].pt for match in good_matches]
    )
    target_points = np.float32(
        [fixed_keypoints[match.trainIdx].pt for match in good_matches]
    )

    transform, inliers = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=4000,
        confidence=0.995,
    )
    if transform is None or inliers is None:
        raise RegistrationError("Feature registration did not converge.")

    inlier_count = int(inliers.sum())
    if inlier_count < 6:
        raise RegistrationError("Feature registration did not produce enough inliers.")

    confidence = inlier_count / max(1, len(good_matches))
    return transform.astype(np.float32), "ORB + RANSAC affine", inlier_count, confidence


def _estimate_affine_with_sift(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[np.ndarray, str, int, float]:
    if not hasattr(cv2, "SIFT_create"):
        raise RegistrationError("SIFT is not available in this OpenCV build.")

    sift = cv2.SIFT_create(nfeatures=6000)
    fixed_keypoints, fixed_descriptors = sift.detectAndCompute(fixed_gray, fixed_mask)
    moving_keypoints, moving_descriptors = sift.detectAndCompute(moving_gray, moving_mask)

    if fixed_descriptors is None or moving_descriptors is None:
        raise RegistrationError("Not enough feature detail was found for SIFT registration.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(moving_descriptors, fixed_descriptors, k=2)
    good_matches = []
    for candidate in matches:
        if len(candidate) != 2:
            continue
        best, next_best = candidate
        if best.distance < 0.75 * next_best.distance:
            good_matches.append(best)

    if len(good_matches) < 8:
        raise RegistrationError("Not enough matching SIFT features were found.")

    source_points = np.float32(
        [moving_keypoints[match.queryIdx].pt for match in good_matches]
    )
    target_points = np.float32(
        [fixed_keypoints[match.trainIdx].pt for match in good_matches]
    )

    transform, inliers = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=4000,
        confidence=0.995,
    )
    if transform is None or inliers is None:
        raise RegistrationError("SIFT registration did not converge.")

    inlier_count = int(inliers.sum())
    if inlier_count < 6:
        raise RegistrationError("SIFT registration did not produce enough inliers.")

    confidence = inlier_count / max(1, len(good_matches))
    return transform.astype(np.float32), "SIFT + RANSAC affine", inlier_count, confidence


def _estimate_translation_with_phase_correlation(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[np.ndarray, str, int, float]:
    fixed_box = _simple_mask_box(fixed_mask)
    moving_box = _simple_mask_box(moving_mask)
    if fixed_box is None or moving_box is None:
        raise RegistrationError("No usable image content was found for registration.")

    fixed_x, fixed_y, fixed_width, fixed_height = fixed_box
    moving_x, moving_y, moving_width, moving_height = moving_box
    fixed_content = fixed_gray[
        fixed_y : fixed_y + fixed_height,
        fixed_x : fixed_x + fixed_width,
    ]
    moving_content = moving_gray[
        moving_y : moving_y + moving_height,
        moving_x : moving_x + moving_width,
    ]

    height = min(fixed_content.shape[0], moving_content.shape[0])
    width = min(fixed_content.shape[1], moving_content.shape[1])
    if height < 32 or width < 32:
        raise RegistrationError("Images are too small for registration.")

    fixed_crop = _center_crop(fixed_content, width, height).astype(np.float32)
    moving_crop = _center_crop(moving_content, width, height).astype(np.float32)

    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(fixed_crop * window, moving_crop * window)
    dx, dy = shift

    fixed_origin_x = fixed_x + (fixed_width - width) / 2.0
    fixed_origin_y = fixed_y + (fixed_height - height) / 2.0
    moving_origin_x = moving_x + (moving_width - width) / 2.0
    moving_origin_y = moving_y + (moving_height - height) / 2.0

    transform = np.array(
        [
            [1.0, 0.0, fixed_origin_x - moving_origin_x - dx],
            [0.0, 1.0, fixed_origin_y - moving_origin_y - dy],
        ],
        dtype=np.float32,
    )

    if response < 0.05:
        raise RegistrationError("Registration confidence is too low.")

    return transform, "phase correlation translation", 0, float(response)


def _warp_to_fixed_space(
    moving: np.ndarray,
    moving_mask: np.ndarray,
    fixed_mask: np.ndarray,
    transform: np.ndarray,
    fixed_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = fixed_size
    warped = cv2.warpAffine(
        moving,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    warped_moving_mask = cv2.warpAffine(
        moving_mask,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    overlap_mask = cv2.bitwise_and(fixed_mask, warped_moving_mask)
    overlap_mask = cv2.erode(overlap_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return warped, overlap_mask


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coordinates = cv2.findNonZero(mask)
    if coordinates is None:
        return None
    x, y, width, height = cv2.boundingRect(coordinates)

    submask = mask[y : y + height, x : x + width] > 0
    row_range = _longest_true_run(submask.mean(axis=1) > 0.65)
    column_range = _longest_true_run(submask.mean(axis=0) > 0.65)
    if row_range is None or column_range is None:
        return int(x), int(y), int(width), int(height)

    row_start, row_end = row_range
    column_start, column_end = column_range
    cropped_width = column_end - column_start
    cropped_height = row_end - row_start
    if min(cropped_width, cropped_height) < 32:
        return int(x), int(y), int(width), int(height)

    return (
        int(x + column_start),
        int(y + row_start),
        int(cropped_width),
        int(cropped_height),
    )


def _content_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    near_black = gray <= _black_border_threshold(gray)
    component_count, labels = cv2.connectedComponents(
        near_black.astype(np.uint8),
        connectivity=8,
    )
    if component_count <= 1:
        return np.full(gray.shape, 255, dtype=np.uint8)

    edge_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    edge_labels = edge_labels[edge_labels != 0]
    if edge_labels.size == 0:
        return np.full(gray.shape, 255, dtype=np.uint8)

    border_black = np.isin(labels, edge_labels)
    invalid = cv2.dilate(
        border_black.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    valid = ~invalid

    if np.count_nonzero(valid) < gray.size * 0.05:
        return np.full(gray.shape, 255, dtype=np.uint8)
    return (valid.astype(np.uint8) * 255)


def _normalize_for_registration(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[np.ndarray, ImageAdjustments]:
    adjustments = _estimate_masked_adjustments(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
    )
    return apply_adjustments(moving, adjustments), adjustments


def _estimate_masked_adjustments(
    reference: np.ndarray,
    target: np.ndarray,
    reference_mask: np.ndarray,
    target_mask: np.ndarray,
) -> ImageAdjustments:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    reference_values = reference_gray[reference_mask > 0]
    target_values = target_gray[target_mask > 0]
    if reference_values.size < 1024 or target_values.size < 1024:
        return ImageAdjustments()

    reference_low, reference_high = np.percentile(reference_values, [5, 95])
    target_low, target_high = np.percentile(target_values, [5, 95])
    target_range = max(float(target_high - target_low), 1.0)
    reference_range = max(float(reference_high - reference_low), 1.0)

    contrast = float(np.clip(reference_range / target_range, 0.5, 2.0))
    reference_mid = float((reference_low + reference_high) / 2.0)
    target_mid = float((target_low + target_high) / 2.0)
    brightness = int(round(np.clip(reference_mid - target_mid * contrast, -100, 100)))

    return ImageAdjustments(brightness=brightness, contrast=contrast)


def _black_border_threshold(gray: np.ndarray) -> int:
    edge_values = np.concatenate(
        [
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1],
        ]
    )
    return int(np.clip(np.percentile(edge_values, 60) + 3, 12, 32))


def _simple_mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coordinates = cv2.findNonZero(mask)
    if coordinates is None:
        return None
    x, y, width, height = cv2.boundingRect(coordinates)
    return int(x), int(y), int(width), int(height)


def _prepare_gray(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mask is not None and np.any(mask > 0):
        valid_pixels = gray[mask > 0]
        fill_value = int(np.median(valid_pixels))
        gray = gray.copy()
        gray[mask == 0] = fill_value
    gray = cv2.equalizeHist(gray)
    return gray


def _center_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    start_x = max(0, (image.shape[1] - width) // 2)
    start_y = max(0, (image.shape[0] - height) // 2)
    return image[start_y : start_y + height, start_x : start_x + width]


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _make_even_sized(frames: list[np.ndarray]) -> list[np.ndarray]:
    height, width = frames[0].shape[:2]
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    return [frame[:even_height, :even_width].copy() for frame in frames]


def _dense_flow(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    height, width = first_gray.shape
    max_dimension = max(height, width)
    scale = min(1.0, 1200.0 / float(max_dimension))
    if scale < 1.0:
        flow_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        first_flow = cv2.resize(first_gray, flow_size, interpolation=cv2.INTER_AREA)
        second_flow = cv2.resize(second_gray, flow_size, interpolation=cv2.INTER_AREA)
    else:
        first_flow = first_gray
        second_flow = second_gray

    flow = cv2.calcOpticalFlowFarneback(
        first_flow,
        second_flow,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=35,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )

    if scale < 1.0:
        small_height, small_width = first_flow.shape
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow[:, :, 0] *= width / float(small_width)
        flow[:, :, 1] *= height / float(small_height)
    return flow.astype(np.float32)


def _interpolate_with_flow(
    first: np.ndarray,
    second: np.ndarray,
    flow_forward: np.ndarray,
    flow_backward: np.ndarray,
    amount: float,
) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    height, width = first.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    first_map_x = grid_x - flow_forward[:, :, 0] * amount
    first_map_y = grid_y - flow_forward[:, :, 1] * amount
    second_map_x = grid_x - flow_backward[:, :, 0] * (1.0 - amount)
    second_map_y = grid_y - flow_backward[:, :, 1] * (1.0 - amount)

    first_warped = cv2.remap(
        first,
        first_map_x,
        first_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    second_warped = cv2.remap(
        second,
        second_map_x,
        second_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return cv2.addWeighted(first_warped, 1.0 - amount, second_warped, amount, 0.0)


def _longest_true_run(values: np.ndarray) -> tuple[int, int] | None:
    best_start = 0
    best_end = 0
    current_start: int | None = None

    for index, value in enumerate(values):
        if value and current_start is None:
            current_start = index
        elif not value and current_start is not None:
            if index - current_start > best_end - best_start:
                best_start, best_end = current_start, index
            current_start = None

    if current_start is not None and len(values) - current_start > best_end - best_start:
        best_start, best_end = current_start, len(values)

    if best_end == best_start:
        return None
    return best_start, best_end
