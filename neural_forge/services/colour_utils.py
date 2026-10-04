"""Colour conversion, white balance and CIELAB region statistics."""

import base64
import math
from dataclasses import dataclass

import cv2
import numpy as np

EPSILON = 1e-9


@dataclass(frozen=True)
class LabStatistics:
    """Robust CIELAB statistics of a masked region (L in 0-100, a and b roughly -128..127)."""

    lightness: float
    red_green: float
    yellow_blue: float
    lightness_iqr: float
    red_green_iqr: float
    yellow_blue_iqr: float
    pixel_count: int

    @property
    def chroma(self) -> float:
        """Colourfulness sqrt(a^2 + b^2)."""
        return math.hypot(self.red_green, self.yellow_blue)

    def as_array(self) -> np.ndarray:
        """Return (L, a, b) medians as an array."""
        return np.array([self.lightness, self.red_green, self.yellow_blue], dtype=np.float64)


@dataclass(frozen=True)
class LabDifference:
    """Difference between a region and a reference region in CIELAB."""

    delta_lightness: float
    delta_red_green: float
    delta_yellow_blue: float
    delta_e: float


def bgr_to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an 8-bit BGR image to float CIELAB with L in [0, 100]."""
    normalized = image_bgr.astype(np.float32) / 255.0
    return cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)


def masked_lab_statistics(lab_image: np.ndarray, mask: np.ndarray, min_pixels: int = 1) -> LabStatistics | None:
    """Median and IQR of L, a and b inside a mask; None when too few pixels."""
    selection = mask > 0
    pixel_count = int(selection.sum())
    if pixel_count < max(1, min_pixels):
        return None
    pixels = lab_image[selection]
    quartiles = np.percentile(pixels, [25, 50, 75], axis=0)
    return LabStatistics(
        lightness=float(quartiles[1, 0]),
        red_green=float(quartiles[1, 1]),
        yellow_blue=float(quartiles[1, 2]),
        lightness_iqr=float(quartiles[2, 0] - quartiles[0, 0]),
        red_green_iqr=float(quartiles[2, 1] - quartiles[0, 1]),
        yellow_blue_iqr=float(quartiles[2, 2] - quartiles[0, 2]),
        pixel_count=pixel_count,
    )


def delta_e76(first: np.ndarray, second: np.ndarray) -> float:
    """CIE76 colour difference."""
    return float(np.linalg.norm(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)))


def region_difference(region: LabStatistics, reference: LabStatistics) -> LabDifference:
    """Difference of a region relative to a reference region."""
    return LabDifference(
        delta_lightness=region.lightness - reference.lightness,
        delta_red_green=region.red_green - reference.red_green,
        delta_yellow_blue=region.yellow_blue - reference.yellow_blue,
        delta_e=delta_e76(region.as_array(), reference.as_array()),
    )


def individual_typology_angle(statistics: LabStatistics) -> float:
    """Individual typology angle (degrees) describing constitutive skin tone."""
    return math.degrees(math.atan2(statistics.lightness - 50.0, max(statistics.yellow_blue, EPSILON)))


def grey_world_gains(image_bgr: np.ndarray, dark_cutoff: int, bright_cutoff: int) -> np.ndarray:
    """Per-channel BGR gains that make the mean of well-exposed pixels neutral grey."""
    pixels = image_bgr.reshape(-1, 3).astype(np.float64)
    luminance = pixels.mean(axis=1)
    valid = (luminance > dark_cutoff) & (luminance < bright_cutoff)
    if valid.sum() < 100:
        return np.ones(3, dtype=np.float64)
    channel_means = pixels[valid].mean(axis=0)
    grey_level = float(channel_means.mean())
    return grey_level / np.maximum(channel_means, EPSILON)


def apply_channel_gains(image_bgr: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Apply BGR gains and clip to 8-bit."""
    corrected = image_bgr.astype(np.float32) * np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def mean_chroma(lab_image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean chroma of an image region, used to judge grey-world validity."""
    if mask is not None:
        pixels = lab_image[mask > 0]
    else:
        pixels = lab_image.reshape(-1, 3)
    if len(pixels) == 0:
        return 0.0
    return float(np.hypot(pixels[:, 1].mean(), pixels[:, 2].mean()))


def skin_mask_ycrcb(image_bgr: np.ndarray, cr_range: tuple[float, float], cb_range: tuple[float, float]) -> np.ndarray:
    """Skin segmentation in YCrCb space with morphological clean-up; returns a 0/255 mask."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    cr_channel = ycrcb[:, :, 1]
    cb_channel = ycrcb[:, :, 2]
    mask = (
        (cr_channel >= cr_range[0])
        & (cr_channel <= cr_range[1])
        & (cb_channel >= cb_range[0])
        & (cb_channel <= cb_range[1])
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def laplacian_sharpness(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian as a sharpness score."""
    if image_bgr.size == 0:
        return 0.0
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Clip a box to image bounds and round to integers."""
    x_min = int(max(0, min(width - 1, math.floor(box[0]))))
    y_min = int(max(0, min(height - 1, math.floor(box[1]))))
    x_max = int(max(x_min + 1, min(width, math.ceil(box[2]))))
    y_max = int(max(y_min + 1, min(height, math.ceil(box[3]))))
    return x_min, y_min, x_max, y_max


def expand_box(box: tuple[float, float, float, float], padding_fraction: float) -> tuple[float, float, float, float]:
    """Expand a box by a fraction of its size on every side."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (
        box[0] - width * padding_fraction,
        box[1] - height * padding_fraction,
        box[2] + width * padding_fraction,
        box[3] + height * padding_fraction,
    )


def crop_image(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Crop an image to a (clipped) box."""
    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = clip_box(box, width, height)
    return image[y_min:y_max, x_min:x_max].copy()


def encode_jpeg_base64(image_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode an image as base64 JPEG."""
    success, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")
