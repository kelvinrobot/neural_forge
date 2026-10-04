"""Colour calibration agent: reference-card or grey-world white balance and CIELAB conversion."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from neural_forge.core.frame_context import FrameContext
from neural_forge.schemas.readings import CalibrationInfo
from neural_forge.services.colour_utils import apply_channel_gains, bgr_to_lab, grey_world_gains, mean_chroma


@dataclass(frozen=True)
class CardDetection:
    """A detected neutral reference card."""

    gains: np.ndarray
    confidence: float
    area_fraction: float


class ColourCalibrationAgent:
    """Corrects colour per frame and records calibration confidence for colour measurements."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._smoothed_gains: np.ndarray | None = None
        self._reference_samples: list[np.ndarray] = []
        self._session_reference_gains: np.ndarray | None = None
        self._reference_grey_world_gains: np.ndarray | None = None
        self._method_counts: Counter[str] = Counter()
        self._confidence_sum = 0.0
        self._frames_calibrated = 0

    def reference_progress(self) -> float:
        """Fraction of reference-card frames collected during the colour reference step."""
        required = max(1, int(self._config["reference_frames_required"]))
        if self._session_reference_gains is not None:
            return 1.0
        return min(1.0, len(self._reference_samples) / required)

    def apply(self, frame: FrameContext, capture_reference: bool, card_hint: bool) -> CalibrationInfo:
        """Calibrate one frame, store the corrected image and CIELAB image on the frame."""
        image = frame.image_bgr
        grey_world = self._clip_gains(
            grey_world_gains(image, int(self._config["grey_world_dark_cutoff"]), int(self._config["grey_world_bright_cutoff"]))
        )
        card: CardDetection | None = None
        if capture_reference or card_hint:
            card = self._detect_reference_card(image)
        if card is not None:
            gains = card.gains
            method = "reference_card"
            confidence = card.confidence
            if capture_reference:
                self._reference_samples.append(card.gains)
                if len(self._reference_samples) >= int(self._config["reference_frames_required"]):
                    self._session_reference_gains = np.median(np.stack(self._reference_samples), axis=0)
                    self._reference_grey_world_gains = grey_world
        elif self._session_reference_gains is not None:
            gains = self._session_reference_gains
            method = "session_reference"
            illumination_change = 0.0
            if self._reference_grey_world_gains is not None:
                illumination_change = float(
                    np.max(np.abs(grey_world - self._reference_grey_world_gains) / np.maximum(self._reference_grey_world_gains, 1e-6))
                )
            tolerance = float(self._config["illumination_change_tolerance"])
            penalty = min(1.0, illumination_change / max(tolerance, 1e-6)) * 0.5
            confidence = float(self._config["session_reference_confidence"]) * (1.0 - penalty)
        else:
            smoothing = float(self._config["smoothing_factor"])
            if self._smoothed_gains is None:
                self._smoothed_gains = grey_world
            else:
                self._smoothed_gains = (1.0 - smoothing) * self._smoothed_gains + smoothing * grey_world
            gains = self._smoothed_gains
            method = "grey_world"
            scene_bias = mean_chroma(bgr_to_lab(cv2.resize(image, (80, max(1, int(80 * image.shape[0] / image.shape[1]))))))
            chroma_limit = float(self._config["grey_world_chroma_limit"])
            confidence = float(self._config["grey_world_base_confidence"]) * (1.0 - min(1.0, scene_bias / max(chroma_limit, 1e-6)) * 0.6)
        gains = self._clip_gains(gains)
        calibrated = apply_channel_gains(image, gains)
        frame.calibrated_bgr = calibrated
        frame.lab_image = bgr_to_lab(calibrated)
        info = CalibrationInfo(
            method=method,  # type: ignore[arg-type]
            gains_bgr=(float(gains[0]), float(gains[1]), float(gains[2])),
            confidence=float(min(1.0, max(0.0, confidence))),
            card_detected=card is not None,
        )
        frame.calibration = info
        self._method_counts[method] += 1
        self._confidence_sum += info.confidence
        self._frames_calibrated += 1
        return info

    def summary(self) -> tuple[str, float]:
        """Most used calibration method and mean confidence over the session."""
        if self._frames_calibrated == 0:
            return "none", 0.0
        method = self._method_counts.most_common(1)[0][0]
        return method, self._confidence_sum / self._frames_calibrated

    def _clip_gains(self, gains: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(gains, dtype=np.float64), float(self._config["min_gain"]), float(self._config["max_gain"]))

    def _detect_reference_card(self, image: np.ndarray) -> CardDetection | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        neutral_mask = (
            (hsv[:, :, 1] <= int(self._config["card_max_saturation"])) & (hsv[:, :, 2] >= int(self._config["card_min_value"]))
        ).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        neutral_mask = cv2.morphologyEx(neutral_mask, cv2.MORPH_OPEN, kernel)
        neutral_mask = cv2.morphologyEx(neutral_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(neutral_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(image.shape[0] * image.shape[1])
        min_area = float(self._config["card_min_area_fraction"]) * image_area
        max_area = float(self._config["card_max_area_fraction"]) * image_area
        best: CardDetection | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                continue
            card_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(card_mask, [approximation.reshape(-1, 2)], 255)
            erosion_size = max(3, int(np.sqrt(area) * 0.1) | 1)
            card_mask = cv2.erode(card_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size)))
            pixels = image[card_mask > 0].astype(np.float64)
            if len(pixels) < 200:
                continue
            channel_means = pixels.mean(axis=0)
            channel_std = float(pixels.std(axis=0).mean())
            max_std = float(self._config["card_max_channel_std"])
            if channel_std > max_std:
                continue
            target = float(channel_means.mean())
            gains = target / np.maximum(channel_means, 1e-6)
            uniformity = 1.0 - channel_std / max_std
            confidence = float(self._config["card_confidence"]) * (0.7 + 0.3 * uniformity)
            detection = CardDetection(gains=gains, confidence=confidence, area_fraction=area / image_area)
            if best is None or detection.area_fraction > best.area_fraction:
                best = detection
        return best
