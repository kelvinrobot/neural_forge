"""Pigmentation agent: depigmented and hyperpigmented patch detection relative to surrounding skin."""

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, BestCropBuffer, CropCandidate, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.schemas.readings import AgentResult, EvidenceCrop, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import crop_image, expand_box, skin_mask_ycrcb
from neural_forge.services.regions import LEFT_BROW, LIPS_OUTER, RIGHT_BROW, eye_polygon, polygon_mask


@dataclass
class PatchMeasurement:
    """Pigmentation measurements of one analysed frame."""

    depigmented_fraction: float
    hyperpigmented_fraction: float
    depigmented_patches: int
    hyperpigmented_patches: int
    depigmented_contrast: float


class PigmentationAgent(AreaAgent):
    """Finds skin patches markedly lighter or darker than the patient's own surrounding skin in CIELAB."""

    name = "pigmentation"
    area = "skin"
    required_trackers = frozenset()

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._measurements: dict[str, list[PatchMeasurement]] = defaultdict(list)
        self._frame_counter: dict[str, int] = defaultdict(int)
        self._crops = BestCropBuffer(self.threshold_int("crops_to_keep"))

    def _skin_envelope(self, frame_context: FrameContext) -> np.ndarray:
        image = frame_context.colour_image()
        skin = skin_mask_ycrcb(image, self.threshold_range("skin_cr_range"), self.threshold_range("skin_cb_range"))
        kernel_size = max(5, int(min(image.shape[:2]) * 0.03) | 1)
        closed = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        envelope = np.zeros_like(closed)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(envelope, [largest], -1, 255, thickness=-1)
        face = frame_context.face
        if face is not None:
            exclusion = np.zeros_like(envelope)
            for polygon in (eye_polygon(face, "right"), eye_polygon(face, "left"), face.points(RIGHT_BROW), face.points(LEFT_BROW), face.points(LIPS_OUTER)):
                exclusion = cv2.bitwise_or(exclusion, polygon_mask(envelope.shape, polygon))
            exclusion = cv2.dilate(exclusion, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
            envelope = cv2.bitwise_and(envelope, cv2.bitwise_not(exclusion))
        return envelope

    def _patches(self, mask: np.ndarray, lightness: np.ndarray, envelope_area: int) -> tuple[np.ndarray, int, float, tuple[int, int, int, int] | None]:
        minimum_area = self.threshold_float("min_patch_area_fraction") * envelope_area
        border_minimum = self.threshold_float("border_contrast_min")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        kept = np.zeros_like(mask)
        contrasts: list[float] = []
        largest_box: tuple[int, int, int, int] | None = None
        largest_area = 0
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            component = (labels == label).astype(np.uint8) * 255
            ring = cv2.bitwise_and(cv2.dilate(component, ring_kernel), cv2.bitwise_not(component))
            if not ring.any():
                continue
            contrast = abs(float(lightness[component > 0].mean()) - float(lightness[ring > 0].mean()))
            if contrast < border_minimum:
                continue
            kept = cv2.bitwise_or(kept, component)
            contrasts.append(contrast)
            if area > largest_area:
                largest_area = area
                x, y, width, height = (int(value) for value in stats[label, :4])
                largest_box = (x, y, x + width, y + height)
        return kept, len(contrasts), float(np.mean(contrasts)) if contrasts else 0.0, largest_box

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Analyse lightness patches inside the skin envelope every few frames."""
        step_instance = frame_context.step_instance or ""
        self._frame_counter[step_instance] += 1
        if (self._frame_counter[step_instance] - 1) % max(1, self.threshold_int("analysis_interval_frames")) != 0:
            return []
        envelope = self._skin_envelope(frame_context)
        envelope_area = int((envelope > 0).sum())
        if envelope_area < self.threshold_int("min_skin_pixels"):
            return []
        lab = frame_context.lab()
        kernel = max(3, self.threshold_int("smoothing_kernel") | 1)
        lightness = cv2.medianBlur(np.ascontiguousarray(lab[:, :, 0]).astype(np.float32), min(kernel, 5))
        chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
        inside = envelope > 0
        reference = float(np.median(lightness[inside]))
        specular = (lightness >= self.threshold_float("specular_min_lightness")) & (chroma <= self.threshold_float("specular_max_chroma"))
        depigmented_mask = ((lightness >= reference + self.threshold_float("depigmented_delta_l")) & inside & ~specular).astype(np.uint8) * 255
        hyperpigmented_mask = ((lightness <= reference - self.threshold_float("hyperpigmented_delta_l")) & inside).astype(np.uint8) * 255
        opening = np.ones((3, 3), dtype=np.uint8)
        depigmented_mask = cv2.morphologyEx(depigmented_mask, cv2.MORPH_OPEN, opening)
        hyperpigmented_mask = cv2.morphologyEx(hyperpigmented_mask, cv2.MORPH_OPEN, opening)
        depigmented, depigmented_count, depigmented_contrast, depigmented_box = self._patches(depigmented_mask, lightness, envelope_area)
        hyperpigmented, hyperpigmented_count, _, hyperpigmented_box = self._patches(hyperpigmented_mask, lightness, envelope_area)
        measurement = PatchMeasurement(
            depigmented_fraction=float((depigmented > 0).sum()) / envelope_area,
            hyperpigmented_fraction=float((hyperpigmented > 0).sum()) / envelope_area,
            depigmented_patches=depigmented_count,
            hyperpigmented_patches=hyperpigmented_count,
            depigmented_contrast=depigmented_contrast,
        )
        self._measurements[step_instance].append(measurement)
        box = depigmented_box or hyperpigmented_box
        if box is not None:
            crop = crop_image(frame_context.colour_image(), expand_box(box, self.threshold_float("crop_padding_fraction")))
            self._crops.offer(CropCandidate(crop, measurement.depigmented_fraction + measurement.hyperpigmented_fraction, step_instance, frame_context.timestamp_s, "pigmentation patch"))
        colour_confidence = frame_context.calibration_confidence
        return [
            self.make_reading(frame_context, "depigmented_fraction", measurement.depigmented_fraction, "fraction", "Fraction of visible skin markedly lighter than the surrounding skin.", confidence=colour_confidence),
            self.make_reading(frame_context, "hyperpigmented_fraction", measurement.hyperpigmented_fraction, "fraction", "Fraction of visible skin markedly darker than the surrounding skin.", confidence=colour_confidence),
        ]

    def finalize(self) -> AgentResult:
        """Evaluate consistent depigmented or hyperpigmented patches."""
        all_measurements = [measurement for measurements in self._measurements.values() for measurement in measurements]
        if not all_measurements:
            return self.insufficient_result("Skin pigmentation could not be analysed.")
        confidence = self.data_confidence(self.steps_with_data)
        metrics: list[Reading] = []
        depigmented_threshold = self.threshold_float("depigmented_fraction_flag")
        hyperpigmented_threshold = self.threshold_float("hyperpigmented_fraction_flag")
        consistency = self.threshold_float("consistency_fraction")
        depigmented_steps: list[str] = []
        hyperpigmented_steps: list[str] = []
        for step_instance, measurements in self._measurements.items():
            if not measurements:
                continue
            depigmented = robust_median([measurement.depigmented_fraction for measurement in measurements])
            hyperpigmented = robust_median([measurement.hyperpigmented_fraction for measurement in measurements])
            step_confidence = self.data_confidence([step_instance])
            metrics.append(self.summary_reading("depigmented_fraction", depigmented, "fraction", f"Median depigmented skin fraction in {step_instance}.", step_confidence, step_instance))
            metrics.append(self.summary_reading("hyperpigmented_fraction", hyperpigmented, "fraction", f"Median hyperpigmented skin fraction in {step_instance}.", step_confidence, step_instance))
            depigmented_consistency = float(np.mean([measurement.depigmented_fraction >= depigmented_threshold and measurement.depigmented_patches > 0 for measurement in measurements]))
            hyperpigmented_consistency = float(np.mean([measurement.hyperpigmented_fraction >= hyperpigmented_threshold and measurement.hyperpigmented_patches > 0 for measurement in measurements]))
            if depigmented_consistency >= consistency:
                depigmented_steps.append(step_instance)
            if hyperpigmented_consistency >= consistency:
                hyperpigmented_steps.append(step_instance)
        overall_depigmented = robust_median([measurement.depigmented_fraction for measurement in all_measurements])
        contrast = robust_median([measurement.depigmented_contrast for measurement in all_measurements if measurement.depigmented_patches > 0])
        metrics.append(self.summary_reading("depigmented_border_contrast", contrast, "CIELAB L*", "Lightness step at the edge of depigmented patches.", confidence, ",".join(self.steps_with_data)))
        flags: list[ScreeningFlag] = []
        if depigmented_steps:
            flags.append(
                self.make_flag(
                    "vitiligo_pattern_depigmentation",
                    "Well-defined pale skin patches",
                    FlagSeverity.MODERATE,
                    confidence * min(1.0, 0.5 + len(depigmented_steps) * 0.15),
                    "Patches of skin much lighter than the surrounding skin with distinct borders were seen, a pattern that can occur with vitiligo.",
                    depigmented_steps,
                    evidence=[f"steps: {', '.join(depigmented_steps)}", f"border contrast {contrast:.1f} L*"],
                )
            )
        if hyperpigmented_steps:
            flags.append(
                self.make_flag(
                    "hyperpigmented_patches",
                    "Darker skin patches",
                    FlagSeverity.LOW,
                    confidence * min(1.0, 0.4 + len(hyperpigmented_steps) * 0.15),
                    "Patches of skin darker than the surrounding skin were seen consistently.",
                    hyperpigmented_steps,
                    evidence=[f"steps: {', '.join(hyperpigmented_steps)}"],
                )
            )
        evidence: list[EvidenceCrop] = [self.make_evidence_crop(candidate.label, candidate) for candidate in self._crops.items()] if flags else []
        return self.build_result(
            description=f"Median depigmented skin fraction {overall_depigmented:.3f}.",
            metrics=metrics,
            flags=flags,
            confidence=confidence,
            value=overall_depigmented,
            unit="fraction",
            evidence=evidence,
            limitations=["Shadows, hair and uneven lighting can mimic pigmentation changes."],
        )

    def _reset_measurements(self) -> None:
        self._measurements.clear()
        self._frame_counter.clear()
        self._crops.clear()
