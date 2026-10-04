"""Lesion finder agent: YOLOE text-prompted detection with a MedSigLIP sliding-grid fallback."""

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, BestCropBuffer, CropCandidate
from neural_forge.core.exceptions import ConfigurationError, ModelInferenceError, ModelNotAvailableError
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.logging import get_logger, log_event
from neural_forge.models.yoloe_detector import Detection
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import crop_image, expand_box, laplacian_sharpness, skin_mask_ycrcb

logger = get_logger(__name__)


@dataclass
class CandidateRegion:
    """A tracked skin region that may contain a lesion."""

    region_id: str
    step_instance: str
    bbox: tuple[float, float, float, float]
    crop: np.ndarray
    sharpness: float
    timestamp: float
    hits: int
    max_confidence: float
    label: str
    source: str


def intersection_over_union(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """Intersection over union of two boxes."""
    x_min = max(first[0], second[0])
    y_min = max(first[1], second[1])
    x_max = min(first[2], second[2])
    y_max = min(first[3], second[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


class LesionFinderAgent(AreaAgent):
    """Finds spots, rashes and patches on skin and tracks them across frames."""

    name = "lesion_finder"
    area = "skin"
    required_trackers = frozenset()

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._regions: dict[str, list[CandidateRegion]] = defaultdict(list)
        self._best_frames: dict[str, BestCropBuffer] = {}
        self._frames_seen: dict[str, int] = defaultdict(int)
        self._detector_unavailable = False
        self._detector_error: str | None = None
        self._region_counter = 0
        self._fallback_done = False
        self._fallback_limitations: list[str] = []
        self.latest_detection_count = 0

    def _skin_mask(self, image: np.ndarray) -> np.ndarray:
        return skin_mask_ycrcb(image, self.threshold_range("skin_cr_range"), self.threshold_range("skin_cb_range"))

    def _accept_detection(self, detection: Detection, skin: np.ndarray, frame_area: float) -> bool:
        area_fraction = detection.area / frame_area
        if area_fraction < self.threshold_float("min_box_area_fraction") or area_fraction > self.threshold_float("max_box_area_fraction"):
            return False
        x_min, y_min, x_max, y_max = detection.bbox
        box_skin = skin[y_min:y_max, x_min:x_max]
        if box_skin.size == 0:
            return False
        return float((box_skin > 0).mean()) >= self.threshold_float("min_skin_fraction_in_box")

    def _track(self, step_instance: str, detection: Detection, image: np.ndarray, timestamp: float) -> None:
        box = tuple(float(value) for value in detection.bbox)
        crop = crop_image(image, expand_box(box, self.threshold_float("crop_padding_fraction")))  # type: ignore[arg-type]
        sharpness = laplacian_sharpness(crop)
        for region in self._regions[step_instance]:
            if intersection_over_union(region.bbox, box) >= self.threshold_float("iou_match_threshold"):  # type: ignore[arg-type]
                region.hits += 1
                region.max_confidence = max(region.max_confidence, detection.confidence)
                region.bbox = box  # type: ignore[assignment]
                if sharpness > region.sharpness:
                    region.crop = crop
                    region.sharpness = sharpness
                    region.timestamp = timestamp
                return
        self._region_counter += 1
        self._regions[step_instance].append(
            CandidateRegion(
                region_id=f"region-{self._region_counter}",
                step_instance=step_instance,
                bbox=box,  # type: ignore[arg-type]
                crop=crop,
                sharpness=sharpness,
                timestamp=timestamp,
                hits=1,
                max_confidence=detection.confidence,
                label=detection.label,
                source="detector",
            )
        )

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure skin coverage, keep sharp frames and run detection at the configured interval."""
        image = frame_context.colour_image()
        skin = self._skin_mask(image)
        skin_fraction = float((skin > 0).mean())
        if skin_fraction < self.threshold_float("min_skin_fraction_frame"):
            return []
        step_instance = frame_context.step_instance or ""
        if step_instance not in self._best_frames:
            self._best_frames[step_instance] = BestCropBuffer(self.threshold_int("best_frames_per_step"))
        self._best_frames[step_instance].offer(
            CropCandidate(image=image.copy(), sharpness=frame_context.quality.blur_variance if frame_context.quality else 0.0, protocol_step=step_instance, timestamp=frame_context.timestamp_s, label="skin frame")
        )
        self._frames_seen[step_instance] += 1
        readings = [self.make_reading(frame_context, "skin_coverage", skin_fraction, "fraction", "Fraction of the frame showing skin.")]
        if self._detector_unavailable or self._frames_seen[step_instance] % max(1, self.threshold_int("detection_interval_frames")) != 1:
            return readings
        try:
            detections = self._registry.lesion_detector().detect(image)
        except (ModelNotAvailableError, ModelInferenceError) as error:
            self._detector_unavailable = True
            self._detector_error = error.message
            log_event(logger, logging.WARNING, "lesion_detector_unavailable", error=error.message)
            return readings
        frame_area = float(image.shape[0] * image.shape[1])
        accepted = [detection for detection in detections if self._accept_detection(detection, skin, frame_area)]
        for detection in accepted:
            self._track(step_instance, detection, image, frame_context.timestamp_s)
        frame_context.annotations["lesion_detections"] = accepted
        self.latest_detection_count = len(accepted)
        readings.append(self.make_reading(frame_context, "candidate_regions_detected", float(len(accepted)), "count", "Skin spots, patches or rashes detected in this frame."))
        return readings

    def _grid_fallback(self, step_instance: str, limitations: list[str]) -> list[CandidateRegion]:
        buffer = self._best_frames.get(step_instance)
        if buffer is None or len(buffer) == 0:
            return []
        frame = buffer.items()[0]
        image = frame.image
        skin = self._skin_mask(image)
        cells = max(2, self.threshold_int("grid_cells"))
        height, width = image.shape[:2]
        crops: list[np.ndarray] = []
        boxes: list[tuple[float, float, float, float]] = []
        for row in range(cells):
            for column in range(cells):
                box = (column * width / cells, row * height / cells, (column + 1) * width / cells, (row + 1) * height / cells)
                cell_skin = skin[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                if cell_skin.size == 0 or float((cell_skin > 0).mean()) < self.threshold_float("grid_min_skin_fraction"):
                    continue
                crops.append(crop_image(image, box))
                boxes.append(box)
        if not crops:
            return []
        try:
            results = self._registry.classifier().classify(crops, str(self.threshold("grid_task")))
        except (ModelNotAvailableError, ModelInferenceError, ConfigurationError) as error:
            limitations.append(f"Grid-based skin screening was unavailable: {error.message}")
            return []
        abnormal_label = str(self.threshold("grid_abnormal_label"))
        threshold = self.threshold_float("grid_abnormal_probability")
        regions: list[CandidateRegion] = []
        for crop, box, result in zip(crops, boxes, results):
            probability = result.label_scores.get(abnormal_label, 0.0)
            if probability < threshold:
                continue
            self._region_counter += 1
            regions.append(
                CandidateRegion(
                    region_id=f"region-{self._region_counter}",
                    step_instance=step_instance,
                    bbox=box,
                    crop=crop,
                    sharpness=laplacian_sharpness(crop),
                    timestamp=frame.timestamp,
                    hits=self.threshold_int("min_region_hits"),
                    max_confidence=probability,
                    label="abnormal skin patch",
                    source="grid",
                )
            )
        return regions

    def prepare_candidates(self) -> list[CandidateRegion]:
        """Return confirmed candidate regions, running the grid fallback once where detection found nothing."""
        if not self._fallback_done:
            self._fallback_done = True
            for step_instance in list(self._best_frames.keys()):
                confirmed = [region for region in self._regions.get(step_instance, []) if region.hits >= self.threshold_int("min_region_hits")]
                if confirmed and not self._detector_unavailable:
                    continue
                self._regions[step_instance].extend(self._grid_fallback(step_instance, self._fallback_limitations))
        maximum = self.threshold_int("max_regions_per_step")
        minimum_hits = self.threshold_int("min_region_hits")
        selected: list[CandidateRegion] = []
        for regions in self._regions.values():
            confirmed = sorted((region for region in regions if region.hits >= minimum_hits), key=lambda region: region.max_confidence, reverse=True)
            selected.extend(confirmed[:maximum])
        return selected

    def finalize(self) -> AgentResult:
        """Summarise detected candidate regions per step."""
        if not self.steps_with_data:
            return self.insufficient_result("Skin areas could not be examined.")
        candidates = self.prepare_candidates()
        limitations = list(self._fallback_limitations)
        if self._detector_unavailable:
            limitations.append(f"The text-prompted lesion detector was unavailable ({self._detector_error}); grid screening was used instead.")
        confidence = self.data_confidence(self.steps_with_data)
        metrics: list[Reading] = [
            self.summary_reading("candidate_regions_total", float(len(candidates)), "count", "Skin regions selected for closer analysis.", confidence, ",".join(self.steps_with_data)),
        ]
        by_step: dict[str, int] = defaultdict(int)
        for region in candidates:
            by_step[region.step_instance] += 1
        for step_instance in self.steps_with_data:
            metrics.append(self.summary_reading("candidate_regions", float(by_step.get(step_instance, 0)), "count", f"Candidate skin regions in {step_instance}.", self.data_confidence([step_instance]), step_instance))
        flags: list[ScreeningFlag] = []
        if candidates:
            flags.append(
                self.make_flag(
                    "skin_findings_present",
                    "Visible skin spots or patches",
                    FlagSeverity.LOW,
                    confidence * float(np.mean([region.max_confidence for region in candidates])),
                    f"{len(candidates)} skin region(s) with visible spots, patches or rash were found and analysed further.",
                    sorted({region.step_instance for region in candidates}),
                    evidence=[f"{region.region_id}: {region.label} ({region.source}, {region.max_confidence:.2f})" for region in candidates[:10]],
                )
            )
        return self.build_result(
            description=f"{len(candidates)} candidate skin region(s) found." if candidates else "No distinct skin lesions were found.",
            metrics=metrics,
            flags=flags,
            confidence=confidence,
            value=float(len(candidates)),
            unit="regions",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._regions.clear()
        self._best_frames.clear()
        self._frames_seen.clear()
        self._detector_unavailable = False
        self._detector_error = None
        self._region_counter = 0
        self._fallback_done = False
        self._fallback_limitations.clear()
        self.latest_detection_count = 0
