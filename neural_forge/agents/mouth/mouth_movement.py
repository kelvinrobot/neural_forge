"""Mouth movement agent: tongue protrusion deviation, lateral tongue reach and mouth opening symmetry."""

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.neural_forge_face import FaceLandmarks
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import masked_lab_statistics
from neural_forge.services.geometry import asymmetry_ratio, euclidean_distance
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import (
    FACE_CHIN,
    LOWER_LIP_INNER,
    LOWER_LIP_OUTER,
    MOUTH_LEFT_CORNER,
    MOUTH_RIGHT_CORNER,
    UPPER_LIP_INNER,
    facial_midline,
    inter_ocular_distance,
    lips_mask,
    mouth_interior_mask,
    mouth_width,
    polygon_mask,
    reference_skin_mask,
)


@dataclass(frozen=True)
class TongueSegmentation:
    """Largest tongue-coloured region near the mouth."""

    mask: np.ndarray
    pixel_count: int


def lateral_axis(face: FaceLandmarks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midline point, downward midline direction and the unit normal pointing to the patient's left."""
    point, direction = facial_midline(face)
    normal = np.array([-direction[1], direction[0]])
    toward_left = face.point(MOUTH_LEFT_CORNER) - face.point(MOUTH_RIGHT_CORNER)
    if float(np.dot(normal, toward_left)) < 0:
        normal = -normal
    return point, direction, normal


def segment_tongue(
    face: FaceLandmarks,
    lab: np.ndarray,
    extension_mouth_widths: float,
    skin_delta_a: float,
    teeth_min_lightness: float,
    teeth_max_chroma: float,
    min_pixels: int,
) -> TongueSegmentation | None:
    """Segment the tongue as the largest region clearly redder than the patient's own skin, excluding lips and teeth."""
    shape = lab.shape[:2]
    skin_statistics = masked_lab_statistics(lab, reference_skin_mask(face, shape), 100)
    if skin_statistics is None:
        return None
    width = mouth_width(face)
    right_corner = face.point(MOUTH_RIGHT_CORNER)
    left_corner = face.point(MOUTH_LEFT_CORNER)
    top = face.point(UPPER_LIP_INNER)[1]
    bottom = max(face.point(LOWER_LIP_OUTER)[1], face.point(LOWER_LIP_INNER)[1]) + extension_mouth_widths * width
    x_min = min(right_corner[0], left_corner[0]) - 0.35 * width
    x_max = max(right_corner[0], left_corner[0]) + 0.35 * width
    search = polygon_mask(shape, np.array([[x_min, top], [x_max, top], [x_max, bottom], [x_min, bottom]]))
    lightness = lab[:, :, 0]
    red_green = lab[:, :, 1]
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    reddish = red_green >= skin_statistics.red_green + skin_delta_a
    teeth = (lightness >= teeth_min_lightness) & (chroma <= teeth_max_chroma)
    candidate = (reddish & ~teeth).astype(np.uint8) * 255
    lips = cv2.dilate(lips_mask(face, shape), np.ones((3, 3), dtype=np.uint8))
    candidate = cv2.bitwise_and(candidate, search)
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(lips))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < min_pixels:
        return None
    return TongueSegmentation(mask=(labels == largest).astype(np.uint8) * 255, pixel_count=area)


class MouthMovementAgent(AreaAgent):
    """Measures midline tongue protrusion, lateral reach and opening symmetry relative to the facial midline."""

    name = "mouth_movement"
    area = "mouth"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._values: dict[str, list[float]] = defaultdict(list)

    def _segment(self, face: FaceLandmarks, lab: np.ndarray) -> TongueSegmentation | None:
        return segment_tongue(
            face,
            lab,
            self.threshold_float("search_extension_mouth_widths"),
            self.threshold_float("tongue_skin_delta_a"),
            self.threshold_float("teeth_min_lightness"),
            self.threshold_float("teeth_max_chroma"),
            self.threshold_int("tongue_min_pixels"),
        )

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure opening symmetry or tongue position depending on the step."""
        face = frame_context.face
        if face is None:
            return []
        distance = inter_ocular_distance(face)
        if distance < 1.0:
            return []
        step_id = base_step_id(frame_context.step_instance or "")
        midline_point, _, left_normal = lateral_axis(face)
        lateral_steps = {str(key): str(value) for key, value in self.threshold("lateral_steps").items()}
        if step_id in [str(step) for step in self.threshold_list("opening_steps")]:
            interior = mouth_interior_mask(face, frame_context.image_bgr.shape[:2])
            ys, xs = np.nonzero(interior)
            if len(xs) < 50:
                return []
            lateral = (xs - midline_point[0]) * left_normal[0] + (ys - midline_point[1]) * left_normal[1]
            left_area = float((lateral > 0).sum())
            right_area = float((lateral < 0).sum())
            opening_asymmetry = abs(left_area - right_area) / max(left_area + right_area, 1.0)
            jaw_deviation = float(np.dot(face.point(FACE_CHIN) - midline_point, left_normal)) / distance
            opening_height = euclidean_distance(face.point(UPPER_LIP_INNER), face.point(LOWER_LIP_INNER)) / distance
            self._values["opening_asymmetry"].append(opening_asymmetry)
            self._values["jaw_deviation"].append(jaw_deviation)
            self._values["opening_height"].append(opening_height)
            return [
                self.make_reading(frame_context, "mouth_opening_asymmetry", opening_asymmetry, "ratio", "Left-right difference in mouth opening area."),
                self.make_reading(frame_context, "jaw_deviation", jaw_deviation, "IOD", "Chin offset from the facial midline (positive toward the patient's left)."),
            ]
        if step_id in [str(step) for step in self.threshold_list("protrusion_steps")] or step_id in lateral_steps:
            segmentation = self._segment(face, frame_context.lab())
            if segmentation is None:
                return []
            ys, xs = np.nonzero(segmentation.mask)
            lateral = ((xs - midline_point[0]) * left_normal[0] + (ys - midline_point[1]) * left_normal[1]) / distance
            if step_id in lateral_steps:
                direction = lateral_steps[step_id]
                reach = float(np.percentile(lateral, 98)) if direction == "left" else float(-np.percentile(lateral, 2))
                self._values[f"reach_{direction}"].append(reach)
                return [self.make_reading(frame_context, "tongue_lateral_reach", reach, "IOD", f"Tongue reach toward the patient's {direction}.", side=direction)]  # type: ignore[arg-type]
            _, downward, _ = lateral_axis(face)
            along = (xs - midline_point[0]) * downward[0] + (ys - midline_point[1]) * downward[1]
            tip_mask = along >= np.percentile(along, 90)
            tip_deviation = float(np.median(lateral[tip_mask]))
            centroid_deviation = float(np.median(lateral))
            self._values["tongue_tip_deviation"].append(tip_deviation)
            self._values["tongue_centroid_deviation"].append(centroid_deviation)
            return [
                self.make_reading(frame_context, "tongue_tip_deviation", tip_deviation, "IOD", "Tongue tip offset from the facial midline (positive toward the patient's left)."),
            ]
        return []

    def finalize(self) -> AgentResult:
        """Evaluate tongue deviation, lateral reach asymmetry and opening asymmetry."""
        if not self.steps_with_data:
            return self.insufficient_result("Mouth and tongue movements could not be measured.")
        opening_steps = [str(step) for step in self.threshold_list("opening_steps")]
        protrusion_steps = [str(step) for step in self.threshold_list("protrusion_steps")]
        lateral_steps = {str(key): str(value) for key, value in self.threshold("lateral_steps").items()}
        opening_confidence = self.data_confidence(opening_steps)
        protrusion_confidence = self.data_confidence(protrusion_steps)
        lateral_confidence = self.data_confidence(list(lateral_steps.keys()))
        medians = {key: robust_median(values) for key, values in self._values.items()}
        tip = medians.get("tongue_tip_deviation", float("nan"))
        opening_asymmetry = medians.get("opening_asymmetry", float("nan"))
        jaw = medians.get("jaw_deviation", float("nan"))
        reach_left = medians.get("reach_left", float("nan"))
        reach_right = medians.get("reach_right", float("nan"))
        metrics = [
            self.summary_reading("tongue_tip_deviation", tip, "IOD", "Median tongue tip offset from midline (positive toward the left).", protrusion_confidence, ",".join(protrusion_steps)),
            self.summary_reading("tongue_centroid_deviation", medians.get("tongue_centroid_deviation", float("nan")), "IOD", "Median tongue body offset from midline.", protrusion_confidence, ",".join(protrusion_steps)),
            self.summary_reading("mouth_opening_asymmetry", opening_asymmetry, "ratio", "Median left-right opening area difference.", opening_confidence, ",".join(opening_steps)),
            self.summary_reading("jaw_deviation", jaw, "IOD", "Median chin offset during wide opening.", opening_confidence, ",".join(opening_steps)),
            self.summary_reading("mouth_opening_height", medians.get("opening_height", float("nan")), "IOD", "Median mouth opening height.", opening_confidence, ",".join(opening_steps)),
            self.summary_reading("tongue_lateral_reach", reach_left, "IOD", "Tongue reach toward the left.", lateral_confidence, ",".join(lateral_steps.keys()), side="left"),
            self.summary_reading("tongue_lateral_reach", reach_right, "IOD", "Tongue reach toward the right.", lateral_confidence, ",".join(lateral_steps.keys()), side="right"),
        ]
        flags: list[ScreeningFlag] = []
        tip_threshold = self.threshold_float("tongue_deviation_flag_iod")
        if np.isfinite(tip) and abs(tip) >= tip_threshold:
            side = "left" if tip > 0 else "right"
            flags.append(self.make_flag("tongue_deviation", "Tongue deviates to one side", FlagSeverity.MODERATE, self.exceedance_confidence(tip, tip_threshold, protrusion_confidence), f"The tongue pointed toward the {side} when stuck out straight, which can indicate weakness of the {side} tongue muscles (hypoglossal nerve).", protrusion_steps, evidence=[f"tip deviation {tip:+.3f} IOD"], side=side))  # type: ignore[arg-type]
        opening_threshold = self.threshold_float("opening_asymmetry_flag")
        jaw_threshold = self.threshold_float("jaw_deviation_flag_iod")
        if (np.isfinite(opening_asymmetry) and opening_asymmetry >= opening_threshold) or (np.isfinite(jaw) and abs(jaw) >= jaw_threshold):
            side = "left" if np.isfinite(jaw) and jaw > 0 else "right"
            flags.append(self.make_flag("mouth_opening_asymmetry", "Uneven mouth opening", FlagSeverity.LOW, max(self.exceedance_confidence(opening_asymmetry, opening_threshold, opening_confidence) if np.isfinite(opening_asymmetry) and opening_asymmetry >= opening_threshold else 0.0, self.exceedance_confidence(jaw, jaw_threshold, opening_confidence) if np.isfinite(jaw) and abs(jaw) >= jaw_threshold else 0.0), f"The mouth opened unevenly and the jaw shifted toward the {side}, which can relate to jaw joint or facial nerve problems.", opening_steps, evidence=[f"opening asymmetry {opening_asymmetry:.2f}", f"jaw deviation {jaw:+.3f} IOD"], side=side))  # type: ignore[arg-type]
        if np.isfinite(reach_left) and np.isfinite(reach_right):
            reach_asymmetry = asymmetry_ratio(max(reach_left, 0.0), max(reach_right, 0.0))
            metrics.append(self.summary_reading("tongue_reach_asymmetry", reach_asymmetry, "ratio", "Relative difference between left and right tongue reach.", lateral_confidence, ",".join(lateral_steps.keys())))
            reach_threshold = self.threshold_float("lateral_reach_asymmetry_flag")
            if reach_asymmetry >= reach_threshold:
                weaker_direction = "left" if reach_left < reach_right else "right"
                flags.append(self.make_flag("tongue_reach_asymmetry", "Uneven side-to-side tongue movement", FlagSeverity.LOW, self.exceedance_confidence(reach_asymmetry, reach_threshold, lateral_confidence), f"The tongue reached less far toward the {weaker_direction}.", list(lateral_steps.keys()), evidence=[f"reach asymmetry {reach_asymmetry:.2f}"]))
        confidence_values = [value for value in (opening_confidence, protrusion_confidence, lateral_confidence) if value > 0]
        return self.build_result(
            description=f"Tongue tip deviation {tip:+.3f} IOD, opening asymmetry {opening_asymmetry:.2f}.",
            metrics=metrics,
            flags=flags,
            confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
            value=tip,
            unit="IOD",
            limitations=["Tongue segmentation uses colour and may be affected by lighting, lipstick or food staining."],
        )

    def _reset_measurements(self) -> None:
        self._values.clear()
