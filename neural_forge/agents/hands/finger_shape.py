"""Finger shape agent: fingertip profile ratio, Schamroth window test and zero-shot clubbing."""

from collections import defaultdict

import cv2
import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, BestCropBuffer, CropCandidate, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.neural_forge_hands import HandLandmarks
from neural_forge.schemas.readings import AgentResult, EvidenceCrop, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import clip_box, laplacian_sharpness, skin_mask_ycrcb
from neural_forge.services.geometry import euclidean_distance, unit_vector
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import FINGER_CHAINS, finger_box, width_profile


class FingerShapeAgent(AreaAgent):
    """Estimates distal-to-interphalangeal depth ratio from side views and checks the nail-to-nail window."""

    name = "finger_shape"
    area = "hands"
    required_trackers = frozenset({TrackerName.HANDS})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._depth_ratios: dict[str, list[float]] = defaultdict(list)
        self._window_present: list[bool] = []
        self._window_ratios: list[float] = []
        self._crops = BestCropBuffer(self.threshold_int("crops_to_keep"))

    def _skin_roi(self, frame_context: FrameContext, box: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[int, int]]:
        x_min, y_min, x_max, y_max = clip_box(box, frame_context.width, frame_context.height)
        roi = frame_context.colour_image()[y_min:y_max, x_min:x_max]
        mask = skin_mask_ycrcb(roi, self.threshold_range("skin_cr_range"), self.threshold_range("skin_cb_range"))
        return mask, (x_min, y_min)

    def _depth_ratio(self, frame_context: FrameContext, hand: HandLandmarks) -> float:
        finger = str(self.threshold("finger"))
        chain = FINGER_CHAINS[finger]
        box = finger_box(hand, finger, self.threshold_float("crop_padding_fraction"))
        mask, offset = self._skin_roi(frame_context, box)
        if mask.size == 0:
            return float("nan")
        offset_vector = np.array(offset, dtype=np.float64)
        distal_joint = hand.point(chain[2]) - offset_vector
        tip = hand.point(chain[3]) - offset_vector
        phalanx_length = euclidean_distance(distal_joint, tip)
        if phalanx_length < 4.0:
            return float("nan")
        axis = unit_vector(tip - distal_joint)
        interphalangeal_depth = width_profile(mask, distal_joint, axis, 1.5 * phalanx_length)
        if interphalangeal_depth <= 1.0:
            return float("nan")
        start_fraction, end_fraction = self.threshold_range("nail_region_fraction")
        samples = self.threshold_int("profile_samples")
        distal_depths = [
            width_profile(mask, distal_joint + axis * phalanx_length * fraction, axis, 1.5 * phalanx_length)
            for fraction in np.linspace(start_fraction, end_fraction, samples)
        ]
        distal_depth = max(distal_depths) if distal_depths else 0.0
        if distal_depth <= 1.0:
            return float("nan")
        return distal_depth / interphalangeal_depth

    def _window_ratio(self, frame_context: FrameContext) -> float:
        hands = frame_context.hands
        if hands is None or hands.count == 0:
            return float("nan")
        finger = str(self.threshold("finger"))
        chain = FINGER_CHAINS[finger]
        points = []
        phalanx_lengths = []
        for hand in hands.hands:
            points.extend([hand.point(chain[1]), hand.point(chain[2]), hand.point(chain[3])])
            phalanx_lengths.append(euclidean_distance(hand.point(chain[2]), hand.point(chain[3])))
        stacked = np.array(points)
        x_min, y_min = stacked.min(axis=0)
        x_max, y_max = stacked.max(axis=0)
        size = max(x_max - x_min, y_max - y_min, 1.0)
        padding = size * 0.25
        mask, _ = self._skin_roi(frame_context, (x_min - padding, y_min - padding, x_max + padding, y_max + padding))
        if mask.size == 0:
            return float("nan")
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0.0
        hole_areas = [
            cv2.contourArea(contour)
            for contour, relation in zip(contours, hierarchy[0])
            if relation[3] >= 0
        ]
        phalanx = float(np.mean(phalanx_lengths)) if phalanx_lengths else 0.0
        if phalanx < 4.0:
            return float("nan")
        return (max(hole_areas) if hole_areas else 0.0) / (phalanx * phalanx)

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure the fingertip depth ratio in side views or the nail window in the nail-to-nail step."""
        hands = frame_context.hands
        if hands is None or hands.count == 0:
            return []
        step_id = base_step_id(frame_context.step_instance or "")
        side_steps = {str(key): str(value) for key, value in self.threshold("side_view_steps").items()}
        window_steps = [str(step) for step in self.threshold_list("window_steps")]
        if step_id in side_steps:
            side = side_steps[step_id]
            hand = hands.get(side) or hands.largest()  # type: ignore[arg-type]
            if hand is None:
                return []
            ratio = self._depth_ratio(frame_context, hand)
            if not np.isfinite(ratio):
                return []
            self._depth_ratios[side].append(ratio)
            box = finger_box(hand, str(self.threshold("finger")), self.threshold_float("crop_padding_fraction"))
            x_min, y_min, x_max, y_max = clip_box(box, frame_context.width, frame_context.height)
            crop = frame_context.colour_image()[y_min:y_max, x_min:x_max].copy()
            self._crops.offer(CropCandidate(crop, laplacian_sharpness(crop), frame_context.step_instance or step_id, frame_context.timestamp_s, f"{side} index fingertip side view"))
            return [self.make_reading(frame_context, "distal_interphalangeal_depth_ratio", ratio, "ratio", f"{side.capitalize()} index fingertip depth relative to the last joint.", side=side)]  # type: ignore[arg-type]
        if step_id in window_steps:
            window_ratio = self._window_ratio(frame_context)
            if not np.isfinite(window_ratio):
                return []
            present = window_ratio >= self.threshold_float("window_min_area_ratio")
            self._window_present.append(bool(present))
            self._window_ratios.append(window_ratio)
            return [self.make_reading(frame_context, "schamroth_window_area", window_ratio, "phalanx^2", "Area of the gap between the two nails (diamond window).")]
        return []

    def _zero_shot(self, limitations: list[str]) -> tuple[float, list[EvidenceCrop]]:
        candidates = self._crops.items()
        results = self.safe_classify([candidate.image for candidate in candidates], str(self.threshold("zero_shot_task")), limitations)
        if not results:
            return float("nan"), []
        clubbed_label = str(self.threshold("clubbed_label"))
        probability = self.mean_label_scores(results).get(clubbed_label, float("nan"))
        question = str(self.threshold("consultation_question"))
        evidence = []
        for candidate, result in zip(candidates, results):
            doubtful = self.needs_consultation(result)
            evidence.append(self.make_evidence_crop(candidate.label, candidate, result, doubtful, question if doubtful else None))
        return probability, evidence

    def finalize(self) -> AgentResult:
        """Combine profile ratio, window test and zero-shot score into a clubbing screen."""
        if not self.steps_with_data:
            return self.insufficient_result("Fingertip shape could not be measured.")
        side_steps = {str(key): str(value) for key, value in self.threshold("side_view_steps").items()}
        window_steps = [str(step) for step in self.threshold_list("window_steps")]
        side_confidence = self.data_confidence(list(side_steps.keys()))
        window_confidence = self.data_confidence(window_steps)
        metrics: list[Reading] = []
        limitations: list[str] = []
        ratio_threshold = self.threshold_float("dpd_ipd_ratio_flag")
        signs: list[str] = []
        sign_confidences: list[float] = []
        for side in ("right", "left"):
            ratio = robust_median(self._depth_ratios.get(side, []))
            metrics.append(self.summary_reading("distal_interphalangeal_depth_ratio", ratio, "ratio", f"{side.capitalize()} index finger distal/interphalangeal depth ratio.", side_confidence, ",".join(side_steps.keys()), side=side))  # type: ignore[arg-type]
            if np.isfinite(ratio) and ratio > ratio_threshold:
                signs.append(f"{side} depth ratio {ratio:.2f}")
                sign_confidences.append(self.exceedance_confidence(ratio, ratio_threshold, side_confidence))
        absent_fraction = float(np.mean([not present for present in self._window_present])) if self._window_present else float("nan")
        metrics.append(self.summary_reading("schamroth_window_area", robust_median(self._window_ratios), "phalanx^2", "Median nail-to-nail window area.", window_confidence, ",".join(window_steps)))
        metrics.append(self.summary_reading("schamroth_window_absent_fraction", absent_fraction, "fraction", "Fraction of frames without a visible nail-to-nail window.", window_confidence, ",".join(window_steps)))
        if np.isfinite(absent_fraction) and absent_fraction >= self.threshold_float("window_absent_fraction"):
            signs.append(f"window absent in {absent_fraction:.0%} of frames")
            sign_confidences.append(window_confidence * absent_fraction)
        probability, evidence = self._zero_shot(limitations)
        metrics.append(self.summary_reading("zero_shot_clubbing_probability", probability, "probability", "Zero-shot similarity of fingertip crops to clubbed fingers.", side_confidence, ",".join(side_steps.keys())))
        if np.isfinite(probability) and probability >= self.threshold_float("zero_shot_flag_probability"):
            signs.append(f"zero-shot clubbing probability {probability:.2f}")
            sign_confidences.append(probability * side_confidence)
        flags: list[ScreeningFlag] = []
        all_steps = list(side_steps.keys()) + window_steps
        if len(signs) >= 2:
            flags.append(
                self.make_flag(
                    "finger_clubbing",
                    "Signs of finger clubbing",
                    FlagSeverity.MODERATE,
                    float(np.mean(sign_confidences)),
                    "Several fingertip shape measures suggested clubbing, which can be associated with chronic heart or lung conditions.",
                    all_steps,
                    evidence=signs,
                )
            )
        elif len(signs) == 1:
            flags.append(
                self.make_flag(
                    "possible_finger_clubbing",
                    "One sign of possible finger clubbing",
                    FlagSeverity.LOW,
                    sign_confidences[0] * 0.7,
                    "One fingertip shape measure suggested possible clubbing; other measures did not confirm it.",
                    all_steps,
                    evidence=signs,
                )
            )
        confidence_values = [value for value in (side_confidence, window_confidence) if value > 0]
        return self.build_result(
            description=f"Fingertip shape: {len(signs)} clubbing sign(s) detected." if signs else "Fingertip shape measures were within screening reference limits.",
            metrics=metrics,
            flags=flags,
            confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
            value=float(len(signs)),
            unit="signs",
            evidence=evidence,
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._depth_ratios.clear()
        self._window_present.clear()
        self._window_ratios.clear()
        self._crops.clear()
