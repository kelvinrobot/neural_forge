"""Joint appearance agent: finger alignment angles, joint width ratios and zero-shot knuckle swelling."""

from collections import defaultdict

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, BestCropBuffer, CropCandidate, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.neural_forge_hands import HandLandmarks
from neural_forge.schemas.readings import AgentResult, EvidenceCrop, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import clip_box, laplacian_sharpness, skin_mask_ycrcb
from neural_forge.services.geometry import euclidean_distance, unit_vector, vector_angle_degrees
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import FINGER_CHAINS, hand_box, width_profile


def signed_ulnar_deviation(hand: HandLandmarks, finger: str) -> float:
    """Angle in degrees between metacarpal and proximal phalanx, positive toward the little finger."""
    chain = FINGER_CHAINS[finger]
    wrist = hand.point(0)
    knuckle = hand.point(chain[0])
    proximal_joint = hand.point(chain[1])
    metacarpal = knuckle - wrist
    phalanx = proximal_joint - knuckle
    angle = vector_angle_degrees(metacarpal, phalanx)
    if not np.isfinite(angle):
        return float("nan")
    ulnar_direction = unit_vector(hand.point(17) - hand.point(5))
    sign = 1.0 if float(np.dot(unit_vector(phalanx) - unit_vector(metacarpal), ulnar_direction)) >= 0 else -1.0
    return sign * angle


class JointAppearanceAgent(AreaAgent):
    """Screens for ulnar drift of the fingers and swelling of the finger joints."""

    name = "joint_appearance"
    area = "hands"
    required_trackers = frozenset({TrackerName.HANDS})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._deviation: dict[str, list[float]] = defaultdict(list)
        self._width_ratio: dict[str, list[float]] = defaultdict(list)
        self._crops = BestCropBuffer(self.threshold_int("crops_to_keep"))

    def _joint_width_ratios(self, frame_context: FrameContext, hand: HandLandmarks) -> list[float]:
        box = hand_box(hand, self.threshold_float("crop_padding_fraction"))
        x_min, y_min, x_max, y_max = clip_box(box, frame_context.width, frame_context.height)
        roi = frame_context.colour_image()[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return []
        mask = skin_mask_ycrcb(roi, self.threshold_range("skin_cr_range"), self.threshold_range("skin_cb_range"))
        offset = np.array([x_min, y_min], dtype=np.float64)
        ratios: list[float] = []
        for finger in [str(name) for name in self.threshold_list("fingers")]:
            chain = FINGER_CHAINS[finger]
            knuckle = hand.point(chain[0]) - offset
            proximal_joint = hand.point(chain[1]) - offset
            distal_joint = hand.point(chain[2]) - offset
            segment_length = euclidean_distance(knuckle, proximal_joint)
            if segment_length < 4.0:
                continue
            axis = unit_vector(distal_joint - knuckle)
            joint_width = width_profile(mask, proximal_joint, axis, segment_length)
            proximal_shaft = width_profile(mask, (knuckle + proximal_joint) / 2.0, axis, segment_length)
            middle_shaft = width_profile(mask, (proximal_joint + distal_joint) / 2.0, axis, segment_length)
            shafts = [value for value in (proximal_shaft, middle_shaft) if value > 1.0]
            if joint_width <= 1.0 or not shafts:
                continue
            ratios.append(joint_width / float(np.mean(shafts)))
        return ratios

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure finger deviation angles and joint-to-shaft width ratios for each hand."""
        hands = frame_context.hands
        if hands is None or hands.count == 0:
            return []
        step_id = base_step_id(frame_context.step_instance or "")
        swelling_steps = [str(step) for step in self.threshold_list("swelling_steps")]
        readings: list[Reading] = []
        for hand in hands.hands:
            deviations = [signed_ulnar_deviation(hand, str(finger)) for finger in self.threshold_list("fingers")]
            finite = [value for value in deviations if np.isfinite(value)]
            if not finite:
                continue
            mean_deviation = float(np.mean(finite))
            self._deviation[hand.side].append(mean_deviation)
            readings.append(self.make_reading(frame_context, "mean_ulnar_deviation", mean_deviation, "degrees", f"{hand.side.capitalize()} hand mean finger deviation toward the little finger.", side=hand.side))
            if step_id in swelling_steps:
                ratios = self._joint_width_ratios(frame_context, hand)
                if ratios:
                    maximum_ratio = float(max(ratios))
                    self._width_ratio[hand.side].append(maximum_ratio)
                    readings.append(self.make_reading(frame_context, "joint_width_ratio", maximum_ratio, "ratio", f"{hand.side.capitalize()} hand widest finger joint relative to the finger shaft.", side=hand.side))
                box = hand_box(hand, self.threshold_float("crop_padding_fraction"))
                x_min, y_min, x_max, y_max = clip_box(box, frame_context.width, frame_context.height)
                crop = frame_context.colour_image()[y_min:y_max, x_min:x_max].copy()
                self._crops.offer(CropCandidate(crop, laplacian_sharpness(crop), frame_context.step_instance or step_id, frame_context.timestamp_s, f"back of {hand.side} hand"))
        return readings

    def _zero_shot(self, limitations: list[str]) -> tuple[float, list[EvidenceCrop]]:
        candidates = self._crops.items()
        results = self.safe_classify([candidate.image for candidate in candidates], str(self.threshold("zero_shot_task")), limitations)
        if not results:
            return float("nan"), []
        probability = self.mean_label_scores(results).get(str(self.threshold("swollen_label")), float("nan"))
        question = str(self.threshold("consultation_question"))
        evidence = []
        for candidate, result in zip(candidates, results):
            doubtful = self.needs_consultation(result)
            evidence.append(self.make_evidence_crop(candidate.label, candidate, result, doubtful, question if doubtful else None))
        return probability, evidence

    def finalize(self) -> AgentResult:
        """Evaluate ulnar deviation, joint swelling and left-right differences."""
        if not self.steps_with_data:
            return self.insufficient_result("Finger joints could not be measured.")
        alignment_steps = [str(step) for step in self.threshold_list("alignment_steps")]
        swelling_steps = [str(step) for step in self.threshold_list("swelling_steps")]
        alignment_confidence = self.data_confidence(alignment_steps)
        swelling_confidence = self.data_confidence(swelling_steps)
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        deviation_threshold = self.threshold_float("ulnar_deviation_mean_degrees")
        width_threshold = self.threshold_float("pip_width_ratio_flag")
        medians: dict[str, float] = {}
        for side in ("right", "left"):
            deviation = robust_median(self._deviation.get(side, []))
            width_ratio = robust_median(self._width_ratio.get(side, []))
            medians[side] = deviation
            metrics.append(self.summary_reading("mean_ulnar_deviation", deviation, "degrees", f"{side.capitalize()} hand median finger deviation toward the little finger.", alignment_confidence, ",".join(alignment_steps), side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("joint_width_ratio", width_ratio, "ratio", f"{side.capitalize()} hand widest joint-to-shaft ratio.", swelling_confidence, ",".join(swelling_steps), side=side))  # type: ignore[arg-type]
            if np.isfinite(deviation) and deviation >= deviation_threshold:
                flags.append(
                    self.make_flag(
                        "ulnar_deviation",
                        "Fingers drifting toward the little finger",
                        FlagSeverity.MODERATE,
                        self.exceedance_confidence(deviation, deviation_threshold, alignment_confidence),
                        f"The fingers of the {side} hand angled toward the little-finger side at the knuckles, a pattern seen in inflammatory arthritis.",
                        alignment_steps,
                        evidence=[f"mean deviation {deviation:.1f} degrees"],
                        side=side,  # type: ignore[arg-type]
                    )
                )
            if np.isfinite(width_ratio) and width_ratio >= width_threshold:
                flags.append(
                    self.make_flag(
                        "finger_joint_swelling",
                        "Widened finger joints",
                        FlagSeverity.MODERATE,
                        self.exceedance_confidence(width_ratio, width_threshold, swelling_confidence),
                        f"A finger joint of the {side} hand looked wider than the finger on either side of it, which can indicate swelling.",
                        swelling_steps,
                        evidence=[f"joint-to-shaft ratio {width_ratio:.2f}"],
                        side=side,  # type: ignore[arg-type]
                    )
                )
        if np.isfinite(medians.get("right", float("nan"))) and np.isfinite(medians.get("left", float("nan"))):
            difference = abs(medians["right"] - medians["left"])
            metrics.append(self.summary_reading("deviation_asymmetry", difference, "degrees", "Left-right difference in finger deviation.", alignment_confidence, ",".join(alignment_steps)))
            asymmetry_threshold = self.threshold_float("asymmetry_degrees")
            if difference >= asymmetry_threshold and not any(flag.code == "ulnar_deviation" for flag in flags):
                worse = "right" if medians["right"] > medians["left"] else "left"
                flags.append(
                    self.make_flag(
                        "finger_alignment_asymmetry",
                        "Left-right difference in finger alignment",
                        FlagSeverity.LOW,
                        self.exceedance_confidence(difference, asymmetry_threshold, alignment_confidence),
                        f"Finger alignment differed between hands; the {worse} hand angled more toward the little finger.",
                        alignment_steps,
                        evidence=[f"difference {difference:.1f} degrees"],
                        side=worse,  # type: ignore[arg-type]
                    )
                )
        probability, evidence = self._zero_shot(limitations)
        metrics.append(self.summary_reading("zero_shot_knuckle_swelling_probability", probability, "probability", "Zero-shot similarity of hand crops to swollen knuckles.", swelling_confidence, ",".join(swelling_steps)))
        if np.isfinite(probability) and probability >= self.threshold_float("zero_shot_flag_probability") and not any(flag.code == "finger_joint_swelling" for flag in flags):
            flags.append(
                self.make_flag(
                    "finger_joint_swelling",
                    "Possible knuckle swelling",
                    FlagSeverity.MODERATE,
                    probability * swelling_confidence,
                    "Image comparison suggested swollen knuckles.",
                    swelling_steps,
                    evidence=[f"zero-shot probability {probability:.2f}"],
                )
            )
        return self.build_result(
            description=f"Finger deviation: right {medians.get('right', float('nan')):.1f} degrees, left {medians.get('left', float('nan')):.1f} degrees.",
            metrics=metrics,
            flags=flags,
            confidence=float(np.mean([value for value in (alignment_confidence, swelling_confidence) if value > 0] or [0.0])),
            value=float(np.nanmax([medians.get("right", np.nan), medians.get("left", np.nan)])) if any(np.isfinite(value) for value in medians.values()) else None,
            unit="degrees",
            evidence=evidence,
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._deviation.clear()
        self._width_ratio.clear()
        self._crops.clear()
