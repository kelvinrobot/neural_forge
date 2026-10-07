"""Hand colour agent: palm pallor against the patient's own forearm skin and nail-bed cyanosis."""

import cv2
import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.models.neural_forge_hands import HandLandmarks
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import LabStatistics, masked_lab_statistics, skin_mask_ycrcb
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import forearm_patch_mask, nail_bed_mask, palm_mask


class HandColourAgent(AreaAgent):
    """Measures palm and nail-bed colour in CIELAB relative to reference skin."""

    name = "hand_colour"
    area = "hands"
    required_trackers = frozenset({TrackerName.HANDS})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._palm_a: list[float] = []
        self._palm_delta_a: list[float] = []
        self._palm_lightness: list[float] = []
        self._nail_a: list[float] = []
        self._nail_delta_a: list[float] = []
        self._nail_delta_b: list[float] = []
        self._calibration: list[float] = []

    def _forearm_reference(self, frame_context: FrameContext, hand: HandLandmarks, lab: np.ndarray, skin: np.ndarray) -> LabStatistics | None:
        region = forearm_patch_mask(
            hand,
            lab.shape[:2],
            self.threshold_float("forearm_length_fraction"),
            self.threshold_float("forearm_width_fraction"),
        )
        return masked_lab_statistics(lab, cv2.bitwise_and(region, skin), self.threshold_int("min_region_pixels"))

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure palm colour on the palm step and nail-bed colour on the back-of-hand step."""
        hands = frame_context.hands
        if hands is None or hands.count == 0:
            return []
        step_id = base_step_id(frame_context.step_instance or "")
        lab = frame_context.lab()
        shape = lab.shape[:2]
        skin = skin_mask_ycrcb(frame_context.colour_image(), self.threshold_range("skin_cr_range"), self.threshold_range("skin_cb_range"))
        minimum_pixels = self.threshold_int("min_region_pixels")
        colour_confidence = frame_context.calibration_confidence
        readings: list[Reading] = []
        palm_steps = [str(step) for step in self.threshold_list("palm_steps")]
        nail_steps = [str(step) for step in self.threshold_list("nail_steps")]
        for hand in hands.hands:
            if step_id in palm_steps:
                palm_statistics = masked_lab_statistics(lab, cv2.bitwise_and(palm_mask(hand, shape), skin), minimum_pixels)
                if palm_statistics is None:
                    continue
                reference = self._forearm_reference(frame_context, hand, lab, skin)
                self._palm_a.append(palm_statistics.red_green)
                self._palm_lightness.append(palm_statistics.lightness)
                readings.append(self.make_reading(frame_context, "palm_a_star", palm_statistics.red_green, "CIELAB a*", f"{hand.side.capitalize()} palm redness.", confidence=colour_confidence, side=hand.side))
                if reference is not None:
                    delta_a = palm_statistics.red_green - reference.red_green
                    self._palm_delta_a.append(delta_a)
                    readings.append(self.make_reading(frame_context, "palm_minus_forearm_a_star", delta_a, "CIELAB a*", f"{hand.side.capitalize()} palm redness relative to forearm skin.", confidence=colour_confidence, side=hand.side))
            elif step_id in nail_steps:
                nail_region = np.zeros(shape, dtype=np.uint8)
                for finger in [str(name) for name in self.threshold_list("nail_fingers")]:
                    nail_region = cv2.bitwise_or(nail_region, nail_bed_mask(hand, finger, shape, self.threshold_float("nail_roi_radius_fraction")))
                nail_statistics = masked_lab_statistics(lab, nail_region, minimum_pixels)
                dorsal_statistics = masked_lab_statistics(lab, cv2.bitwise_and(palm_mask(hand, shape), skin), minimum_pixels)
                if nail_statistics is None or dorsal_statistics is None:
                    continue
                delta_a = nail_statistics.red_green - dorsal_statistics.red_green
                delta_b = nail_statistics.yellow_blue - dorsal_statistics.yellow_blue
                self._nail_a.append(nail_statistics.red_green)
                self._nail_delta_a.append(delta_a)
                self._nail_delta_b.append(delta_b)
                readings.append(self.make_reading(frame_context, "nail_bed_a_star", nail_statistics.red_green, "CIELAB a*", f"{hand.side.capitalize()} nail-bed redness.", confidence=colour_confidence, side=hand.side))
                readings.append(self.make_reading(frame_context, "nail_minus_skin_b_star", delta_b, "CIELAB b*", f"{hand.side.capitalize()} nail-bed yellow-blue relative to skin (negative is bluer).", confidence=colour_confidence, side=hand.side))
        if readings:
            self._calibration.append(colour_confidence)
        return readings

    def finalize(self) -> AgentResult:
        """Evaluate palmar pallor and nail-bed cyanosis thresholds."""
        if not self.steps_with_data:
            return self.insufficient_result("Hand colour could not be measured.")
        palm_steps = [str(step) for step in self.threshold_list("palm_steps")]
        nail_steps = [str(step) for step in self.threshold_list("nail_steps")]
        calibration = robust_median(self._calibration) if self._calibration else 0.0
        palm_confidence = self.data_confidence(palm_steps) * calibration
        nail_confidence = self.data_confidence(nail_steps) * calibration
        palm_a = robust_median(self._palm_a)
        palm_delta = robust_median(self._palm_delta_a)
        nail_a = robust_median(self._nail_a)
        nail_delta_a = robust_median(self._nail_delta_a)
        nail_delta_b = robust_median(self._nail_delta_b)
        metrics = [
            self.summary_reading("palm_a_star", palm_a, "CIELAB a*", "Median palm redness.", palm_confidence, ",".join(palm_steps)),
            self.summary_reading("palm_lightness", robust_median(self._palm_lightness), "CIELAB L*", "Median palm lightness.", palm_confidence, ",".join(palm_steps)),
            self.summary_reading("palm_minus_forearm_a_star", palm_delta, "CIELAB a*", "Palm redness relative to forearm skin.", palm_confidence, ",".join(palm_steps)),
            self.summary_reading("nail_bed_a_star", nail_a, "CIELAB a*", "Median nail-bed redness.", nail_confidence, ",".join(nail_steps)),
            self.summary_reading("nail_minus_skin_a_star", nail_delta_a, "CIELAB a*", "Nail-bed redness relative to back-of-hand skin.", nail_confidence, ",".join(nail_steps)),
            self.summary_reading("nail_minus_skin_b_star", nail_delta_b, "CIELAB b*", "Nail-bed yellow-blue relative to back-of-hand skin.", nail_confidence, ",".join(nail_steps)),
        ]
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        if calibration < self.threshold_float("min_calibration_confidence"):
            limitations.append("Colour calibration confidence was low; hand colour findings are less reliable.")
        pallor_a = self.threshold_float("palm_pallor_a_max")
        pallor_delta = self.threshold_float("palm_pallor_skin_delta_a_max")
        if np.isfinite(palm_a) and palm_a <= pallor_a and (not np.isfinite(palm_delta) or palm_delta <= pallor_delta):
            flags.append(
                self.make_flag(
                    "palmar_pallor",
                    "Pale palms",
                    FlagSeverity.MODERATE,
                    self.shortfall_confidence(palm_a, pallor_a, palm_confidence),
                    "The palms looked pale compared with the forearm skin, which can be a sign of anaemia.",
                    palm_steps,
                    evidence=[f"palm a* {palm_a:.1f}", f"palm minus forearm a* {palm_delta:+.1f}"],
                )
            )
        cyanosis_b = self.threshold_float("nail_cyanosis_skin_delta_b_max")
        cyanosis_a = self.threshold_float("nail_cyanosis_skin_delta_a_max")
        if np.isfinite(nail_delta_b) and nail_delta_b <= cyanosis_b and np.isfinite(nail_delta_a) and nail_delta_a <= cyanosis_a:
            flags.append(
                self.make_flag(
                    "nail_bed_cyanosis",
                    "Bluish nail beds",
                    FlagSeverity.HIGH,
                    self.exceedance_confidence(nail_delta_b, cyanosis_b, nail_confidence),
                    "The nail beds looked bluish compared with the surrounding skin, which can indicate poor oxygen delivery.",
                    nail_steps,
                    evidence=[f"nail minus skin b* {nail_delta_b:+.1f}", f"nail minus skin a* {nail_delta_a:+.1f}"],
                )
            )
        confidence_values = [value for value in (palm_confidence, nail_confidence) if value > 0]
        return self.build_result(
            description=f"Palm redness a* {palm_a:.1f} ({palm_delta:+.1f} vs forearm); nail-bed b* difference {nail_delta_b:+.1f}.",
            metrics=metrics,
            flags=flags,
            confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
            value=palm_a,
            unit="CIELAB a*",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        for buffer in (self._palm_a, self._palm_delta_a, self._palm_lightness, self._nail_a, self._nail_delta_a, self._nail_delta_b, self._calibration):
            buffer.clear()
