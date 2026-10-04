"""Face colour agent: lip and facial skin colour in CIELAB for cyanosis and pallor screening."""

import cv2
import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import masked_lab_statistics
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import lips_mask, reference_skin_mask


class FaceColourAgent(AreaAgent):
    """Compares lip colour with the patient's own cheek and forehead skin."""

    name = "face_colour"
    area = "face"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._lip_a: list[float] = []
        self._lip_b: list[float] = []
        self._lip_lightness: list[float] = []
        self._skin_a: list[float] = []
        self._skin_lightness: list[float] = []
        self._delta_a: list[float] = []
        self._delta_b: list[float] = []
        self._calibration_confidence: list[float] = []

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure lip and reference skin CIELAB medians for one frame."""
        face = frame_context.face
        if face is None:
            return []
        lab = frame_context.lab()
        shape = lab.shape[:2]
        lip_region = cv2.erode(lips_mask(face, shape), np.ones((3, 3), dtype=np.uint8))
        minimum_pixels = self.threshold_int("min_region_pixels")
        lip_statistics = masked_lab_statistics(lab, lip_region, minimum_pixels)
        skin_statistics = masked_lab_statistics(lab, reference_skin_mask(face, shape), minimum_pixels)
        if lip_statistics is None or skin_statistics is None:
            return []
        delta_a = lip_statistics.red_green - skin_statistics.red_green
        delta_b = lip_statistics.yellow_blue - skin_statistics.yellow_blue
        self._lip_a.append(lip_statistics.red_green)
        self._lip_b.append(lip_statistics.yellow_blue)
        self._lip_lightness.append(lip_statistics.lightness)
        self._skin_a.append(skin_statistics.red_green)
        self._skin_lightness.append(skin_statistics.lightness)
        self._delta_a.append(delta_a)
        self._delta_b.append(delta_b)
        self._calibration_confidence.append(frame_context.calibration_confidence)
        colour_confidence = frame_context.calibration_confidence
        return [
            self.make_reading(frame_context, "lip_a_star", lip_statistics.red_green, "CIELAB a*", "Lip redness.", confidence=colour_confidence),
            self.make_reading(frame_context, "lip_minus_skin_a_star", delta_a, "CIELAB a*", "Lip redness relative to the patient's own skin.", confidence=colour_confidence),
            self.make_reading(frame_context, "lip_minus_skin_b_star", delta_b, "CIELAB b*", "Lip yellow-blue difference relative to skin (negative is bluer).", confidence=colour_confidence),
        ]

    def finalize(self) -> AgentResult:
        """Evaluate cyanosis and pallor thresholds on robust medians."""
        if not self._lip_a:
            return self.insufficient_result("Lip and skin regions could not be measured.")
        calibration = robust_median(self._calibration_confidence)
        confidence = self.data_confidence(self.steps_with_data) * calibration
        steps = ",".join(self.steps_with_data)
        lip_a = robust_median(self._lip_a)
        lip_b = robust_median(self._lip_b)
        delta_a = robust_median(self._delta_a)
        delta_b = robust_median(self._delta_b)
        skin_a = robust_median(self._skin_a)
        metrics = [
            self.summary_reading("lip_a_star", lip_a, "CIELAB a*", "Median lip redness.", confidence, steps),
            self.summary_reading("lip_b_star", lip_b, "CIELAB b*", "Median lip yellow-blue value.", confidence, steps),
            self.summary_reading("lip_lightness", robust_median(self._lip_lightness), "CIELAB L*", "Median lip lightness.", confidence, steps),
            self.summary_reading("skin_a_star", skin_a, "CIELAB a*", "Median reference skin redness.", confidence, steps),
            self.summary_reading("skin_lightness", robust_median(self._skin_lightness), "CIELAB L*", "Median reference skin lightness.", confidence, steps),
            self.summary_reading("lip_minus_skin_a_star", delta_a, "CIELAB a*", "Lip redness relative to skin.", confidence, steps),
            self.summary_reading("lip_minus_skin_b_star", delta_b, "CIELAB b*", "Lip yellow-blue difference relative to skin.", confidence, steps),
            self.summary_reading("calibration_confidence", calibration, "fraction", "Median colour calibration confidence.", calibration, steps),
        ]
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        if calibration < self.threshold_float("min_calibration_confidence"):
            limitations.append("Colour calibration confidence was low; colour findings are less reliable.")
        cyanosis_a = self.threshold_float("cyanosis_lip_skin_delta_a_max")
        cyanosis_b = self.threshold_float("cyanosis_lip_skin_delta_b_max")
        pallor_lip_a = self.threshold_float("pallor_lip_a_max")
        pallor_delta_a = self.threshold_float("pallor_lip_skin_delta_a_max")
        cyanosis = delta_a <= cyanosis_a and delta_b <= cyanosis_b
        if cyanosis:
            flags.append(
                self.make_flag(
                    "central_cyanosis_lips",
                    "Bluish lip colour",
                    FlagSeverity.HIGH,
                    confidence * min(1.0, 0.6 + 0.1 * abs(delta_b - cyanosis_b)),
                    "The lips looked bluish and not redder than the surrounding skin, which can indicate low blood oxygen.",
                    self.steps_with_data,
                    evidence=[f"lip minus skin a* {delta_a:.1f}", f"lip minus skin b* {delta_b:.1f}"],
                )
            )
        elif lip_a <= pallor_lip_a and delta_a <= pallor_delta_a:
            flags.append(
                self.make_flag(
                    "facial_pallor",
                    "Pale lips and face",
                    FlagSeverity.MODERATE,
                    self.shortfall_confidence(lip_a, pallor_lip_a, confidence) if pallor_lip_a > 0 else confidence * 0.5,
                    "The lips were pale and close to the skin colour, which can accompany anaemia or poor circulation.",
                    self.steps_with_data,
                    evidence=[f"lip a* {lip_a:.1f}", f"lip minus skin a* {delta_a:.1f}"],
                )
            )
        description = f"Lip redness a* {lip_a:.1f} ({delta_a:+.1f} relative to skin), lip b* difference {delta_b:+.1f}."
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=confidence,
            value=delta_a,
            unit="CIELAB a*",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        for buffer in (
            self._lip_a,
            self._lip_b,
            self._lip_lightness,
            self._skin_a,
            self._skin_lightness,
            self._delta_a,
            self._delta_b,
            self._calibration_confidence,
        ):
            buffer.clear()
