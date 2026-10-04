"""Eye appearance agent: sclera yellowness and redness, conjunctival pallor and zero-shot eye findings."""

from typing import Any

import cv2
import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, BestCropBuffer, CropCandidate, robust_median
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, EvidenceCrop, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import crop_image, laplacian_sharpness, masked_lab_statistics
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import both_eyes_crop_box, cheek_mask, lower_conjunctiva_mask, sclera_mask


class EyeAppearanceAgent(AreaAgent):
    """Measures sclera and conjunctiva colour in CIELAB and classifies eye crops zero-shot."""

    name = "eye_appearance"
    area = "eyes"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._sclera_b: dict[str, list[float]] = {"right": [], "left": []}
        self._sclera_a: dict[str, list[float]] = {"right": [], "left": []}
        self._sclera_skin_ratio: list[float] = []
        self._conjunctiva_a: dict[str, list[float]] = {"right": [], "left": []}
        self._conjunctiva_skin_delta_a: list[float] = []
        self._calibration: list[float] = []
        self._crops = BestCropBuffer(self.threshold_int("crops_to_keep"))

    def _skin_statistics(self, frame_context: FrameContext, lab: np.ndarray):
        face = frame_context.face
        shape = lab.shape[:2]
        skin_region = cv2.bitwise_or(cheek_mask(face, "right", shape), cheek_mask(face, "left", shape))  # type: ignore[arg-type]
        return masked_lab_statistics(lab, skin_region, self.threshold_int("min_sclera_pixels"))

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure sclera colour in close-up steps and conjunctiva colour in the lower-eyelid step."""
        face = frame_context.face
        if face is None:
            return []
        step_id = base_step_id(frame_context.step_instance or "")
        lab = frame_context.lab()
        shape = lab.shape[:2]
        readings: list[Reading] = []
        colour_confidence = frame_context.calibration_confidence
        skin_statistics = self._skin_statistics(frame_context, lab)
        if step_id in [str(step) for step in self.threshold_list("sclera_steps")]:
            bright_enough = (lab[:, :, 0] >= self.threshold_float("sclera_min_lightness")).astype(np.uint8) * 255
            sclera_b_values: list[float] = []
            for side in ("right", "left"):
                region = cv2.bitwise_and(sclera_mask(face, side, shape), bright_enough)
                region = cv2.erode(region, np.ones((3, 3), dtype=np.uint8))
                statistics = masked_lab_statistics(lab, region, self.threshold_int("min_sclera_pixels"))
                if statistics is None:
                    continue
                self._sclera_b[side].append(statistics.yellow_blue)
                self._sclera_a[side].append(statistics.red_green)
                sclera_b_values.append(statistics.yellow_blue)
                readings.append(self.make_reading(frame_context, "sclera_b_star", statistics.yellow_blue, "CIELAB b*", f"{side.capitalize()} sclera yellowness.", confidence=colour_confidence, side=side))  # type: ignore[arg-type]
                readings.append(self.make_reading(frame_context, "sclera_a_star", statistics.red_green, "CIELAB a*", f"{side.capitalize()} sclera redness.", confidence=colour_confidence, side=side))  # type: ignore[arg-type]
            if sclera_b_values and skin_statistics is not None and skin_statistics.yellow_blue > 1.0:
                self._sclera_skin_ratio.append(float(np.mean(sclera_b_values)) / skin_statistics.yellow_blue)
            if readings:
                crop = crop_image(frame_context.colour_image(), both_eyes_crop_box(face, self.threshold_float("crop_padding_fraction")))
                self._crops.offer(
                    CropCandidate(
                        image=crop,
                        sharpness=laplacian_sharpness(crop),
                        protocol_step=frame_context.step_instance or step_id,
                        timestamp=frame_context.timestamp_s,
                        label="eyes close-up",
                    )
                )
        elif step_id in [str(step) for step in self.threshold_list("conjunctiva_steps")]:
            if skin_statistics is None:
                return []
            skin_reference = skin_statistics.as_array()
            exclusion = self.threshold_float("skin_exclusion_delta_e")
            colour_distance = np.linalg.norm(lab - skin_reference.reshape(1, 1, 3), axis=2)
            not_skin = ((colour_distance >= exclusion) & (lab[:, :, 0] >= 25.0)).astype(np.uint8) * 255
            for side in ("right", "left"):
                band = lower_conjunctiva_mask(face, side, shape, self.threshold_float("conjunctiva_band_fraction"))
                region = cv2.bitwise_and(band, not_skin)
                statistics = masked_lab_statistics(lab, region, self.threshold_int("min_conjunctiva_pixels"))
                if statistics is None:
                    continue
                self._conjunctiva_a[side].append(statistics.red_green)
                self._conjunctiva_skin_delta_a.append(statistics.red_green - skin_statistics.red_green)
                readings.append(self.make_reading(frame_context, "conjunctiva_a_star", statistics.red_green, "CIELAB a*", f"{side.capitalize()} lower-eyelid conjunctiva redness.", confidence=colour_confidence, side=side))  # type: ignore[arg-type]
        if readings:
            self._calibration.append(colour_confidence)
        return readings

    def _zero_shot(self, limitations: list[str]) -> tuple[list[ScreeningFlag], list[EvidenceCrop], dict[str, float]]:
        candidates = self._crops.items()
        if not candidates:
            return [], [], {}
        task = str(self.threshold("zero_shot_task"))
        results = self.safe_classify([candidate.image for candidate in candidates], task, limitations)
        if not results:
            return [], [], {}
        mean_scores = self.mean_label_scores(results)
        normal_label = str(self.threshold("normal_label"))
        flag_probability = self.threshold_float("zero_shot_flag_probability")
        label_flags: dict[str, dict[str, Any]] = self.threshold("label_flags")
        flags: list[ScreeningFlag] = []
        sclera_steps = [str(step) for step in self.threshold_list("sclera_steps")]
        for label, probability in mean_scores.items():
            if label == normal_label or probability < flag_probability or label not in label_flags:
                continue
            definition = label_flags[label]
            flags.append(
                self.make_flag(
                    str(definition["code"]),
                    str(definition["title"]),
                    FlagSeverity(str(definition["severity"])),
                    probability * self.data_confidence(sclera_steps),
                    f"Image comparison suggested an appearance resembling {label.replace('_', ' ')} (zero-shot probability {probability:.2f}).",
                    sclera_steps,
                    evidence=[f"{label}: {probability:.2f}"],
                )
            )
        evidence: list[EvidenceCrop] = []
        question = str(self.threshold("consultation_question"))
        for candidate, result in zip(candidates, results):
            doubtful = self.needs_consultation(result)
            evidence.append(
                self.make_evidence_crop(
                    "eyes close-up",
                    candidate,
                    classification=result,
                    needs_consultation=doubtful,
                    consultation_question=question if doubtful else None,
                )
            )
        return flags, evidence, mean_scores

    def finalize(self) -> AgentResult:
        """Evaluate jaundice, redness and pallor thresholds and zero-shot eye findings."""
        if not self.steps_with_data:
            return self.insufficient_result("The sclera and conjunctiva could not be measured.")
        sclera_steps = [str(step) for step in self.threshold_list("sclera_steps")]
        conjunctiva_steps = [str(step) for step in self.threshold_list("conjunctiva_steps")]
        calibration = robust_median(self._calibration) if self._calibration else 0.0
        sclera_confidence = self.data_confidence(sclera_steps) * calibration
        conjunctiva_confidence = self.data_confidence(conjunctiva_steps) * calibration
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        if calibration < self.threshold_float("min_calibration_confidence"):
            limitations.append("Colour calibration confidence was low; eye colour findings are less reliable.")
        sclera_b = robust_median(self._sclera_b["right"] + self._sclera_b["left"])
        sclera_a = robust_median(self._sclera_a["right"] + self._sclera_a["left"])
        ratio = robust_median(self._sclera_skin_ratio)
        conjunctiva_a = robust_median(self._conjunctiva_a["right"] + self._conjunctiva_a["left"])
        conjunctiva_delta = robust_median(self._conjunctiva_skin_delta_a)
        for side in ("right", "left"):
            metrics.append(self.summary_reading("sclera_b_star", robust_median(self._sclera_b[side]), "CIELAB b*", f"{side.capitalize()} sclera yellowness.", sclera_confidence, ",".join(sclera_steps), side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("sclera_a_star", robust_median(self._sclera_a[side]), "CIELAB a*", f"{side.capitalize()} sclera redness.", sclera_confidence, ",".join(sclera_steps), side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("conjunctiva_a_star", robust_median(self._conjunctiva_a[side]), "CIELAB a*", f"{side.capitalize()} conjunctiva redness.", conjunctiva_confidence, ",".join(conjunctiva_steps), side=side))  # type: ignore[arg-type]
        metrics.append(self.summary_reading("sclera_to_skin_yellowness_ratio", ratio, "ratio", "Sclera yellowness relative to cheek skin yellowness.", sclera_confidence, ",".join(sclera_steps)))
        metrics.append(self.summary_reading("conjunctiva_minus_skin_a_star", conjunctiva_delta, "CIELAB a*", "Conjunctiva redness relative to skin.", conjunctiva_confidence, ",".join(conjunctiva_steps)))
        jaundice_b = self.threshold_float("jaundice_sclera_b_min")
        jaundice_ratio = self.threshold_float("jaundice_sclera_to_skin_b_ratio_min")
        if np.isfinite(sclera_b) and sclera_b >= jaundice_b and (not np.isfinite(ratio) or ratio >= jaundice_ratio):
            flags.append(
                self.make_flag(
                    "scleral_icterus",
                    "Yellowish whites of the eyes",
                    FlagSeverity.MODERATE,
                    self.exceedance_confidence(sclera_b, jaundice_b, sclera_confidence),
                    "The whites of the eyes looked yellowish, which can be a sign of jaundice.",
                    sclera_steps,
                    evidence=[f"sclera b* {sclera_b:.1f}", f"sclera/skin ratio {ratio:.2f}"],
                )
            )
        redness_threshold = self.threshold_float("sclera_redness_a_min")
        if np.isfinite(sclera_a) and sclera_a >= redness_threshold:
            flags.append(
                self.make_flag(
                    "scleral_redness",
                    "Red whites of the eyes",
                    FlagSeverity.LOW,
                    self.exceedance_confidence(sclera_a, redness_threshold, sclera_confidence),
                    "The whites of the eyes looked red, which can accompany irritation, infection or dryness.",
                    sclera_steps,
                    evidence=[f"sclera a* {sclera_a:.1f}"],
                )
            )
        pallor_a = self.threshold_float("conjunctiva_pallor_a_max")
        pallor_delta = self.threshold_float("conjunctiva_pallor_skin_delta_a_max")
        if np.isfinite(conjunctiva_a) and conjunctiva_a <= pallor_a and (not np.isfinite(conjunctiva_delta) or conjunctiva_delta <= pallor_delta):
            flags.append(
                self.make_flag(
                    "conjunctival_pallor",
                    "Pale inner eyelid",
                    FlagSeverity.MODERATE,
                    self.shortfall_confidence(conjunctiva_a, pallor_a, conjunctiva_confidence),
                    "The inside of the lower eyelid looked pale, which can be a sign of anaemia.",
                    conjunctiva_steps,
                    evidence=[f"conjunctiva a* {conjunctiva_a:.1f}", f"relative to skin {conjunctiva_delta:+.1f}"],
                )
            )
        zero_shot_flags, evidence, scores = self._zero_shot(limitations)
        flags.extend(zero_shot_flags)
        for label, probability in scores.items():
            metrics.append(self.summary_reading(f"zero_shot_{label}", probability, "probability", f"Zero-shot similarity to '{label.replace('_', ' ')}'.", sclera_confidence, ",".join(sclera_steps)))
        description = f"Sclera yellowness b* {sclera_b:.1f}, sclera redness a* {sclera_a:.1f}, conjunctiva redness a* {conjunctiva_a:.1f}."
        confidence_values = [value for value in (sclera_confidence, conjunctiva_confidence) if value > 0]
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
            value=sclera_b,
            unit="CIELAB b*",
            evidence=evidence,
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        for buffers in (self._sclera_b, self._sclera_a, self._conjunctiva_a):
            for values in buffers.values():
                values.clear()
        self._sclera_skin_ratio.clear()
        self._conjunctiva_skin_delta_a.clear()
        self._calibration.clear()
        self._crops.clear()
