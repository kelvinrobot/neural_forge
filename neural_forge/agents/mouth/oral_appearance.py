"""Oral appearance agent: tongue and gum colour, white patches and zero-shot oral findings."""

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, BestCropBuffer, CropCandidate, robust_median
from neural_forge.agents.mouth.mouth_movement import segment_tongue
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, EvidenceCrop, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import crop_image, laplacian_sharpness, masked_lab_statistics
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import mouth_corner_box, mouth_crop_box, mouth_interior_mask, mouth_width, reference_skin_mask


class OralAppearanceAgent(AreaAgent):
    """Collects mouth-interior, gum and lip-corner crops and measures oral colours."""

    name = "oral_appearance"
    area = "mouth"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        capacity = self.threshold_int("crops_to_keep")
        self._crops: dict[str, BestCropBuffer] = {
            "oral_interior": BestCropBuffer(capacity),
            "gums": BestCropBuffer(capacity),
            "lip_corners": BestCropBuffer(capacity * 2),
        }
        self._values: dict[str, list[float]] = defaultdict(list)
        self._calibration: list[float] = []

    def _role(self, step_id: str) -> str | None:
        for role, key in (("interior", "interior_steps"), ("gums", "gum_steps"), ("corners", "corner_steps"), ("tongue", "tongue_steps")):
            if step_id in [str(step) for step in self.threshold_list(key)]:
                return role
        return None

    def _offer(self, task: str, image: np.ndarray, frame_context: FrameContext, label: str) -> None:
        self._crops[task].offer(
            CropCandidate(
                image=image,
                sharpness=laplacian_sharpness(image),
                protocol_step=frame_context.step_instance or "",
                timestamp=frame_context.timestamp_s,
                label=label,
            )
        )

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure oral colours and keep the sharpest crops for each region."""
        face = frame_context.face
        if face is None:
            return []
        role = self._role(base_step_id(frame_context.step_instance or ""))
        if role is None:
            return []
        lab = frame_context.lab()
        shape = lab.shape[:2]
        image = frame_context.colour_image()
        colour_confidence = frame_context.calibration_confidence
        skin = masked_lab_statistics(lab, reference_skin_mask(face, shape), 100)
        readings: list[Reading] = []
        padding = self.threshold_float("crop_padding_fraction")
        if role == "tongue":
            segmentation = segment_tongue(
                face,
                lab,
                self.threshold_float("search_extension_mouth_widths"),
                self.threshold_float("tongue_skin_delta_a"),
                self.threshold_float("teeth_min_lightness"),
                self.threshold_float("teeth_max_chroma"),
                self.threshold_int("tongue_min_pixels"),
            )
            if segmentation is None:
                return []
            statistics = masked_lab_statistics(lab, segmentation.mask, self.threshold_int("tongue_min_pixels"))
            if statistics is None:
                return []
            self._values["tongue_a"].append(statistics.red_green)
            self._values["tongue_b"].append(statistics.yellow_blue)
            self._values["tongue_lightness"].append(statistics.lightness)
            ys, xs = np.nonzero(segmentation.mask)
            box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            width = box[2] - box[0]
            height = box[3] - box[1]
            self._offer("oral_interior", crop_image(image, (box[0] - padding * width, box[1] - padding * height, box[2] + padding * width, box[3] + padding * height)), frame_context, "tongue")
            readings.append(self.make_reading(frame_context, "tongue_a_star", statistics.red_green, "CIELAB a*", "Tongue redness.", confidence=colour_confidence))
            readings.append(self.make_reading(frame_context, "tongue_b_star", statistics.yellow_blue, "CIELAB b*", "Tongue yellow-blue value (negative is bluer).", confidence=colour_confidence))
        elif role == "interior":
            interior = mouth_interior_mask(face, shape)
            erosion = max(3, int(mouth_width(face) * self.threshold_float("interior_erosion_mouth_fraction")) | 1)
            core = cv2.erode(interior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion)))
            total = int((core > 0).sum())
            if total < self.threshold_int("min_interior_pixels"):
                return []
            lightness = lab[:, :, 0]
            chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
            white = (lightness >= self.threshold_float("white_patch_min_lightness")) & (chroma <= self.threshold_float("white_patch_max_chroma")) & (core > 0)
            white_fraction = float(white.sum()) / total
            self._values["white_patch_fraction"].append(white_fraction)
            self._offer("oral_interior", crop_image(image, mouth_crop_box(face, padding)), frame_context, "mouth interior")
            readings.append(self.make_reading(frame_context, "white_patch_fraction", white_fraction, "fraction", "Fraction of the inner mouth that is white and colourless."))
        elif role == "gums":
            interior = mouth_interior_mask(face, shape)
            if skin is None:
                return []
            reddish = (lab[:, :, 1] >= skin.red_green + self.threshold_float("gum_skin_delta_a_min")).astype(np.uint8) * 255
            statistics = masked_lab_statistics(lab, cv2.bitwise_and(interior, reddish), self.threshold_int("min_interior_pixels"))
            self._offer("gums", crop_image(image, mouth_crop_box(face, padding)), frame_context, "gums")
            if statistics is None:
                return [self.make_reading(frame_context, "gum_region_visible", 0.0, "flag", "Gum tissue was not clearly separated in this frame.")]
            delta = statistics.red_green - skin.red_green
            self._values["gum_minus_skin_a"].append(delta)
            readings.append(self.make_reading(frame_context, "gum_minus_skin_a_star", delta, "CIELAB a*", "Gum redness relative to skin.", confidence=colour_confidence))
        elif role == "corners":
            corner_fraction = self.threshold_float("corner_box_fraction")
            for side in ("right", "left"):
                self._offer("lip_corners", crop_image(image, mouth_corner_box(face, side, corner_fraction * 2.0)), frame_context, f"{side} mouth corner")
            readings.append(self.make_reading(frame_context, "mouth_corner_crops", 2.0, "count", "Mouth corner crops captured."))
        if readings:
            self._calibration.append(colour_confidence)
        return readings

    def _classify(self, limitations: list[str]) -> tuple[list[ScreeningFlag], list[EvidenceCrop], list[Reading]]:
        tasks: dict[str, str] = {str(key): str(value) for key, value in self.threshold("zero_shot_tasks").items()}
        normal_labels: dict[str, str] = {str(key): str(value) for key, value in self.threshold("normal_labels").items()}
        questions: dict[str, str] = {str(key): str(value) for key, value in self.threshold("consultation_questions").items()}
        label_flags: dict[str, dict[str, Any]] = self.threshold("label_flags")
        flag_probability = self.threshold_float("zero_shot_flag_probability")
        flags: list[ScreeningFlag] = []
        evidence: list[EvidenceCrop] = []
        metrics: list[Reading] = []
        for region, task in tasks.items():
            candidates = self._crops[region].items()
            results = self.safe_classify([candidate.image for candidate in candidates], task, limitations)
            if not results:
                continue
            scores = self.mean_label_scores(results)
            steps = sorted({candidate.protocol_step for candidate in candidates})
            confidence = self.data_confidence(steps)
            for label, probability in scores.items():
                metrics.append(self.summary_reading(f"zero_shot_{label}", probability, "probability", f"Zero-shot similarity to '{label.replace('_', ' ')}'.", confidence, ",".join(steps)))
                if label == normal_labels.get(region) or probability < flag_probability or label not in label_flags:
                    continue
                definition = label_flags[label]
                flags.append(
                    self.make_flag(
                        str(definition["code"]),
                        str(definition["title"]),
                        FlagSeverity(str(definition["severity"])),
                        probability * confidence,
                        f"Image comparison suggested an appearance resembling {label.replace('_', ' ')} (zero-shot probability {probability:.2f}).",
                        steps,
                        evidence=[f"{label}: {probability:.2f}"],
                    )
                )
            for candidate, result in zip(candidates, results):
                doubtful = self.needs_consultation(result)
                evidence.append(self.make_evidence_crop(candidate.label, candidate, result, doubtful, questions.get(region) if doubtful else None))
        return flags, evidence, metrics

    def finalize(self) -> AgentResult:
        """Evaluate colour thresholds and zero-shot findings for the mouth."""
        if not self.steps_with_data:
            return self.insufficient_result("The inside of the mouth could not be examined.")
        calibration = robust_median(self._calibration) if self._calibration else 0.0
        tongue_steps = [str(step) for step in self.threshold_list("tongue_steps")]
        interior_steps = [str(step) for step in self.threshold_list("interior_steps")]
        gum_steps = [str(step) for step in self.threshold_list("gum_steps")]
        tongue_confidence = self.data_confidence(tongue_steps) * calibration
        interior_confidence = self.data_confidence(interior_steps)
        gum_confidence = self.data_confidence(gum_steps) * calibration
        tongue_a = robust_median(self._values.get("tongue_a", []))
        tongue_b = robust_median(self._values.get("tongue_b", []))
        white_fraction = robust_median(self._values.get("white_patch_fraction", []))
        gum_delta = robust_median(self._values.get("gum_minus_skin_a", []))
        metrics = [
            self.summary_reading("tongue_a_star", tongue_a, "CIELAB a*", "Median tongue redness.", tongue_confidence, ",".join(tongue_steps)),
            self.summary_reading("tongue_b_star", tongue_b, "CIELAB b*", "Median tongue yellow-blue value.", tongue_confidence, ",".join(tongue_steps)),
            self.summary_reading("tongue_lightness", robust_median(self._values.get("tongue_lightness", [])), "CIELAB L*", "Median tongue lightness.", tongue_confidence, ",".join(tongue_steps)),
            self.summary_reading("white_patch_fraction", white_fraction, "fraction", "Median fraction of white patches inside the mouth.", interior_confidence, ",".join(interior_steps)),
            self.summary_reading("gum_minus_skin_a_star", gum_delta, "CIELAB a*", "Median gum redness relative to skin.", gum_confidence, ",".join(gum_steps)),
        ]
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        if calibration < self.threshold_float("min_calibration_confidence"):
            limitations.append("Colour calibration confidence was low; mouth colour findings are less reliable.")
        blue_threshold = self.threshold_float("blue_tongue_b_max")
        pale_threshold = self.threshold_float("pale_tongue_a_max")
        if np.isfinite(tongue_b) and tongue_b <= blue_threshold:
            flags.append(self.make_flag("blue_tongue", "Bluish tongue", FlagSeverity.HIGH, tongue_confidence * 0.8, "The tongue looked bluish, which can be a sign of low blood oxygen (central cyanosis).", tongue_steps, evidence=[f"tongue b* {tongue_b:.1f}"]))
        elif np.isfinite(tongue_a) and tongue_a <= pale_threshold:
            flags.append(self.make_flag("pale_tongue", "Pale tongue", FlagSeverity.MODERATE, self.shortfall_confidence(tongue_a, pale_threshold, tongue_confidence), "The tongue looked pale, which can accompany anaemia.", tongue_steps, evidence=[f"tongue a* {tongue_a:.1f}"]))
        white_threshold = self.threshold_float("white_patch_fraction_flag")
        if np.isfinite(white_fraction) and white_fraction >= white_threshold:
            flags.append(self.make_flag("oral_white_patches", "White patches inside the mouth", FlagSeverity.MODERATE, self.exceedance_confidence(white_fraction, white_threshold, interior_confidence), "White patches were seen inside the mouth, which can occur with oral thrush or other conditions.", interior_steps, evidence=[f"white fraction {white_fraction:.2f}"]))
        gum_threshold = self.threshold_float("gum_redness_delta_a_flag")
        if np.isfinite(gum_delta) and gum_delta >= gum_threshold:
            flags.append(self.make_flag("gum_redness", "Red gums", FlagSeverity.LOW, self.exceedance_confidence(gum_delta, gum_threshold, gum_confidence), "The gums looked redder than expected, which can accompany gum inflammation.", gum_steps, evidence=[f"gum minus skin a* {gum_delta:.1f}"]))
        zero_shot_flags, evidence, zero_shot_metrics = self._classify(limitations)
        existing_codes = {flag.code for flag in flags}
        flags.extend(flag for flag in zero_shot_flags if flag.code not in existing_codes)
        metrics.extend(zero_shot_metrics)
        confidence_values = [value for value in (tongue_confidence, interior_confidence, gum_confidence) if value > 0]
        return self.build_result(
            description=f"Tongue a* {tongue_a:.1f}, b* {tongue_b:.1f}; white patch fraction {white_fraction:.2f}.",
            metrics=metrics,
            flags=flags,
            confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
            value=tongue_a,
            unit="CIELAB a*",
            evidence=evidence,
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        for buffer in self._crops.values():
            buffer.clear()
        self._values.clear()
        self._calibration.clear()
