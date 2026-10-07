"""Pain agent: approximate PSPI-style pain expression score from blendshapes."""

from collections import defaultdict

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.models.neural_forge_face import FaceLandmarks
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.landmark_service import TrackerName


class PainAgent(AreaAgent):
    """Approximates the Prkachin and Solomon Pain Intensity score from facial action blendshapes."""

    name = "pain"
    area = "face"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._scores: dict[str, list[float]] = defaultdict(list)
        self._closure_started_at: float | None = None

    def _mean_blendshapes(self, face: FaceLandmarks, key: str) -> float:
        names = [str(name) for name in self.threshold_list(key)]
        return float(np.mean([face.blendshape(name) for name in names])) if names else 0.0

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Compute a per-frame PSPI approximation."""
        face = frame_context.face
        if face is None or not face.blendshapes:
            return []
        scale = self.threshold_float("intensity_scale")
        brow_lowerer = self._mean_blendshapes(face, "brow_lowerer") * scale
        cheek_raiser = self._mean_blendshapes(face, "cheek_raiser") * scale
        lid_tightener = self._mean_blendshapes(face, "lid_tightener") * scale
        nose_wrinkler = self._mean_blendshapes(face, "nose_wrinkler") * scale
        upper_lip_raiser = self._mean_blendshapes(face, "upper_lip_raiser") * scale
        eye_closure_value = self._mean_blendshapes(face, "eye_closure")
        sustained_closure = 0.0
        if eye_closure_value >= self.threshold_float("eye_closure_threshold"):
            if self._closure_started_at is None:
                self._closure_started_at = frame_context.timestamp_s
            if frame_context.timestamp_s - self._closure_started_at >= self.threshold_float("min_eye_closure_seconds"):
                sustained_closure = 1.0
        else:
            self._closure_started_at = None
        pspi = brow_lowerer + max(cheek_raiser, lid_tightener) + max(nose_wrinkler, upper_lip_raiser) + sustained_closure
        self._scores[frame_context.step_instance or ""].append(float(pspi))
        return [
            self.make_reading(
                frame_context,
                "pspi_score",
                pspi,
                "score (0-16)",
                "Approximate pain expression score from brow lowering, eye squeeze, nose wrinkle and upper-lip raise.",
                details={
                    "brow_lowerer": brow_lowerer,
                    "orbital_tightening": max(cheek_raiser, lid_tightener),
                    "levator_contraction": max(nose_wrinkler, upper_lip_raiser),
                    "eye_closure": sustained_closure,
                },
            )
        ]

    def on_step_start(self, step_instance: str, timestamp: float) -> None:
        """Reset eye-closure tracking at the start of each step."""
        super().on_step_start(step_instance, timestamp)
        self._closure_started_at = None

    def finalize(self) -> AgentResult:
        """Summarise pain expression scores over the observation steps."""
        all_scores = [score for scores in self._scores.values() for score in scores]
        if not all_scores:
            return self.insufficient_result("No usable face frames were captured for the pain expression assessment.")
        scores = np.array(all_scores)
        median_score = float(np.median(scores))
        high_threshold = self.threshold_float("pspi_high_frame_threshold")
        high_fraction = float((scores >= high_threshold).mean())
        percentile_90 = float(np.percentile(scores, 90))
        confidence = self.data_confidence(self.steps_with_data)
        steps = ",".join(self.steps_with_data)
        metrics = [
            self.summary_reading("pspi_median", median_score, "score (0-16)", "Median approximate pain expression score.", confidence, steps),
            self.summary_reading("pspi_p90", percentile_90, "score (0-16)", "90th percentile approximate pain expression score.", confidence, steps),
            self.summary_reading("pspi_high_fraction", high_fraction, "fraction", "Fraction of frames with a high pain expression score.", confidence, steps),
        ]
        for step_instance, step_scores in self._scores.items():
            if step_scores:
                metrics.append(
                    self.summary_reading("pspi_median", float(np.median(step_scores)), "score (0-16)", f"Median pain expression score during {step_instance}.", confidence, step_instance)
                )
        flags: list[ScreeningFlag] = []
        median_threshold = self.threshold_float("pspi_flag_median")
        fraction_threshold = self.threshold_float("pspi_high_fraction_flag")
        if median_score >= median_threshold or high_fraction >= fraction_threshold:
            flag_confidence = max(
                self.exceedance_confidence(median_score, median_threshold, confidence) if median_score >= median_threshold else 0.0,
                self.exceedance_confidence(high_fraction, fraction_threshold, confidence) if high_fraction >= fraction_threshold else 0.0,
            )
            flags.append(
                self.make_flag(
                    "pain_expression",
                    "Facial signs of pain or discomfort",
                    FlagSeverity.MODERATE,
                    flag_confidence,
                    "Brow lowering, eye squeezing and upper-lip raising were frequent, which can accompany pain or discomfort.",
                    self.steps_with_data,
                    evidence=[f"median score {median_score:.2f}", f"high-score fraction {high_fraction:.2f}"],
                )
            )
        return self.build_result(
            description=f"Approximate pain expression score: median {median_score:.1f} of 16.",
            metrics=metrics,
            flags=flags,
            confidence=confidence,
            value=median_score,
            unit="score (0-16)",
        )

    def _reset_measurements(self) -> None:
        self._scores.clear()
        self._closure_started_at = None
