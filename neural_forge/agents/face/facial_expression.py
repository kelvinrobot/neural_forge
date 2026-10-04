"""Facial expression agent: blendshape activity range, expression achievement and hypomimia screening."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.landmark_service import TrackerName


@dataclass
class ExpressionSample:
    """Blendshape vector of one frame."""

    timestamp: float
    step_elapsed: float
    values: np.ndarray
    blink: float


class FacialExpressionAgent(AreaAgent):
    """Measures how much and how differently the face moves during expression and observation steps."""

    name = "facial_expression"
    area = "face"
    required_trackers = frozenset({TrackerName.FACE})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._blendshape_names: list[str] = [str(name) for name in self.threshold_list("key_blendshapes")]
        self._samples: dict[str, list[ExpressionSample]] = defaultdict(list)

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record the key blendshape vector for one frame."""
        face = frame_context.face
        if face is None or not face.blendshapes:
            return []
        values = np.array([face.blendshape(name) for name in self._blendshape_names], dtype=np.float64)
        blink = (face.blendshape("eyeBlinkLeft") + face.blendshape("eyeBlinkRight")) / 2.0
        step_instance = frame_context.step_instance or ""
        sample = ExpressionSample(
            timestamp=frame_context.timestamp_s,
            step_elapsed=self.step_elapsed(step_instance, frame_context.timestamp_s),
            values=values,
            blink=blink,
        )
        self._samples[step_instance].append(sample)
        activity = float(values.mean())
        return [
            self.make_reading(frame_context, "facial_activity", activity, "score", "Mean activation of key facial expression blendshapes.")
        ]

    def _samples_for(self, step_ids: list[str]) -> list[ExpressionSample]:
        wanted = set(step_ids)
        collected: list[ExpressionSample] = []
        for step_instance, samples in self._samples.items():
            if base_step_id(step_instance) in wanted:
                collected.extend(samples)
        return collected

    def _expression_range(self, samples: list[ExpressionSample]) -> float:
        if len(samples) < 3:
            return float("nan")
        matrix = np.stack([sample.values for sample in samples])
        low = np.percentile(matrix, self.threshold_float("low_percentile"), axis=0)
        high = np.percentile(matrix, self.threshold_float("high_percentile"), axis=0)
        ranges = np.sort(high - low)[::-1]
        top_count = max(1, min(len(ranges), self.threshold_int("expression_range_top_n")))
        return float(ranges[:top_count].mean())

    def _expressions_achieved(self) -> tuple[int, int, dict[str, float]]:
        prompt_step = str(self.threshold("expression_prompt_step"))
        windows: dict[str, dict] = self.threshold("expression_windows")
        target = self.threshold_float("expression_target_activation")
        samples = self._samples_for([prompt_step])
        peaks: dict[str, float] = {}
        achieved = 0
        for expression, window in windows.items():
            start = float(window["start_seconds"])
            end = float(window["end_seconds"])
            names = [str(name) for name in window["blendshapes"]]
            indices = [self._blendshape_names.index(name) for name in names if name in self._blendshape_names]
            window_samples = [sample for sample in samples if start <= sample.step_elapsed < end]
            if not window_samples or not indices:
                peaks[expression] = float("nan")
                continue
            activations = [float(sample.values[indices].mean()) for sample in window_samples]
            peak = float(np.percentile(activations, 90))
            peaks[expression] = peak
            if peak >= target:
                achieved += 1
        return achieved, len(windows), peaks

    def _blink_rate(self, step_ids: list[str]) -> float:
        threshold = self.threshold_float("blink_blendshape_threshold")
        total_blinks = 0
        total_seconds = 0.0
        wanted = set(step_ids)
        for step_instance, samples in self._samples.items():
            if base_step_id(step_instance) not in wanted or len(samples) < 2:
                continue
            blinks = np.array([sample.blink for sample in samples])
            closed = blinks >= threshold
            rising_edges = int(np.sum(closed[1:] & ~closed[:-1]))
            total_blinks += rising_edges
            total_seconds += samples[-1].timestamp - samples[0].timestamp
        if total_seconds <= 0:
            return float("nan")
        return total_blinks / (total_seconds / 60.0)

    def _spontaneous_variability(self, samples: list[ExpressionSample]) -> float:
        if len(samples) < 5:
            return float("nan")
        matrix = np.stack([sample.values for sample in samples])
        return float(np.std(matrix, axis=0).mean())

    def finalize(self) -> AgentResult:
        """Summarise expression range, achieved expressions, spontaneous movement and blink rate."""
        if not self.steps_with_data:
            return self.insufficient_result("No usable face frames were captured for the expression assessment.")
        expression_steps = [str(step_id) for step_id in self.threshold_list("expression_steps")]
        observation_steps = [str(step_id) for step_id in self.threshold_list("observation_steps")]
        expression_samples = self._samples_for(expression_steps)
        observation_samples = self._samples_for(observation_steps)
        expression_confidence = self.data_confidence(expression_steps)
        observation_confidence = self.data_confidence(observation_steps)

        expression_range = self._expression_range(expression_samples)
        achieved, total_expressions, peaks = self._expressions_achieved()
        variability = self._spontaneous_variability(observation_samples)
        blink_rate = self._blink_rate(observation_steps)

        metrics: list[Reading] = [
            self.summary_reading("expression_range", expression_range, "score", "Average range of the most active facial blendshapes during expression steps.", expression_confidence, ",".join(expression_steps)),
            self.summary_reading("expressions_achieved", float(achieved), "count", f"Number of prompted expressions clearly produced (of {total_expressions}).", expression_confidence, str(self.threshold("expression_prompt_step"))),
            self.summary_reading("spontaneous_facial_variability", variability, "score", "Spontaneous facial movement while observed naturally.", observation_confidence, ",".join(observation_steps)),
            self.summary_reading("blink_rate", blink_rate, "blinks/min", "Blink rate during natural observation.", observation_confidence, ",".join(observation_steps)),
        ]
        for expression, peak in peaks.items():
            metrics.append(
                self.summary_reading(f"{expression}_expression_peak", peak, "score", f"Peak activation for the {expression} expression.", expression_confidence, str(self.threshold("expression_prompt_step")))
            )

        flags: list[ScreeningFlag] = []
        range_threshold = self.threshold_float("hypomimia_expression_range_max")
        variability_threshold = self.threshold_float("spontaneous_variability_min")
        blink_threshold = self.threshold_float("blink_rate_low_per_minute")
        minimum_achieved = self.threshold_int("min_expressions_achieved")
        reduced_range = np.isfinite(expression_range) and expression_range < range_threshold
        reduced_spontaneous = (np.isfinite(variability) and variability < variability_threshold) or (
            np.isfinite(blink_rate) and blink_rate < blink_threshold
        )
        poor_differentiation = total_expressions > 0 and achieved < minimum_achieved
        if reduced_range and (reduced_spontaneous or poor_differentiation):
            confidence = self.shortfall_confidence(expression_range, range_threshold, min(expression_confidence, max(observation_confidence, expression_confidence * 0.7)))
            flags.append(
                self.make_flag(
                    "reduced_facial_expression",
                    "Reduced facial expression (masked face pattern)",
                    FlagSeverity.MODERATE,
                    confidence,
                    "Facial expressions were small and spontaneous facial movement or blinking was reduced, a pattern sometimes called hypomimia.",
                    expression_steps + observation_steps,
                    evidence=[
                        f"expression range {expression_range:.2f}",
                        f"spontaneous variability {variability:.3f}",
                        f"blink rate {blink_rate:.1f}/min",
                        f"expressions achieved {achieved}/{total_expressions}",
                    ],
                )
            )
        description = (
            f"Expression range {expression_range:.2f}, {achieved}/{total_expressions} prompted expressions produced, blink rate {blink_rate:.1f} per minute."
            if np.isfinite(expression_range)
            else "Expression range could not be measured."
        )
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=(expression_confidence + observation_confidence) / 2.0 if observation_confidence > 0 else expression_confidence,
            value=expression_range,
            unit="score",
        )

    def _reset_measurements(self) -> None:
        self._samples.clear()
