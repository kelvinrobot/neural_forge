"""Capillary refill agent: nail-bed colour return timing after pressure release."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.colour_utils import masked_lab_statistics
from neural_forge.services.landmark_service import TrackerName
from neural_forge.services.regions import nail_bed_mask

FingerKey = tuple[str, str]


@dataclass
class RefillResult:
    """Capillary refill measurement for the pressed finger."""

    finger: FingerKey | None
    baseline_a: float
    blanched_a: float
    refill_seconds: float
    recovered: bool
    observed_seconds: float


class CapillaryRefillAgent(AreaAgent):
    """Tracks each visible nail bed over time and times the colour return of the blanched one."""

    name = "capillary_refill"
    area = "hands"
    required_trackers = frozenset({TrackerName.HANDS})
    supported_states = frozenset({"colour_returned_to_baseline"})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._baseline: dict[FingerKey, list[float]] = defaultdict(list)
        self._release: dict[str, dict[FingerKey, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))

    def on_step_start(self, step_instance: str, timestamp: float) -> None:
        """Reset the release series when a release step (re)starts."""
        super().on_step_start(step_instance, timestamp)
        if base_step_id(step_instance) in [str(step) for step in self.threshold_list("release_steps")]:
            self._release[step_instance] = defaultdict(list)

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Measure nail-bed redness of every visible fingertip."""
        hands = frame_context.hands
        if hands is None or hands.count == 0:
            return []
        step_instance = frame_context.step_instance or ""
        step_id = base_step_id(step_instance)
        elapsed = self.step_elapsed(step_instance, frame_context.timestamp_s)
        lab = frame_context.lab()
        shape = lab.shape[:2]
        is_press = step_id in [str(step) for step in self.threshold_list("press_steps")]
        is_release = step_id in [str(step) for step in self.threshold_list("release_steps")]
        baseline_window = self.threshold_float("baseline_window_seconds")
        if is_press and elapsed > baseline_window:
            return [self.make_reading(frame_context, "pressing_phase", 1.0, "flag", "Pressure applied to the nail.")]
        readings: list[Reading] = []
        for hand in hands.hands:
            for finger in [str(name) for name in self.threshold_list("fingers")]:
                region = nail_bed_mask(hand, finger, shape, self.threshold_float("nail_roi_radius_fraction"))
                statistics = masked_lab_statistics(lab, region, self.threshold_int("min_region_pixels"))
                if statistics is None:
                    continue
                key: FingerKey = (hand.side, finger)
                if is_press and elapsed <= baseline_window:
                    self._baseline[key].append(statistics.red_green)
                elif is_release:
                    self._release[step_instance][key].append((elapsed, statistics.red_green))
                else:
                    continue
                readings.append(
                    self.make_reading(
                        frame_context,
                        "nail_bed_a_star",
                        statistics.red_green,
                        "CIELAB a*",
                        f"{hand.side.capitalize()} {finger} nail-bed redness.",
                        confidence=frame_context.calibration_confidence,
                        side=hand.side,
                        details={"finger": finger, "step_elapsed": elapsed},
                    )
                )
        return readings

    def _smoothed(self, values: np.ndarray) -> np.ndarray:
        window = max(1, self.threshold_int("smoothing_samples"))
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        padded = np.concatenate([np.full(window - 1, values[0]), values])
        return np.convolve(padded, kernel, mode="valid")

    def _measure(self, step_instance: str) -> RefillResult:
        series_by_finger = self._release.get(step_instance, {})
        blanch_window = self.threshold_float("blanch_window_seconds")
        minimum_blanch = self.threshold_float("min_blanch_delta_a")
        best_key: FingerKey | None = None
        best_drop = 0.0
        observed = 0.0
        for key, series in series_by_finger.items():
            if not series or not self._baseline.get(key):
                continue
            observed = max(observed, series[-1][0] - series[0][0])
            early = [value for elapsed, value in series if elapsed - series[0][0] <= blanch_window]
            if not early:
                continue
            drop = float(np.median(self._baseline[key])) - float(min(early))
            if drop > best_drop:
                best_drop = drop
                best_key = key
        if best_key is None or best_drop < minimum_blanch:
            return RefillResult(finger=best_key, baseline_a=float("nan"), blanched_a=float("nan"), refill_seconds=float("nan"), recovered=False, observed_seconds=observed)
        series = series_by_finger[best_key]
        times = np.array([item[0] for item in series]) - series[0][0]
        values = self._smoothed(np.array([item[1] for item in series]))
        baseline = float(np.median(self._baseline[best_key]))
        early_mask = times <= blanch_window
        blanched = float(values[early_mask].min()) if early_mask.any() else float(values[0])
        target = blanched + self.threshold_float("recovery_fraction") * (baseline - blanched)
        sustain = max(1, self.threshold_int("sustain_samples"))
        above = values >= target
        for index in range(len(values) - sustain + 1):
            if times[index] > blanch_window * 0.5 and above[index : index + sustain].all():
                return RefillResult(best_key, baseline, blanched, float(times[index]), True, float(times[-1]))
        return RefillResult(best_key, baseline, blanched, float("nan"), False, float(times[-1]))

    def state_reached(self, step_instance: str, state: str) -> bool:
        """Whether the blanched nail bed has returned to its baseline colour."""
        if state != "colour_returned_to_baseline":
            return False
        if base_step_id(step_instance) not in [str(step) for step in self.threshold_list("release_steps")]:
            return False
        return self._measure(step_instance).recovered

    def insufficiency_reason(self, step_instance: str) -> str | None:
        """Explain missing blanching or missing baseline for the repeat guidance."""
        if base_step_id(step_instance) in [str(step) for step in self.threshold_list("release_steps")]:
            if not self._baseline:
                return "The fingernail colour before pressing was not captured; keep the nail visible before you press."
            result = self._measure(step_instance)
            if result.finger is None or not np.isfinite(result.baseline_a):
                return "No clear whitening of the nail was seen after release; press a little harder and keep the nail facing the camera."
        return super().insufficiency_reason(step_instance)

    def finalize(self) -> AgentResult:
        """Report the capillary refill time and flag it above the threshold."""
        release_steps = [str(step) for step in self.threshold_list("release_steps")]
        press_steps = [str(step) for step in self.threshold_list("press_steps")]
        release_instances = [instance for instance in self._release if base_step_id(instance) in release_steps]
        if not release_instances or not self._baseline:
            return self.insufficient_result("The press-and-release sequence could not be measured.")
        latest_instance = release_instances[-1]
        result = self._measure(latest_instance)
        confidence = self.data_confidence(press_steps + release_steps)
        metrics: list[Reading] = []
        flags: list[ScreeningFlag] = []
        limitations: list[str] = []
        threshold = self.threshold_float("crt_threshold_seconds")
        finger_label = f"{result.finger[0]} {result.finger[1]}" if result.finger else "unknown"
        metrics.append(self.summary_reading("nail_baseline_a_star", result.baseline_a, "CIELAB a*", f"Nail-bed redness before pressing ({finger_label}).", confidence, ",".join(press_steps)))
        metrics.append(self.summary_reading("nail_blanched_a_star", result.blanched_a, "CIELAB a*", "Nail-bed redness immediately after release.", confidence, latest_instance))
        metrics.append(self.summary_reading("capillary_refill_time", result.refill_seconds if result.recovered else result.observed_seconds, "s", "Seconds until nail colour returned to baseline (lower bound if it did not return).", confidence, latest_instance, details={"recovered": result.recovered}))
        if not np.isfinite(result.baseline_a):
            limitations.append("No clear nail blanching was detected, so the refill time could not be measured.")
            return self.build_result(
                description="Capillary refill could not be timed because no clear blanching was seen.",
                metrics=metrics,
                flags=flags,
                confidence=confidence * 0.3,
                limitations=limitations,
            )
        if result.recovered and result.refill_seconds > threshold:
            flags.append(
                self.make_flag(
                    "prolonged_capillary_refill",
                    "Slow capillary refill",
                    FlagSeverity.MODERATE,
                    self.exceedance_confidence(result.refill_seconds, threshold, confidence),
                    f"Nail colour took {result.refill_seconds:.1f} seconds to return, longer than the screening reference of {threshold:.0f} seconds.",
                    press_steps + [latest_instance],
                    evidence=[f"refill {result.refill_seconds:.2f} s on {finger_label}"],
                )
            )
        elif not result.recovered and result.observed_seconds > threshold:
            flags.append(
                self.make_flag(
                    "prolonged_capillary_refill",
                    "Slow capillary refill",
                    FlagSeverity.MODERATE,
                    confidence * 0.7,
                    f"Nail colour had not returned to baseline after {result.observed_seconds:.1f} seconds of observation.",
                    press_steps + [latest_instance],
                    evidence=[f"not recovered after {result.observed_seconds:.1f} s on {finger_label}"],
                )
            )
        description = (
            f"Capillary refill time {result.refill_seconds:.1f} s ({finger_label})."
            if result.recovered
            else f"Nail colour did not return to baseline within {result.observed_seconds:.1f} s."
        )
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=confidence,
            value=result.refill_seconds if result.recovered else result.observed_seconds,
            unit="s",
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._baseline.clear()
        self._release.clear()
