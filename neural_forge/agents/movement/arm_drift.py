"""Arm drift agent: wrist drop and forearm rotation with arms extended and eyes closed."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.agents.movement.coordination import palm_orientation
from neural_forge.core.frame_context import FrameContext
from neural_forge.core.protocol_loader import base_step_id
from neural_forge.schemas.readings import AgentResult, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.geometry import euclidean_distance
from neural_forge.services.landmark_service import TrackerName


@dataclass
class DriftSample:
    """Arm position of one side in one frame."""

    step_elapsed: float
    wrist_drop: float
    orientation: float


@dataclass
class DriftAnalysis:
    """Drift results for one arm."""

    drop: float
    rotation_change: float


class ArmDriftAgent(AreaAgent):
    """Detects one arm drifting down or rotating inward while the eyes are closed."""

    name = "arm_drift"
    area = "movement"
    required_trackers = frozenset({TrackerName.POSE, TrackerName.HANDS})

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._samples: dict[str, dict[str, list[DriftSample]]] = defaultdict(lambda: {"right": [], "left": []})

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record wrist height relative to the shoulder and palm orientation for each arm."""
        pose = frame_context.pose
        if pose is None:
            return []
        minimum_score = self.threshold_float("min_keypoint_score")
        if not pose.all_visible(("left_shoulder", "right_shoulder"), minimum_score):
            return []
        shoulder_width = euclidean_distance(pose.keypoint("left_shoulder"), pose.keypoint("right_shoulder"))
        if shoulder_width < 5.0:
            return []
        step_instance = frame_context.step_instance or ""
        elapsed = self.step_elapsed(step_instance, frame_context.timestamp_s)
        readings: list[Reading] = []
        for side in ("right", "left"):
            if not pose.is_visible(f"{side}_wrist", minimum_score):
                continue
            wrist_drop = float(pose.keypoint(f"{side}_wrist")[1] - pose.keypoint(f"{side}_shoulder")[1]) / shoulder_width
            orientation = float("nan")
            hand = frame_context.hands.get(side) if frame_context.hands is not None else None  # type: ignore[arg-type]
            if hand is not None:
                orientation = palm_orientation(hand)
            self._samples[step_instance][side].append(DriftSample(elapsed, wrist_drop, orientation))
            readings.append(
                self.make_reading(frame_context, "wrist_height_below_shoulder", wrist_drop * self.threshold_float("shoulder_width_cm"), "cm", f"{side.capitalize()} wrist height below the shoulder.", side=side)  # type: ignore[arg-type]
            )
        return readings

    def _analyse(self, samples: list[DriftSample]) -> DriftAnalysis | None:
        minimum_samples = self.threshold_int("min_window_samples")
        if len(samples) < 2 * minimum_samples:
            return None
        elapsed = np.array([sample.step_elapsed for sample in samples])
        drops = np.array([sample.wrist_drop for sample in samples])
        orientations = np.array([sample.orientation for sample in samples])
        start_mask = elapsed <= elapsed.min() + self.threshold_float("start_window_seconds")
        end_mask = elapsed >= elapsed.max() - self.threshold_float("end_window_seconds")
        if start_mask.sum() < minimum_samples or end_mask.sum() < minimum_samples:
            return None
        drop = float(np.median(drops[end_mask]) - np.median(drops[start_mask]))
        start_orientation = orientations[start_mask]
        end_orientation = orientations[end_mask]
        start_orientation = start_orientation[np.isfinite(start_orientation)]
        end_orientation = end_orientation[np.isfinite(end_orientation)]
        rotation_change = float("nan")
        if len(start_orientation) >= minimum_samples and len(end_orientation) >= minimum_samples:
            rotation_change = float(abs(np.median(end_orientation) - np.median(start_orientation)))
        return DriftAnalysis(drop=drop, rotation_change=rotation_change)

    def finalize(self) -> AgentResult:
        """Compare drift and rotation between the arms."""
        drift_steps = [str(step) for step in self.threshold_list("drift_steps")]
        results: dict[str, DriftAnalysis] = {}
        for side in ("right", "left"):
            samples = [sample for instance, sides in self._samples.items() if base_step_id(instance) in drift_steps for sample in sides[side]]
            analysis = self._analyse(samples)
            if analysis is not None:
                results[side] = analysis
        if not results:
            return self.insufficient_result("Arm positions could not be tracked long enough to assess drift.")
        confidence = self.data_confidence(drift_steps)
        steps = ",".join(drift_steps)
        shoulder_cm = self.threshold_float("shoulder_width_cm")
        metrics: list[Reading] = []
        for side, analysis in results.items():
            metrics.append(self.summary_reading("wrist_drift", analysis.drop * shoulder_cm, "cm", f"{side.capitalize()} wrist downward drift from start to end.", confidence, steps, side=side))  # type: ignore[arg-type]
            metrics.append(self.summary_reading("forearm_rotation_change", analysis.rotation_change, "sine", f"{side.capitalize()} forearm rotation change.", confidence, steps, side=side))  # type: ignore[arg-type]
        flags: list[ScreeningFlag] = []
        drop_threshold = self.threshold_float("drift_drop_shoulder_fraction")
        asymmetry_threshold = self.threshold_float("drift_asymmetry_shoulder_fraction")
        rotation_threshold = self.threshold_float("pronation_change_threshold")
        right_drop = results["right"].drop if "right" in results else float("nan")
        left_drop = results["left"].drop if "left" in results else float("nan")
        drifting = [side for side, analysis in results.items() if analysis.drop >= drop_threshold]
        if drifting:
            if len(results) == 2 and abs(right_drop - left_drop) >= asymmetry_threshold:
                worse = "right" if right_drop > left_drop else "left"
                flags.append(
                    self.make_flag(
                        "arm_drift",
                        "One arm drifted downward",
                        FlagSeverity.HIGH,
                        self.exceedance_confidence(results[worse].drop, drop_threshold, confidence),
                        f"With eyes closed, the {worse} arm drifted down more than the other, which can be a sign of weakness on that side.",
                        drift_steps,
                        evidence=[f"right drift {right_drop * shoulder_cm:.1f} cm", f"left drift {left_drop * shoulder_cm:.1f} cm"],
                        side=worse,  # type: ignore[arg-type]
                    )
                )
            elif len(results) == 2:
                flags.append(
                    self.make_flag(
                        "bilateral_arm_drift",
                        "Both arms drifted downward",
                        FlagSeverity.LOW,
                        confidence * 0.5,
                        "Both arms drifted down by a similar amount, which is often due to fatigue.",
                        drift_steps,
                        evidence=[f"right drift {right_drop * shoulder_cm:.1f} cm", f"left drift {left_drop * shoulder_cm:.1f} cm"],
                        side="both",
                    )
                )
            else:
                only_side = drifting[0]
                flags.append(
                    self.make_flag(
                        "arm_drift",
                        "One arm drifted downward",
                        FlagSeverity.HIGH,
                        self.exceedance_confidence(results[only_side].drop, drop_threshold, confidence) * 0.7,
                        f"With eyes closed, the {only_side} arm drifted down; the other arm could not be compared.",
                        drift_steps,
                        evidence=[f"{only_side} drift {results[only_side].drop * shoulder_cm:.1f} cm"],
                        side=only_side,  # type: ignore[arg-type]
                    )
                )
        rotating = [
            side for side, analysis in results.items() if np.isfinite(analysis.rotation_change) and analysis.rotation_change >= rotation_threshold
        ]
        if rotating and len(rotating) == 1:
            side = rotating[0]
            flags.append(
                self.make_flag(
                    "forearm_pronation_drift",
                    "Forearm turned inward",
                    FlagSeverity.MODERATE,
                    self.exceedance_confidence(results[side].rotation_change, rotation_threshold, confidence),
                    f"The {side} forearm rotated palm-down while the eyes were closed (pronator drift).",
                    drift_steps,
                    evidence=[f"rotation change {results[side].rotation_change:.2f}"],
                    side=side,  # type: ignore[arg-type]
                )
            )
        description = "; ".join(f"{side} arm drift {analysis.drop * shoulder_cm:.1f} cm" for side, analysis in results.items()) + "."
        return self.build_result(
            description=description,
            metrics=metrics,
            flags=flags,
            confidence=confidence,
            value=float(np.nanmax([right_drop, left_drop])) * shoulder_cm if np.isfinite(right_drop) or np.isfinite(left_drop) else None,
            unit="cm",
        )

    def _reset_measurements(self) -> None:
        self._samples.clear()
