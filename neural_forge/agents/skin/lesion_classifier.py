"""Lesion classifier agent: MedSigLIP zero-shot classification of each candidate skin region."""

from typing import Any

import numpy as np

from neural_forge.agents.base import AgentContext, AreaAgent, CropCandidate
from neural_forge.agents.skin.lesion_finder import LesionFinderAgent
from neural_forge.core.exceptions import ConfigurationError
from neural_forge.core.frame_context import FrameContext
from neural_forge.schemas.readings import AgentResult, ClassificationResult, EvidenceCrop, FlagSeverity, Reading, ScreeningFlag
from neural_forge.services.landmark_service import TrackerName


class LesionClassifierAgent(AreaAgent):
    """Ranks skin-condition prompts for every crop found by the lesion finder."""

    name = "lesion_classifier"
    area = "skin"
    required_trackers: frozenset[TrackerName] = frozenset()

    def __init__(self, context: AgentContext) -> None:
        super().__init__(context)
        self._finder: LesionFinderAgent | None = None
        self._visible_counts: list[int] = []

    def bind_peers(self, peers: dict[str, AreaAgent]) -> None:
        """Locate the lesion finder whose crops this agent classifies."""
        super().bind_peers(peers)
        finder = peers.get("lesion_finder")
        if not isinstance(finder, LesionFinderAgent):
            raise ConfigurationError("The lesion classifier requires the lesion finder agent in the same area")
        self._finder = finder

    def measure_frame(self, frame_context: FrameContext) -> list[Reading]:
        """Record how many candidate regions are visible in the current frame."""
        if self._finder is None:
            return []
        detections = frame_context.annotations.get("lesion_detections")
        count = len(detections) if detections is not None else self._finder.latest_detection_count
        self._visible_counts.append(count)
        return [self.make_reading(frame_context, "regions_visible", float(count), "count", "Candidate skin regions visible for classification.")]

    def _needs_consultation(self, result: ClassificationResult) -> bool:
        return self.needs_consultation(result) or result.margin <= self.threshold_float("consultation_margin_max")

    def finalize(self) -> AgentResult:
        """Classify every candidate crop and raise flags for confidently matched patterns."""
        if self._finder is None:
            return self.insufficient_result("The lesion finder was not available.")
        candidates = self._finder.prepare_candidates()
        if not candidates:
            return self.build_result(
                description="No candidate skin regions required classification.",
                metrics=[],
                flags=[],
                confidence=self.data_confidence(self.steps_with_data),
            )
        limitations: list[str] = []
        task = str(self.threshold("zero_shot_task"))
        results = self.safe_classify([region.crop for region in candidates], task, limitations)
        if not results:
            return self.build_result(
                description="Candidate skin regions could not be classified.",
                metrics=[],
                flags=[],
                confidence=0.0,
                limitations=limitations,
            )
        normal_label = str(self.threshold("normal_label"))
        top_k = self.threshold_int("top_k")
        flag_probability = self.threshold_float("flag_probability")
        label_flags: dict[str, dict[str, Any]] = self.threshold("label_flags")
        question = str(self.threshold("consultation_question"))
        metrics: list[Reading] = []
        evidence: list[EvidenceCrop] = []
        best_by_label: dict[str, tuple[float, list[str], list[str]]] = {}
        for region, result in zip(candidates, results):
            ranked = sorted(result.label_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
            confidence = self.data_confidence([region.step_instance])
            for rank, (label, probability) in enumerate(ranked, start=1):
                metrics.append(
                    self.summary_reading(
                        "lesion_label_probability",
                        probability,
                        "probability",
                        f"{region.region_id} rank {rank}: {label.replace('_', ' ')}.",
                        confidence,
                        region.step_instance,
                        details={"region_id": region.region_id, "label": label, "rank": rank},
                    )
                )
            if result.top_label != normal_label and result.top_score >= flag_probability:
                previous = best_by_label.get(result.top_label, (0.0, [], []))
                best_by_label[result.top_label] = (
                    max(previous[0], result.top_score * confidence),
                    previous[1] + [region.region_id],
                    previous[2] + [region.step_instance],
                )
            doubtful = self._needs_consultation(result)
            candidate = CropCandidate(region.crop, region.sharpness, region.step_instance, region.timestamp, f"{region.label} ({region.region_id})")
            evidence.append(self.make_evidence_crop(candidate.label, candidate, result, doubtful, question if doubtful else None))
        flags: list[ScreeningFlag] = []
        for label, (confidence, region_ids, steps) in best_by_label.items():
            definition = label_flags.get(label)
            if definition is None:
                continue
            flags.append(
                self.make_flag(
                    str(definition["code"]),
                    str(definition["title"]),
                    FlagSeverity(str(definition["severity"])),
                    confidence,
                    f"{len(region_ids)} skin region(s) resembled {label.replace('_', ' ')} on image comparison. Only a clinician can confirm this.",
                    sorted(set(steps)),
                    evidence=region_ids,
                )
            )
        normal_count = sum(1 for result in results if result.top_label == normal_label)
        metrics.append(self.summary_reading("regions_classified", float(len(results)), "count", "Candidate regions classified.", self.data_confidence(self.steps_with_data), ",".join(self.steps_with_data)))
        metrics.append(self.summary_reading("regions_resembling_normal_skin", float(normal_count), "count", "Regions whose best match was normal skin.", self.data_confidence(self.steps_with_data), ",".join(self.steps_with_data)))
        mean_top = float(np.mean([result.top_score for result in results]))
        return self.build_result(
            description=(
                "Skin regions resembled: " + ", ".join(sorted(best_by_label.keys())) + "."
                if best_by_label
                else "Classified skin regions did not confidently match a listed condition."
            ),
            metrics=metrics,
            flags=flags,
            confidence=self.data_confidence(self.steps_with_data) * mean_top,
            value=mean_top,
            unit="probability",
            evidence=evidence,
            limitations=limitations,
        )

    def _reset_measurements(self) -> None:
        self._visible_counts.clear()
