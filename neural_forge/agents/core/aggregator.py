"""Aggregator agent: robust statistics per agent, step and metric, and the session findings JSON."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from neural_forge.schemas.findings import (
    AgentFindings,
    MetricStatistics,
    PatientContext,
    QualitySummary,
    SessionFindings,
    StepRecord,
    StepStatus,
)
from neural_forge.schemas.readings import AgentResult, QualityAssessment, QualityFlag, Reading

ReadingKey = tuple[str, str, str, str | None, str]


class AggregatorAgent:
    """Collects readings for one session and builds the structured findings."""

    def __init__(self, session_id: str, area: str, aggregation_config: dict[str, Any], screening_notice: str) -> None:
        self._session_id = session_id
        self._area = area
        self._min_confidence = float(aggregation_config["min_reading_confidence"])
        self._excluded_flags = {QualityFlag(value) for value in aggregation_config["excluded_quality_flags"]}
        self._screening_notice = screening_notice
        self._accepted: dict[ReadingKey, list[Reading]] = defaultdict(list)
        self._discarded: Counter[ReadingKey] = Counter()
        self._frames_assessed = 0
        self._frames_usable = 0
        self._brightness_sum = 0.0
        self._blur_sum = 0.0
        self._issue_counts: Counter[str] = Counter()

    def add_readings(self, readings: list[Reading]) -> None:
        """Store readings, discarding low-quality or low-confidence ones."""
        for reading in readings:
            key: ReadingKey = (reading.agent, reading.metric, reading.protocol_step, reading.side, reading.unit)
            if (
                reading.value is None
                or reading.quality_flag in self._excluded_flags
                or reading.confidence < self._min_confidence
            ):
                self._discarded[key] += 1
                continue
            self._accepted[key].append(reading)

    def record_frame_quality(self, quality: QualityAssessment) -> None:
        """Accumulate frame quality statistics."""
        self._frames_assessed += 1
        if quality.usable:
            self._frames_usable += 1
        self._brightness_sum += quality.brightness
        self._blur_sum += quality.blur_variance
        self._issue_counts.update(quality.issues)

    def metric_statistics(self) -> list[MetricStatistics]:
        """Median and interquartile range for every agent, metric, step and side."""
        statistics: list[MetricStatistics] = []
        all_keys = set(self._accepted.keys()) | set(self._discarded.keys())
        for key in sorted(all_keys, key=lambda item: (item[0], item[2], item[1], item[3] or "")):
            agent, metric, protocol_step, side, unit = key
            readings = self._accepted.get(key, [])
            if not readings:
                continue
            values = np.array([reading.value for reading in readings if reading.value is not None], dtype=np.float64)
            quartiles = np.percentile(values, [25, 50, 75])
            statistics.append(
                MetricStatistics(
                    agent=agent,
                    metric=metric,
                    unit=unit,
                    protocol_step=protocol_step,
                    side=side,
                    median=float(quartiles[1]),
                    q1=float(quartiles[0]),
                    q3=float(quartiles[2]),
                    iqr=float(quartiles[2] - quartiles[0]),
                    minimum=float(values.min()),
                    maximum=float(values.max()),
                    count=int(len(values)),
                    discarded=int(self._discarded.get(key, 0)),
                    mean_confidence=float(np.mean([reading.confidence for reading in readings])),
                )
            )
        return statistics

    def quality_counts(self) -> tuple[int, int, float, float, dict[str, int]]:
        """Frames assessed, frames usable, mean brightness, mean blur and the most common issues."""
        assessed = max(1, self._frames_assessed)
        common_issues = dict(self._issue_counts.most_common(8))
        return (
            self._frames_assessed,
            self._frames_usable,
            self._brightness_sum / assessed,
            self._blur_sum / assessed,
            common_issues,
        )

    def build_findings(
        self,
        patient_context: PatientContext,
        step_records: list[StepRecord],
        agent_results: list[AgentResult],
        quality_summary: QualitySummary,
        stopped_early: bool,
    ) -> SessionFindings:
        """Build the structured findings JSON for the session."""
        statistics = self.metric_statistics()
        statistics_by_agent: dict[str, list[MetricStatistics]] = defaultdict(list)
        for item in statistics:
            statistics_by_agent[item.agent].append(item)
        agent_findings = [
            AgentFindings(agent=result.agent, result=result, statistics=statistics_by_agent.get(result.agent, []))
            for result in agent_results
        ]
        flags = [flag for result in agent_results for flag in result.flags]
        evidence = [crop for result in agent_results for crop in result.evidence_crops]
        return SessionFindings(
            session_id=self._session_id,
            area=self._area,
            generated_at=datetime.now(timezone.utc),
            patient_context=patient_context,
            steps=step_records,
            agents=agent_findings,
            flags=flags,
            evidence=evidence,
            overall_confidence=self._overall_confidence(agent_results, step_records),
            quality=quality_summary,
            stopped_early=stopped_early,
            screening_notice=self._screening_notice,
        )

    @staticmethod
    def _overall_confidence(agent_results: list[AgentResult], step_records: list[StepRecord]) -> float:
        contributing = [result.confidence for result in agent_results if result.status == "ok"]
        if not contributing:
            return 0.0
        agent_confidence = float(np.mean(contributing))
        measurable_steps = [record for record in step_records if record.status != StepStatus.SKIPPED]
        if not measurable_steps:
            return 0.0
        completed = [record for record in measurable_steps if record.status in (StepStatus.COMPLETED, StepStatus.REPEATED)]
        completion_fraction = len(completed) / len(measurable_steps)
        return float(min(1.0, max(0.0, agent_confidence * (0.5 + 0.5 * completion_fraction))))
