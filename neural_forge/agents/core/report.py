"""Report agent: final structured report and plain-English summary per area."""

from datetime import datetime, timezone

from neural_forge.agents.core.safety import SafetyAgent
from neural_forge.core.protocol_loader import ConfigRepository
from neural_forge.schemas.findings import SessionFindings, StepStatus
from neural_forge.schemas.protocols import AreaProtocol, localize
from neural_forge.schemas.reports import FinalReport, FindingReportEntry, SafeInterpretation, StepReportEntry


class ReportAgent:
    """Assembles the final report from findings and the safety-reviewed interpretation."""

    def __init__(self, config: ConfigRepository, safety: SafetyAgent, key_metrics_per_agent: int = 12) -> None:
        self._config = config
        self._safety = safety
        self._key_metrics_per_agent = key_metrics_per_agent

    def build(
        self,
        findings: SessionFindings,
        interpretation: SafeInterpretation,
        protocol: AreaProtocol,
        language: str,
    ) -> FinalReport:
        """Build the final report."""
        messages = self._config.ui_messages()
        area_title = localize(protocol.title, language) or protocol.area.value
        steps = [
            StepReportEntry(
                step_number=record.step_number,
                step_id=record.step_id,
                step_instance=record.step_instance,
                title=record.title,
                status=record.status,
                attempts=record.attempts,
                reason=record.reason,
            )
            for record in findings.steps
        ]
        completed = sum(1 for record in findings.steps if record.status == StepStatus.COMPLETED)
        repeated = sum(1 for record in findings.steps if record.status == StepStatus.REPEATED)
        skipped = sum(1 for record in findings.steps if record.status == StepStatus.SKIPPED)
        incomplete = sum(1 for record in findings.steps if record.status in (StepStatus.INCOMPLETE, StepStatus.NOT_REACHED))
        finding_entries = [
            FindingReportEntry(
                agent=agent_findings.agent,
                description=self._safety.sanitize_text(agent_findings.result.description)[0],
                confidence=agent_findings.result.confidence,
                quality_flag=agent_findings.result.quality_flag,
                key_metrics=agent_findings.result.metrics[: self._key_metrics_per_agent],
                flags=agent_findings.result.flags,
                limitations=agent_findings.result.limitations,
            )
            for agent_findings in findings.agents
        ]
        quality = findings.quality
        quality_notes = [
            messages.text(
                "report_quality_frames",
                language,
                usable=quality.frames_usable,
                processed=quality.frames_processed,
                percent=int(round(quality.usable_fraction * 100)),
            ),
            messages.text("report_quality_fps", language, fps=round(quality.measured_fps, 1), dropped=quality.frames_dropped),
            messages.text(
                "report_quality_calibration",
                language,
                method=quality.calibration_method.replace("_", " "),
                percent=int(round(quality.calibration_confidence * 100)),
            ),
        ]
        if quality.common_issues:
            issue_list = ", ".join(f"{issue.replace('_', ' ')} ({count})" for issue, count in quality.common_issues.items())
            quality_notes.append(messages.text("report_quality_issues", language, issues=issue_list))
        limitations: list[str] = list(self._safety.standard_limitations(language))
        for record in findings.steps:
            if record.status in (StepStatus.INCOMPLETE, StepStatus.NOT_REACHED):
                limitations.append(
                    messages.text("report_step_not_completed", language, title=record.title, reason=record.reason or "-")
                )
        for agent_findings in findings.agents:
            for limitation in agent_findings.result.limitations:
                if limitation not in limitations:
                    limitations.append(limitation)
        summary_parts = [
            messages.text(
                "report_intro",
                language,
                area=area_title,
                completed=completed + repeated,
                total=len(findings.steps),
                skipped=skipped,
            )
        ]
        if findings.flags:
            flag_list = "; ".join(
                f"{flag.title} ({int(round(flag.confidence * 100))}%)" for flag in findings.flags
            )
            summary_parts.append(messages.text("report_flags_intro", language, flags=flag_list))
        else:
            summary_parts.append(messages.text("report_no_flags", language))
        summary_parts.append(interpretation.findings_summary)
        summary_parts.append(messages.text(f"report_referral_{interpretation.referral_urgency.value}", language))
        summary_parts.append(interpretation.disclaimer)
        plain_summary = " ".join(part.strip() for part in summary_parts if part and part.strip())
        return FinalReport(
            session_id=findings.session_id,
            area=findings.area,
            area_title=area_title,
            language=language,
            generated_at=datetime.now(timezone.utc),
            patient_context=findings.patient_context,
            steps=steps,
            steps_completed=completed,
            steps_repeated=repeated,
            steps_skipped=skipped,
            steps_incomplete=incomplete,
            stopped_early=findings.stopped_early,
            findings=finding_entries,
            screening_flags=findings.flags,
            interpretation=interpretation,
            plain_english_summary=plain_summary,
            quality_notes=quality_notes,
            limitations=limitations,
            evidence_crops=findings.evidence,
            disclaimer=interpretation.disclaimer,
        )
