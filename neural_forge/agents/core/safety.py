"""Safety agent: rule-based post-check that keeps every output an educational screening flag."""

import re
from typing import Any

from neural_forge.core.exceptions import ConfigurationError
from neural_forge.schemas.findings import ConsultantOpinion, ReasoningResult, ReferralUrgency, SessionFindings, max_urgency
from neural_forge.schemas.protocols import localize, localize_list
from neural_forge.schemas.readings import ScreeningFlag
from neural_forge.schemas.reports import RedFlagAlert, SafeInterpretation


class SafetyAgent:
    """Removes definitive diagnostic language, enforces confidence statements, red flags and the disclaimer."""

    def __init__(self, rules: dict[str, Any]) -> None:
        for required_key in (
            "disclaimer",
            "definitive_language",
            "red_flags",
            "condition_name_prefix",
            "hedging_prefixes",
            "confidence_statement",
            "summary_confidence_statement",
            "standard_limitations",
        ):
            if required_key not in rules:
                raise ConfigurationError(f"safety_rules.yaml is missing '{required_key}'")
        self._rules = rules
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        for item in rules["definitive_language"]:
            try:
                self._patterns.append((re.compile(str(item["pattern"]), re.IGNORECASE), str(item["replacement"])))
            except re.error as error:
                raise ConfigurationError(f"Invalid safety pattern '{item.get('pattern')}': {error}") from error
        self._red_flags: list[dict[str, Any]] = list(rules["red_flags"])
        self._hedging_prefixes = [str(prefix).lower() for prefix in rules["hedging_prefixes"]]

    def disclaimer(self, language: str) -> str:
        """Educational disclaimer in the requested language."""
        return localize(self._rules["disclaimer"], language) or ""

    def standard_limitations(self, language: str) -> list[str]:
        """Limitations that apply to every webcam screening."""
        return localize_list(self._rules["standard_limitations"], language)

    def sanitize_text(self, text: str) -> tuple[str, list[str]]:
        """Replace definitive diagnostic phrases; return the new text and a list of changes."""
        changes: list[str] = []
        result = text
        for pattern, replacement in self._patterns:
            result, count = pattern.subn(replacement, result)
            if count:
                changes.append(f"Replaced definitive wording matching '{pattern.pattern}' ({count}x).")
        return result, changes

    def red_flag_alerts(self, flags: list[ScreeningFlag], language: str) -> list[RedFlagAlert]:
        """Alerts for red-flag screening codes present with sufficient confidence."""
        best_confidence: dict[str, float] = {}
        for flag in flags:
            best_confidence[flag.code] = max(best_confidence.get(flag.code, 0.0), flag.confidence)
        alerts: list[RedFlagAlert] = []
        for rule in self._red_flags:
            minimum = float(rule["min_confidence"])
            matched = [code for code in rule["flag_codes"] if best_confidence.get(code, 0.0) >= minimum]
            if matched:
                alerts.append(
                    RedFlagAlert(
                        code=str(rule["code"]),
                        urgency=ReferralUrgency(rule["urgency"]),
                        message=localize(rule["message"], language) or str(rule["code"]),
                    )
                )
        return alerts

    def _ensure_hedged_name(self, name: str, language: str) -> str:
        lowered = name.strip().lower()
        if any(lowered.startswith(prefix) for prefix in self._hedging_prefixes):
            return name.strip()
        prefix = localize(self._rules["condition_name_prefix"], language) or "Possible"
        return f"{prefix} {name.strip()}"

    def _confidence_statement(self, key: str, confidence: float, language: str) -> str:
        template = localize(self._rules[key], language) or "{percent}%"
        return template.format(percent=int(round(confidence * 100)))

    def review(self, result: ReasoningResult, findings: SessionFindings, language: str) -> SafeInterpretation:
        """Apply all safety rules to a reasoning result."""
        output = result.output
        modifications: list[str] = []
        summary, changes = self.sanitize_text(output.findings_summary)
        modifications.extend(changes)
        summary_statement = self._confidence_statement("summary_confidence_statement", output.confidence, language)
        if summary_statement not in summary:
            summary = f"{summary.rstrip()} {summary_statement}"
            modifications.append("Added overall confidence statement.")
        conditions = []
        for condition in output.possible_conditions:
            name, name_changes = self.sanitize_text(condition.name)
            hedged_name = self._ensure_hedged_name(name, language)
            rationale, rationale_changes = self.sanitize_text(condition.rationale)
            modifications.extend(name_changes + rationale_changes)
            statement = self._confidence_statement("confidence_statement", condition.confidence, language)
            if statement not in rationale:
                rationale = f"{rationale.rstrip()} {statement}".strip()
            if hedged_name != name:
                modifications.append(f"Hedged condition name '{name}'.")
            conditions.append(condition.model_copy(update={"name": hedged_name, "rationale": rationale}))
        next_tests = [self.sanitize_text(test)[0] for test in output.recommended_next_tests]
        consultations: list[ConsultantOpinion] = []
        for opinion in result.consultations:
            observation, observation_changes = self.sanitize_text(opinion.observation)
            finding, finding_changes = self.sanitize_text(opinion.likely_finding)
            modifications.extend(observation_changes + finding_changes)
            consultations.append(opinion.model_copy(update={"observation": observation, "likely_finding": finding}))
        alerts = self.red_flag_alerts(findings.flags, language)
        urgency = output.referral_urgency
        for alert in alerts:
            if max_urgency(urgency, alert.urgency) != urgency:
                modifications.append(f"Referral urgency raised to '{alert.urgency.value}' by red flag '{alert.code}'.")
            urgency = max_urgency(urgency, alert.urgency)
        if alerts:
            alert_text = " ".join(alert.message for alert in alerts)
            summary = f"{alert_text} {summary}"
        return SafeInterpretation(
            findings_summary=summary,
            possible_conditions=conditions,
            confidence=output.confidence,
            recommended_next_tests=next_tests,
            repeat_steps=output.repeat_steps,
            referral_urgency=urgency,
            red_flag_alerts=alerts,
            consultations=consultations,
            source=result.source,
            safety_modifications=modifications,
            disclaimer=self.disclaimer(language),
        )

    def review_answer(self, answer: str, confidence: float, language: str) -> tuple[str, list[str]]:
        """Sanitise a follow-up answer and append its confidence statement."""
        sanitized, changes = self.sanitize_text(answer)
        statement = self._confidence_statement("confidence_statement", confidence, language)
        if statement not in sanitized:
            sanitized = f"{sanitized.rstrip()} {statement}"
        return sanitized, changes
