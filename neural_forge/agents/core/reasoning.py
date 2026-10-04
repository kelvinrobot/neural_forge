"""Reasoning agent: Qwen3-VL-4B combines threshold-based findings into a strict JSON interpretation."""

import json
import logging
from dataclasses import dataclass
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from neural_forge.agents.core.medical_consultant import MedicalConsultantAgent
from neural_forge.core.exceptions import LLMResponseValidationError, LLMServiceError
from neural_forge.core.logging import get_logger, log_event
from neural_forge.core.protocol_loader import ConfigRepository, base_step_id
from neural_forge.core.settings import Settings
from neural_forge.models.llm_clients import extract_json_text, image_part, load_prompt, text_part
from neural_forge.models.registry import ModelRegistry
from neural_forge.schemas.findings import (
    ConsultantOpinion,
    ConsultationPlan,
    ConsultationRequest,
    PossibleCondition,
    QuestionAnswerOutput,
    ReasoningOutput,
    ReasoningResult,
    ReferralUrgency,
    SessionFindings,
    StepStatus,
    max_urgency,
)
from neural_forge.schemas.protocols import AreaProtocol
from neural_forge.schemas.readings import EvidenceCrop
from neural_forge.schemas.reports import FinalReport
from neural_forge.schemas.sessions import QAEntry

logger = get_logger(__name__)
ModelType = TypeVar("ModelType", bound=BaseModel)


@dataclass(frozen=True)
class RuleMatch:
    """A sign-to-condition rule satisfied by the threshold-based flags."""

    rule_id: str
    condition: str
    supporting_signs: list[str]
    confidence: float
    urgency: ReferralUrgency
    next_tests: list[str]
    explanation: str


class ReasoningAgent:
    """Coordinator that explains and combines findings; Python thresholds decide what is abnormal."""

    def __init__(
        self,
        settings: Settings,
        config: ConfigRepository,
        registry: ModelRegistry,
        consultant: MedicalConsultantAgent,
    ) -> None:
        self._settings = settings
        self._config = config
        self._registry = registry
        self._consultant = consultant
        self._system_template = load_prompt(settings.prompts_dir, "reasoning_system.md")
        self._repair_template = load_prompt(settings.prompts_dir, "json_repair.md")
        self._question_template = load_prompt(settings.prompts_dir, "question_system.md")
        self._plan_template = load_prompt(settings.prompts_dir, "consultation_plan.md")
        rules_config = config.sign_condition_rules()
        self._rules: list[dict[str, Any]] = list(rules_config["rules"])
        self._severity_urgency: dict[str, str] = dict(rules_config["flag_severity_urgency"])
        self._reference_ranges: dict[str, Any] = dict(config.reference_ranges()["areas"])

    def match_rules(self, findings: SessionFindings) -> list[RuleMatch]:
        """Return the rules whose required flags are present in the findings."""
        flags_by_code = {}
        for flag in findings.flags:
            existing = flags_by_code.get(flag.code)
            if existing is None or flag.confidence > existing.confidence:
                flags_by_code[flag.code] = flag
        matches: list[RuleMatch] = []
        for rule in self._rules:
            areas = rule.get("areas", [])
            if areas and findings.area not in areas:
                continue
            all_of = list(rule.get("all_of", []))
            any_of = list(rule.get("any_of", []))
            if not all(code in flags_by_code for code in all_of):
                continue
            matched_any = [code for code in any_of if code in flags_by_code]
            if any_of and not matched_any:
                continue
            signs = all_of + matched_any
            if not signs:
                continue
            confidence = min(flags_by_code[code].confidence for code in signs) * float(rule.get("confidence_weight", 1.0))
            matches.append(
                RuleMatch(
                    rule_id=str(rule["id"]),
                    condition=str(rule["condition"]),
                    supporting_signs=signs,
                    confidence=float(min(1.0, max(0.0, confidence))),
                    urgency=ReferralUrgency(rule.get("referral_urgency", "routine")),
                    next_tests=list(rule.get("next_tests", [])),
                    explanation=str(rule.get("explanation", "")),
                )
            )
        return matches

    def urgency_floor(self, findings: SessionFindings, matches: list[RuleMatch]) -> ReferralUrgency:
        """Minimum referral urgency decided by Python from flag severities and matched rules."""
        urgency = ReferralUrgency.NONE
        for flag in findings.flags:
            urgency = max_urgency(urgency, ReferralUrgency(self._severity_urgency.get(flag.severity.value, "none")))
        for match in matches:
            urgency = max_urgency(urgency, match.urgency)
        return urgency

    def build_system_prompt(self, area: str) -> str:
        """Render the system prompt with area reference ranges, rules and the output schema."""
        area_ranges = self._reference_ranges.get(area, [])
        area_rules = [
            {
                "id": rule["id"],
                "condition": rule["condition"],
                "all_of": rule.get("all_of", []),
                "any_of": rule.get("any_of", []),
                "referral_urgency": rule.get("referral_urgency", "routine"),
                "explanation": rule.get("explanation", ""),
            }
            for rule in self._rules
            if not rule.get("areas") or area in rule["areas"]
        ]
        return (
            self._system_template.replace("{{AREA}}", area)
            .replace("{{REFERENCE_RANGES}}", yaml.safe_dump(area_ranges, sort_keys=False, allow_unicode=True))
            .replace("{{SIGN_CONDITION_RULES}}", yaml.safe_dump(area_rules, sort_keys=False, allow_unicode=True))
            .replace("{{OUTPUT_SCHEMA}}", json.dumps(ReasoningOutput.model_json_schema(), indent=2))
        )

    async def interpret(self, findings: SessionFindings, protocol: AreaProtocol, rule_summary_texts: dict[str, str]) -> ReasoningResult:
        """Produce the validated interpretation, falling back to rule-based reasoning if the LLM fails."""
        warnings: list[str] = []
        matches = self.match_rules(findings)
        floor = self.urgency_floor(findings, matches)
        consultations = await self._run_consultations(findings.evidence, warnings)
        payload = {
            "findings": findings.to_llm_payload(),
            "rule_matched_candidates": [
                {
                    "rule_id": match.rule_id,
                    "condition": match.condition,
                    "supporting_signs": match.supporting_signs,
                    "confidence": round(match.confidence, 3),
                    "referral_urgency": match.urgency.value,
                }
                for match in matches
            ],
            "python_minimum_referral_urgency": floor.value,
            "consultant_opinions": [opinion.model_dump() for opinion in consultations],
            "valid_step_ids": [step.id for step in protocol.steps],
        }
        content: list[dict[str, Any]] = [text_part(json.dumps(payload, indent=1, default=str))]
        doubtful_crop = self._most_doubtful_crop(findings.evidence)
        if self._settings.reasoning_attach_image and doubtful_crop is not None:
            content.insert(0, image_part(doubtful_crop.image_jpeg_base64))
            content.append(text_part(f"The attached image is evidence crop '{doubtful_crop.crop_id}' ({doubtful_crop.label})."))
        messages = [
            {"role": "system", "content": self.build_system_prompt(findings.area)},
            {"role": "user", "content": content},
        ]
        source = "llm"
        try:
            output = await self._request_structured(messages, ReasoningOutput, self._settings.reasoning_max_tokens)
        except (LLMServiceError, LLMResponseValidationError) as error:
            log_event(logger, logging.WARNING, "reasoning_fallback", session_id=findings.session_id, error=error.message)
            warnings.append(f"Reasoning model unavailable; rule-based interpretation used ({error.message}).")
            output = self._rule_based_output(findings, matches, floor, rule_summary_texts)
            source = "rule_based"
        output = self._enforce_constraints(output, findings, matches, floor, protocol, warnings)
        return ReasoningResult(
            output=output,
            source=source,  # type: ignore[arg-type]
            consultations=consultations,
            rule_matches=[match.rule_id for match in matches],
            warnings=warnings,
        )

    async def answer_question(
        self,
        findings: SessionFindings,
        report: FinalReport | None,
        question: str,
        history: list[QAEntry],
    ) -> QuestionAnswerOutput:
        """Answer a follow-up question using only the session findings."""
        system_prompt = (
            self._question_template.replace("{{AREA}}", findings.area)
            .replace("{{OUTPUT_SCHEMA}}", json.dumps(QuestionAnswerOutput.model_json_schema(), indent=2))
        )
        payload = {
            "findings": findings.to_llm_payload(),
            "report_summary": report.plain_english_summary if report is not None else None,
            "interpretation": report.interpretation.model_dump(mode="json") if report is not None else None,
            "previous_questions": [{"question": entry.question, "answer": entry.answer} for entry in history[-5:]],
            "question": question,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [text_part(json.dumps(payload, indent=1, default=str))]},
        ]
        return await self._request_structured(messages, QuestionAnswerOutput, self._settings.reasoning_max_tokens)

    async def _request_structured(self, messages: list[dict[str, Any]], model_type: type[ModelType], max_tokens: int) -> ModelType:
        schema = model_type.model_json_schema()
        guided_schema = schema if self._settings.use_guided_json else None
        conversation = list(messages)
        client = self._registry.reasoning_client()
        last_error = "unknown error"
        for _attempt in range(1 + max(0, self._settings.json_repair_attempts)):
            content = await client.complete(conversation, max_tokens=max_tokens, json_schema=guided_schema)
            try:
                return model_type.model_validate_json(extract_json_text(content))
            except (ValueError, ValidationError) as error:
                last_error = str(error)[:1500]
                repair_prompt = (
                    self._repair_template.replace("{{SCHEMA}}", json.dumps(schema, indent=2))
                    .replace("{{ERROR}}", last_error)
                    .replace("{{PREVIOUS_OUTPUT}}", content[:4000])
                )
                conversation = conversation + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": repair_prompt},
                ]
        raise LLMResponseValidationError(f"Model output did not match {model_type.__name__}: {last_error}")

    async def _run_consultations(self, evidence: list[EvidenceCrop], warnings: list[str]) -> list[ConsultantOpinion]:
        candidates = [
            crop for crop in evidence if crop.needs_consultation and crop.image_jpeg_base64 and crop.consultation_question
        ]
        maximum = max(0, self._settings.max_consultations)
        if not candidates or maximum == 0:
            return []
        requests: list[ConsultationRequest] | None = None
        candidate_ids = {crop.crop_id for crop in candidates}
        plan_payload = [
            {
                "crop_id": crop.crop_id,
                "agent": crop.agent,
                "label": crop.label,
                "suggested_question": crop.consultation_question,
                "classification": None
                if crop.classification is None
                else {"top_label": crop.classification.top_label, "top_score": round(crop.classification.top_score, 3)},
            }
            for crop in candidates
        ]
        plan_messages = [
            {"role": "system", "content": self._plan_template.replace("{{MAX_CONSULTATIONS}}", str(maximum)).replace(
                "{{OUTPUT_SCHEMA}}", json.dumps(ConsultationPlan.model_json_schema(), indent=2)
            )},
            {"role": "user", "content": [text_part(json.dumps(plan_payload, indent=1))]},
        ]
        try:
            plan = await self._request_structured(plan_messages, ConsultationPlan, 500)
            requests = [request for request in plan.consultations if request.crop_id in candidate_ids][:maximum]
        except (LLMServiceError, LLMResponseValidationError) as error:
            warnings.append(f"Consultation planning unavailable; doubtful crops were selected automatically ({error.message}).")
        if requests is None:
            ranked = sorted(
                candidates, key=lambda crop: crop.classification.margin if crop.classification is not None else 1.0
            )
            requests = [
                ConsultationRequest(crop_id=crop.crop_id, question=crop.consultation_question or "")
                for crop in ranked[:maximum]
                if crop.consultation_question
            ]
        crops_by_id = {crop.crop_id: crop for crop in candidates}
        opinions: list[ConsultantOpinion] = []
        for request in requests:
            opinion = await self._consultant.consult(crops_by_id[request.crop_id], request.question)
            if opinion is None:
                warnings.append(f"Medical consultant unavailable for crop {request.crop_id}.")
                continue
            opinions.append(opinion)
        return opinions

    @staticmethod
    def _most_doubtful_crop(evidence: list[EvidenceCrop]) -> EvidenceCrop | None:
        doubtful = [crop for crop in evidence if crop.needs_consultation and crop.image_jpeg_base64]
        if not doubtful:
            return None
        return min(doubtful, key=lambda crop: crop.classification.margin if crop.classification is not None else 1.0)

    def _rule_based_output(
        self,
        findings: SessionFindings,
        matches: list[RuleMatch],
        floor: ReferralUrgency,
        rule_summary_texts: dict[str, str],
    ) -> ReasoningOutput:
        if findings.flags:
            flag_titles = "; ".join(f"{flag.title} ({round(flag.confidence * 100)}%)" for flag in findings.flags)
            summary = rule_summary_texts["flags"].format(count=len(findings.flags), flags=flag_titles)
        else:
            summary = rule_summary_texts["none"]
        conditions = [
            PossibleCondition(
                name=match.condition,
                supporting_signs=match.supporting_signs,
                confidence=match.confidence,
                rationale=match.explanation,
            )
            for match in matches
        ]
        next_tests: list[str] = []
        for match in matches:
            for test in match.next_tests:
                if test not in next_tests:
                    next_tests.append(test)
        return ReasoningOutput(
            findings_summary=summary,
            possible_conditions=conditions[:10],
            confidence=findings.overall_confidence,
            recommended_next_tests=next_tests[:10],
            repeat_steps=[],
            referral_urgency=floor,
        )

    @staticmethod
    def _enforce_constraints(
        output: ReasoningOutput,
        findings: SessionFindings,
        matches: list[RuleMatch],
        floor: ReferralUrgency,
        protocol: AreaProtocol,
        warnings: list[str],
    ) -> ReasoningOutput:
        flag_confidence: dict[str, float] = {}
        for flag in findings.flags:
            flag_confidence[flag.code] = max(flag_confidence.get(flag.code, 0.0), flag.confidence)
        conditions: list[PossibleCondition] = []
        for condition in output.possible_conditions:
            signs = [sign for sign in condition.supporting_signs if sign in flag_confidence]
            if not signs:
                warnings.append(f"Removed '{condition.name}': no threshold-based screening flag supports it.")
                continue
            capped_confidence = min(condition.confidence, max(flag_confidence[sign] for sign in signs))
            conditions.append(condition.model_copy(update={"supporting_signs": signs, "confidence": capped_confidence}))
        valid_step_ids = {step.id for step in protocol.steps}
        repeat_steps = [step_id for step_id in output.repeat_steps if base_step_id(step_id) in valid_step_ids]
        for record in findings.steps:
            if record.status == StepStatus.INCOMPLETE and record.step_id not in repeat_steps:
                repeat_steps.append(record.step_id)
        next_tests = list(dict.fromkeys(output.recommended_next_tests))
        for match in matches:
            for test in match.next_tests:
                if test not in next_tests:
                    next_tests.append(test)
        return ReasoningOutput(
            findings_summary=output.findings_summary,
            possible_conditions=conditions,
            confidence=min(output.confidence, findings.overall_confidence),
            recommended_next_tests=next_tests[:10],
            repeat_steps=repeat_steps[:20],
            referral_urgency=max_urgency(output.referral_urgency, floor),
        )
