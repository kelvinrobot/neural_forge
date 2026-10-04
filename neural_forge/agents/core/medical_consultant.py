"""Medical consultant agent: single-turn MedGemma opinion on one image crop and one question."""

import json
import logging

from pydantic import ValidationError

from neural_forge.core.exceptions import LLMServiceError
from neural_forge.core.logging import get_logger, log_event
from neural_forge.core.settings import Settings
from neural_forge.models.llm_clients import extract_json_text, image_part, load_prompt, text_part
from neural_forge.models.registry import ModelRegistry
from neural_forge.schemas.findings import ConsultantOpinion, ConsultantResponse
from neural_forge.schemas.readings import EvidenceCrop

logger = get_logger(__name__)


class MedicalConsultantAgent:
    """Tool used by the reasoning agent for focused single-image questions."""

    def __init__(self, settings: Settings, registry: ModelRegistry) -> None:
        self._settings = settings
        self._registry = registry
        template = load_prompt(settings.prompts_dir, "consultant_system.md")
        schema_text = json.dumps(ConsultantResponse.model_json_schema(), indent=2)
        self._system_prompt = template.replace("{{OUTPUT_SCHEMA}}", schema_text)

    async def consult(self, crop: EvidenceCrop, question: str) -> ConsultantOpinion | None:
        """Ask one focused question about one image crop; returns None when the server is unavailable."""
        if not crop.image_jpeg_base64:
            return None
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [image_part(crop.image_jpeg_base64), text_part(question)]},
        ]
        schema = ConsultantResponse.model_json_schema() if self._settings.use_guided_json else None
        try:
            content = await self._registry.consultant_client().complete(messages, json_schema=schema)
        except LLMServiceError as error:
            log_event(logger, logging.WARNING, "consultant_unavailable", crop_id=crop.crop_id, error=error.message)
            return None
        try:
            parsed = ConsultantResponse.model_validate_json(extract_json_text(content))
        except (ValueError, ValidationError) as error:
            log_event(logger, logging.WARNING, "consultant_unstructured_response", crop_id=crop.crop_id, error=str(error)[:300])
            return ConsultantOpinion(
                crop_id=crop.crop_id,
                question=question,
                observation=content.strip()[:1200] or "No observation returned.",
                likely_finding="unstructured response",
                confidence=0.0,
                recommend_in_person_review=True,
                parsed=False,
            )
        return ConsultantOpinion(
            crop_id=crop.crop_id,
            question=question,
            observation=parsed.observation,
            likely_finding=parsed.likely_finding,
            confidence=parsed.confidence,
            recommend_in_person_review=parsed.recommend_in_person_review,
            parsed=True,
        )
