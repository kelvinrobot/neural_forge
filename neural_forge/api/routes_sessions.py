"""Session lifecycle endpoints: create, summary, report, follow-up question and delete."""

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import Path as PathParameter

from neural_forge.agents.core.reasoning import ReasoningAgent
from neural_forge.agents.core.safety import SafetyAgent
from neural_forge.api.dependencies import (
    ActiveConnectionRegistry,
    get_connection_registry,
    get_reasoning_agent,
    get_safety_agent,
    get_session_repository,
)
from neural_forge.core.exceptions import LLMResponseValidationError, LLMServiceError
from neural_forge.core.protocol_loader import ConfigRepository, get_config_repository
from neural_forge.core.session_store import SessionRepository
from neural_forge.core.settings import Settings, get_settings
from neural_forge.schemas.protocols import build_step_view
from neural_forge.schemas.reports import FinalReport
from neural_forge.schemas.sessions import (
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteSessionResponse,
    QAEntry,
    QuestionRequest,
    QuestionResponse,
    SessionState,
    SessionStatus,
    SessionSummaryResponse,
)

router = APIRouter(tags=["sessions"])

SESSION_ID_PATTERN = r"^[0-9a-f]{32}$"


@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    repository: SessionRepository = Depends(get_session_repository),
    config: ConfigRepository = Depends(get_config_repository),
    safety: SafetyAgent = Depends(get_safety_agent),
) -> CreateSessionResponse:
    """Create an examination session; consent must be true."""
    protocol = config.protocol(request.area)
    now = datetime.now(timezone.utc)
    session = SessionState(
        session_id=uuid.uuid4().hex,
        area=request.area,
        language=request.preferred_language,
        patient_context=request.patient_context,
        consent_given_at=now,
        status=SessionStatus.CREATED,
        created_at=now,
        updated_at=now,
    )
    await repository.create(session)
    return CreateSessionResponse(
        session_id=session.session_id,
        area=session.area,
        language=session.language,
        total_steps=protocol.total_steps,
        steps=[build_step_view(step, protocol.total_steps, session.language) for step in protocol.steps],
        websocket_path=f"/api/v1/ws/{session.area.value}/{session.session_id}",
        educational_notice=safety.disclaimer(session.language),
    )


@router.get("/sessions/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_summary(
    session_id: str = PathParameter(pattern=SESSION_ID_PATTERN),
    repository: SessionRepository = Depends(get_session_repository),
) -> SessionSummaryResponse:
    """Return the aggregated findings JSON."""
    session = await repository.get(session_id)
    if session.findings is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Findings are not available until the examination has been analysed.")
    return SessionSummaryResponse(session_id=session.session_id, area=session.area, status=session.status, findings=session.findings)


@router.get("/sessions/{session_id}/report", response_model=FinalReport)
async def get_report(
    session_id: str = PathParameter(pattern=SESSION_ID_PATTERN),
    repository: SessionRepository = Depends(get_session_repository),
) -> FinalReport:
    """Return the final report after the examination completes."""
    session = await repository.get(session_id)
    if session.report is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The report is not available until the examination has been analysed.")
    return session.report


@router.post("/sessions/{session_id}/question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    session_id: str = PathParameter(pattern=SESSION_ID_PATTERN),
    repository: SessionRepository = Depends(get_session_repository),
    reasoning: ReasoningAgent = Depends(get_reasoning_agent),
    safety: SafetyAgent = Depends(get_safety_agent),
    settings: Settings = Depends(get_settings),
) -> QuestionResponse:
    """Answer a follow-up question using only this session's findings."""
    if len(request.question) > settings.max_question_length:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Questions are limited to {settings.max_question_length} characters.")
    session = await repository.get(session_id)
    if session.findings is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Questions can be asked after the examination has been analysed.")
    try:
        output = await reasoning.answer_question(session.findings, session.report, request.question, session.questions)
    except (LLMServiceError, LLMResponseValidationError) as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"The reasoning service is unavailable: {error.message}") from error
    answer, _ = safety.review_answer(output.answer, output.confidence, session.language)
    session.questions.append(QAEntry(question=request.question, answer=answer, asked_at=datetime.now(timezone.utc)))
    await repository.save(session)
    return QuestionResponse(
        session_id=session_id,
        question=request.question,
        answer=answer,
        related_findings=output.related_findings,
        confidence=output.confidence,
        suggest_professional_follow_up=output.suggest_professional_follow_up,
        disclaimer=safety.disclaimer(session.language),
    )


@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(
    session_id: str = PathParameter(pattern=SESSION_ID_PATTERN),
    repository: SessionRepository = Depends(get_session_repository),
    connections: ActiveConnectionRegistry = Depends(get_connection_registry),
    settings: Settings = Depends(get_settings),
) -> DeleteSessionResponse:
    """Delete a session and all its data, closing any live connection."""
    await connections.cancel(session_id)
    deleted = await repository.delete(session_id)
    crops_directory = Path(settings.crops_dir) / session_id
    if crops_directory.exists():
        shutil.rmtree(crops_directory, ignore_errors=True)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} was not found")
    return DeleteSessionResponse(session_id=session_id, deleted=True)
