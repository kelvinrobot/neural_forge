"""WebSocket endpoint streaming frames into the examination pipeline and protocol messages back."""

import asyncio
import logging
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from neural_forge.agents.core.orchestrator import ExaminationOutcome, SessionOrchestratorAgent
from neural_forge.api.dependencies import get_connection_registry, get_pipeline_factory, get_session_repository
from neural_forge.core.exceptions import ConfigurationError, FrameRejectedError, NeuralForgeError, SessionNotFoundError
from neural_forge.core.logging import get_logger, log_event
from neural_forge.core.pipeline import ExaminationPipeline
from neural_forge.core.session_store import SessionRepository
from neural_forge.core.settings import Settings, get_settings
from neural_forge.schemas.findings import ReferralUrgency
from neural_forge.schemas.messages import (
    CLIENT_MESSAGE_ADAPTER,
    AnalysisStartedMessage,
    ControlMessage,
    ErrorMessage,
    FrameMessage,
    KeepaliveMessage,
    ReferenceCardMessage,
    ReportReadyMessage,
    ServerMessage,
)
from neural_forge.schemas.protocols import AreaName
from neural_forge.schemas.sessions import SessionState, SessionStatus

router = APIRouter(tags=["stream"])
logger = get_logger(__name__)

SESSION_ID_REGEX = re.compile(r"^[0-9a-f]{32}$")
CLOSE_NORMAL = 1000
CLOSE_POLICY_VIOLATION = 1008
CLOSE_INTERNAL_ERROR = 1011


class ConnectionClosed(Exception):
    """Raised when sending to a WebSocket that is already closed."""


class ExaminationConnection:
    """Owns the tasks of one live examination: receiving, keep-alive, orchestration and analysis."""

    def __init__(
        self,
        websocket: WebSocket,
        session: SessionState,
        repository: SessionRepository,
        pipeline: ExaminationPipeline,
        settings: Settings,
        cancel_event: asyncio.Event,
    ) -> None:
        self._websocket = websocket
        self._session = session
        self._repository = repository
        self._pipeline = pipeline
        self._settings = settings
        self._cancel_event = cancel_event
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._examination_finished = False
        self._last_frame_error_time = 0.0

    async def send(self, message: ServerMessage) -> None:
        """Serialise and send one server message."""
        if self._closed:
            raise ConnectionClosed()
        async with self._send_lock:
            try:
                await self._websocket.send_text(message.model_dump_json())
            except (WebSocketDisconnect, RuntimeError) as error:
                self._closed = True
                raise ConnectionClosed() from error

    async def _safe_send(self, message: ServerMessage) -> None:
        try:
            await self.send(message)
        except ConnectionClosed:
            self._closed = True

    async def _receive_loop(self, orchestrator: SessionOrchestratorAgent) -> str:
        loop = asyncio.get_running_loop()
        maximum_message_length = int(self._settings.max_frame_bytes * 1.4) + 1024
        while True:
            try:
                raw_message = await asyncio.wait_for(self._websocket.receive(), timeout=self._settings.ws_idle_timeout_seconds)
            except asyncio.TimeoutError:
                if self._examination_finished:
                    continue
                await self._safe_send(ErrorMessage(code="idle_timeout", message="No messages were received for too long; the session was closed.", recoverable=False))
                return "idle_timeout"
            except (WebSocketDisconnect, RuntimeError):
                return "disconnected"
            if raw_message.get("type") == "websocket.disconnect":
                return "disconnected"
            text = raw_message.get("text")
            if text is None:
                await self._safe_send(ErrorMessage(code="invalid_message", message="Only text JSON messages are accepted."))
                continue
            if len(text) > maximum_message_length:
                await self._safe_send(ErrorMessage(code="message_too_large", message="The message exceeds the maximum allowed size."))
                continue
            try:
                message = CLIENT_MESSAGE_ADAPTER.validate_json(text)
            except ValidationError as error:
                await self._safe_send(ErrorMessage(code="invalid_message", message=f"Invalid message: {error.errors()[0].get('msg', 'validation error')}"))
                continue
            if isinstance(message, FrameMessage):
                try:
                    self._pipeline.frame_ingest.submit(message)
                except FrameRejectedError as error:
                    now = loop.time()
                    if now - self._last_frame_error_time >= 1.0:
                        self._last_frame_error_time = now
                        await self._safe_send(ErrorMessage(code=error.code, message=error.message))
            elif isinstance(message, ControlMessage):
                orchestrator.submit_control(message.action)
            elif isinstance(message, ReferenceCardMessage):
                orchestrator.set_reference_card_hint(message.present)

    async def _keepalive_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._settings.ws_keepalive_interval_seconds)
            await self._safe_send(KeepaliveMessage())

    async def _update_status(self, status: SessionStatus) -> SessionState | None:
        try:
            session = await self._repository.get(self._session.session_id)
        except SessionNotFoundError:
            return None
        session.status = status
        await self._repository.save(session)
        return session

    async def _analyse(self, outcome: ExaminationOutcome) -> None:
        session = await self._update_status(SessionStatus.ANALYSING)
        if session is None:
            return
        messages = self._pipeline.messages
        await self._safe_send(AnalysisStartedMessage(message=messages.text("analysis_started", self._pipeline.language)))
        try:
            findings, report = await self._pipeline.analyse(outcome.step_records, outcome.stopped)
        except Exception as error:
            detail = error.message if isinstance(error, NeuralForgeError) else repr(error)
            log_event(logger, logging.ERROR, "analysis_failed", session_id=self._session.session_id, error=detail)
            await self._update_status(SessionStatus.FAILED)
            await self._safe_send(ErrorMessage(code="analysis_failed", message="The analysis could not be completed.", recoverable=False))
            return
        try:
            session = await self._repository.get(self._session.session_id)
        except SessionNotFoundError:
            return
        session.step_records = outcome.step_records
        session.findings = findings
        session.report = report
        session.status = SessionStatus.STOPPED if outcome.stopped else SessionStatus.COMPLETED
        await self._repository.save(session)
        await self._safe_send(
            ReportReadyMessage(
                session_id=session.session_id,
                report_path=f"/api/v1/sessions/{session.session_id}/report",
                summary=report.plain_english_summary,
                referral_urgency=report.interpretation.referral_urgency if report.interpretation else ReferralUrgency.NONE,
                red_flag_alerts=[alert.message for alert in report.interpretation.red_flag_alerts],
                disclaimer=report.disclaimer,
            )
        )

    async def run(self) -> None:
        """Run the examination until it completes, the client disconnects or the session is deleted."""
        orchestrator = SessionOrchestratorAgent(self._pipeline, self.send, self._settings)
        await self._update_status(SessionStatus.ACTIVE)
        receive_task = asyncio.create_task(self._receive_loop(orchestrator), name="nf-receive")
        keepalive_task = asyncio.create_task(self._keepalive_loop(), name="nf-keepalive")
        orchestrator_task = asyncio.create_task(orchestrator.run(), name="nf-orchestrator")
        cancel_task = asyncio.create_task(self._cancel_event.wait(), name="nf-cancel")
        tasks = {receive_task, keepalive_task, orchestrator_task, cancel_task}
        try:
            done, _ = await asyncio.wait({receive_task, orchestrator_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if orchestrator_task in done:
                error = orchestrator_task.exception()
                if error is None:
                    outcome = orchestrator_task.result()
                    self._examination_finished = True
                    if outcome.started:
                        analysis_task = asyncio.create_task(self._analyse(outcome), name="nf-analysis")
                        tasks.add(analysis_task)
                        await asyncio.wait({analysis_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
                    else:
                        await self._update_status(SessionStatus.INTERRUPTED)
                elif isinstance(error, ConnectionClosed):
                    await self._update_status(SessionStatus.INTERRUPTED)
                else:
                    log_event(logger, logging.ERROR, "orchestrator_failed", session_id=self._session.session_id, error=repr(error))
                    await self._update_status(SessionStatus.FAILED)
                    await self._safe_send(ErrorMessage(code="internal_error", message="The examination stopped because of an internal error.", recoverable=False))
            elif receive_task in done:
                log_event(logger, logging.INFO, "client_left", session_id=self._session.session_id, reason=receive_task.result())
                await self._update_status(SessionStatus.INTERRUPTED)
            else:
                log_event(logger, logging.INFO, "session_deleted_during_stream", session_id=self._session.session_id)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._pipeline.close()
            if not self._closed:
                self._closed = True
                try:
                    await self._websocket.close(code=CLOSE_NORMAL)
                except (RuntimeError, WebSocketDisconnect):
                    log_event(logger, logging.DEBUG, "websocket_already_closed", session_id=self._session.session_id)


async def reject(websocket: WebSocket, code: str, message: str) -> None:
    """Accept, explain the problem and close the connection with a policy-violation code."""
    await websocket.accept()
    await websocket.send_text(ErrorMessage(code=code, message=message, recoverable=False).model_dump_json())
    await websocket.close(code=CLOSE_POLICY_VIOLATION)


@router.websocket("/ws/{area}/{session_id}")
async def examination_stream(websocket: WebSocket, area: str, session_id: str) -> None:
    """Stream camera frames for one session and receive step-by-step examination messages."""
    settings = get_settings()
    repository = get_session_repository()
    connections = get_connection_registry()
    if area not in {item.value for item in AreaName}:
        await reject(websocket, "unknown_area", f"Unknown examination area '{area}'.")
        return
    if not SESSION_ID_REGEX.match(session_id):
        await reject(websocket, "invalid_session", "The session id is not valid.")
        return
    try:
        session = await repository.get(session_id)
    except SessionNotFoundError:
        await reject(websocket, "session_not_found", "The session was not found or has expired.")
        return
    if session.area.value != area:
        await reject(websocket, "area_mismatch", f"This session was created for '{session.area.value}', not '{area}'.")
        return
    if session.status in (SessionStatus.ANALYSING, SessionStatus.COMPLETED, SessionStatus.STOPPED):
        await reject(websocket, "session_finished", "This examination has already finished; create a new session to repeat it.")
        return
    cancel_event = await connections.acquire(session_id)
    if cancel_event is None:
        await reject(websocket, "already_connected", "This session already has an active connection.")
        return
    try:
        try:
            pipeline = get_pipeline_factory().create(session)
        except ConfigurationError as error:
            log_event(logger, logging.ERROR, "pipeline_creation_failed", session_id=session_id, error=error.message)
            await websocket.accept()
            await websocket.send_text(ErrorMessage(code="configuration_error", message="The examination could not be prepared.", recoverable=False).model_dump_json())
            await websocket.close(code=CLOSE_INTERNAL_ERROR)
            return
        await websocket.accept()
        connection = ExaminationConnection(websocket, session, repository, pipeline, settings, cancel_event)
        await connection.run()
    finally:
        await connections.release(session_id)
