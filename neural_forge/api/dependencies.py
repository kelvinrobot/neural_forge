"""Process-wide singletons and FastAPI dependency providers."""

import asyncio
from functools import lru_cache

from neural_forge.agents.core.medical_consultant import MedicalConsultantAgent
from neural_forge.agents.core.reasoning import ReasoningAgent
from neural_forge.agents.core.report import ReportAgent
from neural_forge.agents.core.safety import SafetyAgent
from neural_forge.core.executor import InferenceExecutor
from neural_forge.core.pipeline import ExaminationPipeline
from neural_forge.core.protocol_loader import ConfigRepository, get_config_repository
from neural_forge.core.session_store import InMemorySessionRepository, SessionRepository
from neural_forge.core.settings import Settings, get_settings
from neural_forge.models.registry import ModelRegistry, get_model_registry
from neural_forge.schemas.sessions import SessionState


@lru_cache(maxsize=1)
def get_executor() -> InferenceExecutor:
    """Return the shared inference thread pool."""
    return InferenceExecutor(get_settings().executor_max_workers)


@lru_cache(maxsize=1)
def get_session_repository() -> SessionRepository:
    """Return the session repository (in-memory implementation)."""
    settings = get_settings()
    return InMemorySessionRepository(ttl_seconds=settings.session_ttl_seconds, max_sessions=settings.max_sessions)


@lru_cache(maxsize=1)
def get_safety_agent() -> SafetyAgent:
    """Return the shared safety agent."""
    return SafetyAgent(get_config_repository().safety_rules())


@lru_cache(maxsize=1)
def get_medical_consultant() -> MedicalConsultantAgent:
    """Return the shared medical consultant agent."""
    return MedicalConsultantAgent(get_settings(), get_model_registry())


@lru_cache(maxsize=1)
def get_reasoning_agent() -> ReasoningAgent:
    """Return the shared reasoning agent."""
    return ReasoningAgent(get_settings(), get_config_repository(), get_model_registry(), get_medical_consultant())


@lru_cache(maxsize=1)
def get_report_agent() -> ReportAgent:
    """Return the shared report agent."""
    return ReportAgent(get_config_repository(), get_safety_agent())


class PipelineFactory:
    """Creates a per-session examination pipeline with shared models and agents."""

    def __init__(
        self,
        settings: Settings,
        config: ConfigRepository,
        registry: ModelRegistry,
        executor: InferenceExecutor,
        reasoning: ReasoningAgent,
        safety: SafetyAgent,
        report: ReportAgent,
    ) -> None:
        self._settings = settings
        self._config = config
        self._registry = registry
        self._executor = executor
        self._reasoning = reasoning
        self._safety = safety
        self._report = report

    def create(self, session: SessionState) -> ExaminationPipeline:
        """Build a pipeline containing only the agents of the session's area."""
        return ExaminationPipeline(
            session=session,
            settings=self._settings,
            config=self._config,
            registry=self._registry,
            executor=self._executor,
            reasoning_agent=self._reasoning,
            safety_agent=self._safety,
            report_agent=self._report,
        )


@lru_cache(maxsize=1)
def get_pipeline_factory() -> PipelineFactory:
    """Return the shared pipeline factory."""
    return PipelineFactory(
        settings=get_settings(),
        config=get_config_repository(),
        registry=get_model_registry(),
        executor=get_executor(),
        reasoning=get_reasoning_agent(),
        safety=get_safety_agent(),
        report=get_report_agent(),
    )


class ActiveConnectionRegistry:
    """Tracks which sessions have a live WebSocket so each session has at most one connection."""

    def __init__(self) -> None:
        self._active: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, session_id: str) -> asyncio.Event | None:
        """Register a connection; returns a cancellation event, or None if the session is already connected."""
        async with self._lock:
            if session_id in self._active:
                return None
            event = asyncio.Event()
            self._active[session_id] = event
            return event

    async def release(self, session_id: str) -> None:
        """Remove a connection registration."""
        async with self._lock:
            self._active.pop(session_id, None)

    async def cancel(self, session_id: str) -> bool:
        """Ask an active connection to close (used when a session is deleted)."""
        async with self._lock:
            event = self._active.get(session_id)
            if event is None:
                return False
            event.set()
            return True

    async def count(self) -> int:
        """Number of active connections."""
        async with self._lock:
            return len(self._active)


@lru_cache(maxsize=1)
def get_connection_registry() -> ActiveConnectionRegistry:
    """Return the shared active connection registry."""
    return ActiveConnectionRegistry()
