"""FastAPI application factory and Uvicorn entrypoint."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neural_forge import __version__
from neural_forge.agents.area_registry import validate_protocol_agents
from neural_forge.api import routes_health, routes_protocols, routes_sessions, websocket_stream
from neural_forge.api.dependencies import (
    get_executor,
    get_report_agent,
    get_reasoning_agent,
    get_safety_agent,
    get_session_repository,
)
from neural_forge.core.exceptions import (
    ConfigurationError,
    ConsentRequiredError,
    FrameRejectedError,
    InvalidMessageError,
    LLMServiceError,
    ModelNotAvailableError,
    NeuralForgeError,
    SessionNotFoundError,
    SessionStateError,
)
from neural_forge.core.logging import configure_logging, get_logger, log_event
from neural_forge.core.protocol_loader import get_config_repository
from neural_forge.core.settings import get_settings
from neural_forge.models.registry import get_model_registry

logger = get_logger(__name__)

API_PREFIX = "/api/v1"

STATUS_BY_ERROR: dict[type[NeuralForgeError], int] = {
    SessionNotFoundError: 404,
    SessionStateError: 409,
    ConsentRequiredError: 400,
    InvalidMessageError: 400,
    FrameRejectedError: 400,
    LLMServiceError: 503,
    ModelNotAvailableError: 503,
    ConfigurationError: 500,
}


def validate_configuration() -> None:
    """Load every protocol, threshold file and rule file at startup so configuration errors fail fast."""
    config = get_config_repository()
    problems: list[str] = []
    for area, protocol in config.all_protocols().items():
        config.thresholds(area)
        problems.extend(validate_protocol_agents(protocol))
    config.quality_config()
    config.ui_messages()
    config.zero_shot_prompts()
    get_safety_agent()
    get_reasoning_agent()
    get_report_agent()
    if problems:
        raise ConfigurationError("Protocol configuration problems: " + "; ".join(problems))


async def purge_sessions_periodically(interval_seconds: float) -> None:
    """Delete expired sessions in the background."""
    repository = get_session_repository()
    while True:
        await asyncio.sleep(interval_seconds)
        removed = await repository.purge_expired()
        if removed:
            log_event(logger, logging.INFO, "sessions_purged", count=removed)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Validate configuration on startup and release resources on shutdown."""
    settings = get_settings()
    validate_configuration()
    purge_task = asyncio.create_task(purge_sessions_periodically(settings.session_purge_interval_seconds))
    log_event(logger, logging.INFO, "service_started", version=__version__, port=settings.port)
    try:
        yield
    finally:
        purge_task.cancel()
        await asyncio.gather(purge_task, return_exceptions=True)
        await get_model_registry().aclose()
        get_executor().shutdown()
        log_event(logger, logging.INFO, "service_stopped")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="health care webcam-based visual screening backend. Outputs are screening flags, with  diagnoses.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(routes_sessions.router, prefix=API_PREFIX)
    application.include_router(routes_protocols.router, prefix=API_PREFIX)
    application.include_router(routes_health.router, prefix=API_PREFIX)
    application.include_router(websocket_stream.router, prefix=API_PREFIX)

    @application.exception_handler(NeuralForgeError)
    async def handle_application_error(request: Request, error: NeuralForgeError) -> JSONResponse:
        status_code = 500
        for error_type, mapped_status in STATUS_BY_ERROR.items():
            if isinstance(error, error_type):
                status_code = mapped_status
                break
        log_event(logger, logging.WARNING, "request_failed", path=request.url.path, code=error.code, error=error.message)
        return JSONResponse(status_code=status_code, content={"code": error.code, "detail": error.message})

    return application


app = create_app()


def run() -> None:
    """Start Uvicorn honouring proxy headers and WebSocket keep-alive settings."""
    settings = get_settings()
    uvicorn.run(
        "neural_forge.main:app",
        host=settings.host,
        port=settings.port,
        proxy_headers=True,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        ws_ping_interval=settings.ws_ping_interval_seconds,
        ws_ping_timeout=settings.ws_ping_timeout_seconds,
        log_config=None,
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    run()
