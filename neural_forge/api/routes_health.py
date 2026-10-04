"""Health endpoint reporting LLM server readiness without loading any model."""

import asyncio
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from neural_forge.api.dependencies import get_connection_registry, ActiveConnectionRegistry
from neural_forge.core.settings import Settings, get_settings
from neural_forge.models.registry import ModelRegistry, get_model_registry

router = APIRouter(tags=["health"])


class LLMServerStatus(BaseModel):
    """Reachability of one vLLM server."""

    role: str
    base_url: str
    served_model_name: str
    reachable: bool


class HealthResponse(BaseModel):
    """Service health information."""

    status: Literal["starting", "ready", "degraded"]
    version: str
    uptime_seconds: float
    gpu_available: bool
    models: dict[str, str]
    llm_servers: list[LLMServerStatus]
    perception_assets_present: dict[str, bool]
    active_connections: int


class HealthState:
    """Remembers whether the service has ever been fully ready."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self.ever_ready = False


HEALTH_STATE = HealthState()


def gpu_available() -> bool:
    """Detect an NVIDIA GPU from device files without importing any ML library."""
    return Path("/dev/nvidia0").exists() or Path("/proc/driver/nvidia/version").exists()


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
    connections: ActiveConnectionRegistry = Depends(get_connection_registry),
) -> HealthResponse:
    """Report 'starting' until both LLM servers respond, then 'ready'."""
    reasoning_client = registry.reasoning_client()
    consultant_client = registry.consultant_client()
    reasoning_ok, consultant_ok = await asyncio.gather(reasoning_client.is_healthy(), consultant_client.is_healthy())
    both_ready = reasoning_ok and consultant_ok
    if both_ready:
        HEALTH_STATE.ever_ready = True
        status: Literal["starting", "ready", "degraded"] = "ready"
    elif HEALTH_STATE.ever_ready:
        status = "degraded"
    else:
        status = "starting"
    return HealthResponse(
        status=status,
        version=settings.app_version,
        uptime_seconds=round(time.monotonic() - HEALTH_STATE.started_at, 1),
        gpu_available=gpu_available(),
        models=registry.configured_models(),
        llm_servers=[
            LLMServerStatus(role="reasoning", base_url=reasoning_client.base_url, served_model_name=reasoning_client.served_model_name, reachable=reasoning_ok),
            LLMServerStatus(role="medical_consultant", base_url=consultant_client.base_url, served_model_name=consultant_client.served_model_name, reachable=consultant_ok),
        ],
        perception_assets_present={
            "face_landmarker": settings.face_landmarker_path.exists(),
            "hand_landmarker": settings.hand_landmarker_path.exists(),
            "yoloe": settings.yoloe_weights_path.exists(),
        },
        active_connections=await connections.count(),
    )
