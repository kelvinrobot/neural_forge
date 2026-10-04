"""Protocol listing endpoint."""

from fastapi import APIRouter, Depends, Query

from neural_forge.core.protocol_loader import ConfigRepository, get_config_repository
from neural_forge.schemas.protocols import AreaName, ProtocolResponse, build_protocol_response

router = APIRouter(tags=["protocols"])


@router.get("/protocols/{area}", response_model=ProtocolResponse)
async def get_protocol(
    area: AreaName,
    language: str = Query(default="en", pattern=r"^[a-z]{2,3}$"),
    config: ConfigRepository = Depends(get_config_repository),
) -> ProtocolResponse:
    """Return the examination steps of an area so a client can show them before starting."""
    return build_protocol_response(config.protocol(area), language)
