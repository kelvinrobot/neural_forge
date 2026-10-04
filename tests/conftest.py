"""Shared pytest fixtures."""

import pytest

from neural_forge.core.protocol_loader import ConfigRepository
from tests.synthetic import config_repository


@pytest.fixture(scope="session")
def config() -> ConfigRepository:
    """Configuration repository for the project's config directory."""
    return config_repository()
