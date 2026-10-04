"""Loading, validation and caching of YAML configuration and examination protocols."""

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from neural_forge.core.exceptions import ConfigurationError, ProtocolError
from neural_forge.core.settings import get_settings
from neural_forge.schemas.protocols import AreaName, AreaProtocol, localize, localize_list

REPETITION_SEPARATOR = "__"


def base_step_id(step_instance: str) -> str:
    """Return the protocol step id for a step instance label."""
    return step_instance.split(REPETITION_SEPARATOR, 1)[0]


def make_step_instance(step_id: str, repetition: int) -> str:
    """Return the instance label for a repetition of a step."""
    if repetition <= 1:
        return step_id
    return f"{step_id}{REPETITION_SEPARATOR}{repetition}"


class UIMessages:
    """Localised patient-facing messages shared by all protocols."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._messages: dict[str, dict[str, str]] = data.get("messages", {})
        self._encouragements: dict[str, list[str]] = data.get("encouragements", {})

    def template(self, key: str, language: str) -> str:
        """Return an unformatted message template in the requested language."""
        template = localize(self._messages.get(key), language)
        if template is None:
            raise ConfigurationError(f"UI message '{key}' is missing from messages.yaml")
        return template

    def text(self, key: str, language: str, **values: Any) -> str:
        """Return a formatted message in the requested language."""
        return self.template(key, language).format(**values)

    def encouragement(self, index: int, language: str) -> str:
        """Return an encouraging phrase, rotating through the configured list."""
        phrases = localize_list(self._encouragements, language)
        if not phrases:
            return self.text("step_complete_default", language)
        return phrases[index % len(phrases)]


class ConfigRepository:
    """Loads and caches YAML configuration files from the config directory."""

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self._raw_cache: dict[str, Any] = {}
        self._protocol_cache: dict[AreaName, AreaProtocol] = {}
        self._lock = threading.Lock()

    @property
    def config_dir(self) -> Path:
        """Root configuration directory."""
        return self._config_dir

    def load_yaml(self, relative_path: str) -> Any:
        """Load and cache a YAML file relative to the configuration directory."""
        with self._lock:
            if relative_path in self._raw_cache:
                return self._raw_cache[relative_path]
            path = self._config_dir / relative_path
            if not path.exists():
                raise ConfigurationError(f"Configuration file not found: {path}")
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle)
            except yaml.YAMLError as error:
                raise ConfigurationError(f"Invalid YAML in {path}: {error}") from error
            if data is None:
                raise ConfigurationError(f"Configuration file is empty: {path}")
            self._raw_cache[relative_path] = data
            return data

    def protocol(self, area: AreaName | str) -> AreaProtocol:
        """Return the validated examination protocol for an area."""
        area_name = AreaName(area)
        if area_name in self._protocol_cache:
            return self._protocol_cache[area_name]
        data = self.load_yaml(f"protocols/{area_name.value}.yaml")
        try:
            protocol = AreaProtocol.model_validate(data)
        except ValidationError as error:
            raise ProtocolError(f"Protocol for '{area_name.value}' is invalid: {error}") from error
        if protocol.area != area_name:
            raise ProtocolError(f"Protocol file for '{area_name.value}' declares area '{protocol.area.value}'")
        with self._lock:
            self._protocol_cache[area_name] = protocol
        return protocol

    def all_protocols(self) -> dict[AreaName, AreaProtocol]:
        """Return the protocols for every area."""
        return {area: self.protocol(area) for area in AreaName}

    def thresholds(self, area: AreaName | str) -> dict[str, Any]:
        """Return the threshold configuration for an area."""
        area_name = AreaName(area)
        data = self.load_yaml(f"thresholds/{area_name.value}.yaml")
        if not isinstance(data, dict):
            raise ConfigurationError(f"Thresholds for '{area_name.value}' must be a mapping")
        if data.get("educational_defaults") is not True:
            raise ConfigurationError(
                f"Thresholds for '{area_name.value}' must declare 'educational_defaults: true'"
            )
        return data

    def quality_config(self) -> dict[str, Any]:
        """Return the camera quality and colour calibration configuration."""
        data = self.load_yaml("quality.yaml")
        for required_key in ("global", "camera_setups", "guidance", "colour_calibration", "aggregation"):
            if required_key not in data:
                raise ConfigurationError(f"quality.yaml is missing the '{required_key}' section")
        return data

    def ui_messages(self) -> UIMessages:
        """Return the shared patient-facing messages."""
        return UIMessages(self.load_yaml("messages.yaml"))

    def zero_shot_prompts(self) -> dict[str, Any]:
        """Return the zero-shot classification prompt configuration."""
        data = self.load_yaml("zero_shot_prompts.yaml")
        if "tasks" not in data:
            raise ConfigurationError("zero_shot_prompts.yaml must define 'tasks'")
        return data

    def safety_rules(self) -> dict[str, Any]:
        """Return the safety agent rule configuration."""
        return self.load_yaml("safety_rules.yaml")

    def reference_ranges(self) -> dict[str, Any]:
        """Return the reference ranges used in the reasoning prompt."""
        return self.load_yaml("reasoning/reference_ranges.yaml")

    def sign_condition_rules(self) -> dict[str, Any]:
        """Return the sign-to-condition rules used for reasoning."""
        return self.load_yaml("reasoning/sign_condition_rules.yaml")


@lru_cache(maxsize=1)
def get_config_repository() -> ConfigRepository:
    """Return the process-wide configuration repository."""
    return ConfigRepository(get_settings().config_dir)
