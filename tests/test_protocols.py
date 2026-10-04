"""Tests for protocol loading, threshold completeness and cross-file consistency."""

import pytest

from neural_forge.agents.area_registry import AREA_SPECIFICATIONS, get_area_specification, validate_protocol_agents
from neural_forge.core.protocol_loader import ConfigRepository, base_step_id, make_step_instance
from neural_forge.schemas.protocols import AreaName, CameraSetup, CompletionType, build_protocol_response, localize


@pytest.mark.parametrize("area", list(AreaName))
def test_protocol_loads_and_is_sequential(config: ConfigRepository, area: AreaName) -> None:
    protocol = config.protocol(area)
    assert protocol.area == area
    assert [step.step_number for step in protocol.steps] == list(range(1, protocol.total_steps + 1))
    assert protocol.steps[0].id == "get_positioned"


@pytest.mark.parametrize("area", list(AreaName))
def test_protocol_agents_are_valid(config: ConfigRepository, area: AreaName) -> None:
    assert validate_protocol_agents(config.protocol(area)) == []


@pytest.mark.parametrize("area", list(AreaName))
def test_every_agent_has_thresholds(config: ConfigRepository, area: AreaName) -> None:
    thresholds = config.thresholds(area)
    for agent_name in get_area_specification(area).agent_names:
        assert isinstance(thresholds.get(agent_name), dict), f"missing thresholds for {agent_name}"


@pytest.mark.parametrize("area", list(AreaName))
def test_completion_events_and_states_are_supported(config: ConfigRepository, area: AreaName) -> None:
    protocol = config.protocol(area)
    specification = get_area_specification(area)
    classes = {agent_class.name: agent_class for agent_class in specification.agent_classes}
    for step in protocol.steps:
        criteria = step.completion_criteria
        active_classes = [classes[name] for name in step.active_agents]
        if criteria.type == CompletionType.EVENTS:
            assert any(criteria.event in agent_class.supported_events for agent_class in active_classes), step.id
        if criteria.type == CompletionType.STATE:
            assert any(criteria.state in agent_class.supported_states for agent_class in active_classes), step.id


def test_quality_config_covers_every_camera_setup(config: ConfigRepository) -> None:
    setups = config.quality_config()["camera_setups"]
    for setup in CameraSetup:
        assert setup.value in setups


def test_step_instance_round_trip() -> None:
    assert make_step_instance("area_of_concern", 1) == "area_of_concern"
    assert make_step_instance("area_of_concern", 3) == "area_of_concern__3"
    assert base_step_id("area_of_concern__3") == "area_of_concern"


def test_localisation_falls_back_to_english(config: ConfigRepository) -> None:
    protocol = config.protocol(AreaName.FACE)
    assert localize(protocol.steps[0].instruction, "yo") == protocol.steps[0].instruction["en"]
    response = build_protocol_response(protocol, "ha")
    assert response.total_steps == protocol.total_steps
    assert response.steps[0].title


def test_specific_protocol_requirements(config: ConfigRepository) -> None:
    skin = config.protocol(AreaName.SKIN)
    concern = skin.step_by_id("area_of_concern")
    assert concern is not None and concern.repeatable and concern.skippable and concern.max_repetitions > 1
    posture = config.protocol(AreaName.POSTURE)
    assert all(step.safety_note for step in posture.steps)
    assert posture.total_steps == 14
    assert config.protocol(AreaName.EYES).total_steps == 11
    assert config.protocol(AreaName.FACE).total_steps == 10
    assert config.protocol(AreaName.MOUTH).total_steps == 9
    assert config.protocol(AreaName.HANDS).total_steps == 9
    assert config.protocol(AreaName.MOVEMENT).total_steps == 11
    assert skin.total_steps == 8
    for area in AreaName:
        reference = config.protocol(area).step_by_id("colour_reference")
        if reference is not None:
            assert reference.skippable and reference.captures_colour_reference


def test_registry_covers_all_areas() -> None:
    assert set(AREA_SPECIFICATIONS) == set(AreaName)
    total_agents = sum(len(specification.agent_classes) for specification in AREA_SPECIFICATIONS.values())
    assert total_agents == 28


def test_messages_and_rules_load(config: ConfigRepository) -> None:
    messages = config.ui_messages()
    assert messages.text("progress_frames_missing", "en", captured=1, required=2)
    assert "{count}" in messages.template("rule_summary_flags", "en")
    rules = config.sign_condition_rules()
    assert rules["rules"] and "flag_severity_urgency" in rules
    assert config.reference_ranges()["areas"]["eyes"]
