"""Skin agent tests with synthetic images and unavailable models."""

from neural_forge.agents.area_registry import get_area_specification
from neural_forge.agents.skin.lesion_classifier import LesionClassifierAgent
from neural_forge.agents.skin.lesion_finder import LesionFinderAgent, intersection_over_union
from neural_forge.agents.skin.pigmentation import PigmentationAgent
from tests.synthetic import make_context, make_frame, skin_image


def build_skin_agents() -> dict[str, object]:
    agents = get_area_specification("skin").build_agents(make_context("skin"))
    return {agent.name: agent for agent in agents}


def test_intersection_over_union() -> None:
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)) - 1.0 / 3.0) < 1e-9


def test_lesion_finder_falls_back_gracefully_without_models() -> None:
    agents = build_skin_agents()
    finder = agents["lesion_finder"]
    classifier = agents["lesion_classifier"]
    assert isinstance(finder, LesionFinderAgent) and isinstance(classifier, LesionClassifierAgent)
    finder.on_step_start("forearms_inner", 0.0)
    classifier.on_step_start("forearms_inner", 0.0)
    for index in range(20):
        frame = make_frame("forearms_inner", index * 0.1, image=skin_image())
        assert finder.process_frame(frame)
        assert classifier.process_frame(frame)
    finder_result = finder.finalize()
    assert any("unavailable" in limitation for limitation in finder_result.limitations)
    assert finder_result.flags == []
    classifier_result = classifier.finalize()
    assert classifier_result.flags == []


def test_depigmented_patch_is_flagged() -> None:
    agent = PigmentationAgent(make_context("skin"))
    image = skin_image()
    image[180:300, 260:380] = (205, 210, 238)
    agent.on_step_start("forearms_inner", 0.0)
    for index in range(20):
        agent.process_frame(make_frame("forearms_inner", index * 0.1, image=image))
    assert "vitiligo_pattern_depigmentation" in {flag.code for flag in agent.finalize().flags}


def test_uniform_skin_has_no_pigmentation_flags() -> None:
    agent = PigmentationAgent(make_context("skin"))
    agent.on_step_start("forearms_inner", 0.0)
    for index in range(20):
        agent.process_frame(make_frame("forearms_inner", index * 0.1, image=skin_image()))
    assert agent.finalize().flags == []
