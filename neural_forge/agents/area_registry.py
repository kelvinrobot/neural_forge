"""Maps each examination area to its agents, trackers and protocol file."""

from dataclasses import dataclass

from neural_forge.agents.base import AgentContext, AreaAgent
from neural_forge.agents.eyes.blink_alertness import BlinkAlertnessAgent
from neural_forge.agents.eyes.eye_alignment import EyeAlignmentAgent
from neural_forge.agents.eyes.eye_appearance import EyeAppearanceAgent
from neural_forge.agents.eyes.eye_movement import EyeMovementAgent
from neural_forge.agents.eyes.eyelid import EyelidAgent
from neural_forge.agents.face.face_colour import FaceColourAgent
from neural_forge.agents.face.facial_expression import FacialExpressionAgent
from neural_forge.agents.face.facial_symmetry import FacialSymmetryAgent
from neural_forge.agents.face.pain import PainAgent
from neural_forge.agents.hands.capillary_refill import CapillaryRefillAgent
from neural_forge.agents.hands.finger_shape import FingerShapeAgent
from neural_forge.agents.hands.hand_colour import HandColourAgent
from neural_forge.agents.hands.joint_appearance import JointAppearanceAgent
from neural_forge.agents.mouth.mouth_movement import MouthMovementAgent
from neural_forge.agents.mouth.oral_appearance import OralAppearanceAgent
from neural_forge.agents.movement.arm_drift import ArmDriftAgent
from neural_forge.agents.movement.coordination import CoordinationAgent
from neural_forge.agents.movement.finger_tapping import FingerTappingAgent
from neural_forge.agents.movement.involuntary_movement import InvoluntaryMovementAgent
from neural_forge.agents.movement.tremor import TremorAgent
from neural_forge.agents.posture.balance import BalanceAgent
from neural_forge.agents.posture.functional_test import FunctionalTestAgent
from neural_forge.agents.posture.gait import GaitAgent
from neural_forge.agents.posture.range_of_motion import RangeOfMotionAgent
from neural_forge.agents.posture.static_posture import StaticPostureAgent
from neural_forge.agents.skin.lesion_classifier import LesionClassifierAgent
from neural_forge.agents.skin.lesion_finder import LesionFinderAgent
from neural_forge.agents.skin.pigmentation import PigmentationAgent
from neural_forge.core.exceptions import ConfigurationError
from neural_forge.schemas.protocols import AreaName, AreaProtocol
from neural_forge.services.landmark_service import TrackerName


@dataclass(frozen=True)
class AreaSpecification:
    """Agents and processing requirements of one examination area."""

    area: AreaName
    agent_classes: tuple[type[AreaAgent], ...]
    needs_colour_calibration: bool

    @property
    def agent_names(self) -> list[str]:
        """Names of the area's agents in processing order."""
        return [agent_class.name for agent_class in self.agent_classes]

    @property
    def required_trackers(self) -> frozenset[TrackerName]:
        """Union of the trackers required by the area's agents."""
        trackers: set[TrackerName] = set()
        for agent_class in self.agent_classes:
            trackers.update(agent_class.required_trackers)
        return frozenset(trackers)

    @property
    def protocol_file(self) -> str:
        """Relative path of the area's protocol file."""
        return f"protocols/{self.area.value}.yaml"

    def build_agents(self, context: AgentContext) -> list[AreaAgent]:
        """Instantiate the area's agents and let them discover each other."""
        agents = [agent_class(context) for agent_class in self.agent_classes]
        agents_by_name = {agent.name: agent for agent in agents}
        for agent in agents:
            agent.bind_peers(agents_by_name)
        return agents


AREA_SPECIFICATIONS: dict[AreaName, AreaSpecification] = {
    AreaName.EYES: AreaSpecification(
        area=AreaName.EYES,
        agent_classes=(EyelidAgent, EyeAlignmentAgent, EyeMovementAgent, BlinkAlertnessAgent, EyeAppearanceAgent),
        needs_colour_calibration=True,
    ),
    AreaName.FACE: AreaSpecification(
        area=AreaName.FACE,
        agent_classes=(FacialSymmetryAgent, FacialExpressionAgent, PainAgent, FaceColourAgent),
        needs_colour_calibration=True,
    ),
    AreaName.MOUTH: AreaSpecification(
        area=AreaName.MOUTH,
        agent_classes=(MouthMovementAgent, OralAppearanceAgent),
        needs_colour_calibration=True,
    ),
    AreaName.HANDS: AreaSpecification(
        area=AreaName.HANDS,
        agent_classes=(HandColourAgent, CapillaryRefillAgent, FingerShapeAgent, JointAppearanceAgent),
        needs_colour_calibration=True,
    ),
    AreaName.MOVEMENT: AreaSpecification(
        area=AreaName.MOVEMENT,
        agent_classes=(TremorAgent, FingerTappingAgent, CoordinationAgent, ArmDriftAgent, InvoluntaryMovementAgent),
        needs_colour_calibration=False,
    ),
    AreaName.POSTURE: AreaSpecification(
        area=AreaName.POSTURE,
        agent_classes=(StaticPostureAgent, RangeOfMotionAgent, GaitAgent, BalanceAgent, FunctionalTestAgent),
        needs_colour_calibration=False,
    ),
    AreaName.SKIN: AreaSpecification(
        area=AreaName.SKIN,
        agent_classes=(LesionFinderAgent, LesionClassifierAgent, PigmentationAgent),
        needs_colour_calibration=True,
    ),
}


def get_area_specification(area: AreaName | str) -> AreaSpecification:
    """Return the specification of an area."""
    try:
        return AREA_SPECIFICATIONS[AreaName(area)]
    except (KeyError, ValueError) as error:
        raise ConfigurationError(f"Unknown examination area '{area}'") from error


def validate_protocol_agents(protocol: AreaProtocol) -> list[str]:
    """Return problems where protocol steps reference agents that do not belong to the area."""
    specification = get_area_specification(protocol.area)
    known = set(specification.agent_names)
    problems: list[str] = []
    for step in protocol.steps:
        for agent_name in step.active_agents:
            if agent_name not in known:
                problems.append(f"{protocol.area.value}/{step.id}: unknown agent '{agent_name}'")
    referenced = {agent_name for step in protocol.steps for agent_name in step.active_agents}
    for agent_name in known - referenced:
        problems.append(f"{protocol.area.value}: agent '{agent_name}' is not active in any step")
    return problems
