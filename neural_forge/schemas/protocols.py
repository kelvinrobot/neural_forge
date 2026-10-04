"""Examination protocol schemas and localisation helpers."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LocalizedText = dict[str, str]
LocalizedList = dict[str, list[str]]
DEFAULT_LANGUAGE = "en"


def localize(text: LocalizedText | None, language: str, fallback: str = DEFAULT_LANGUAGE) -> str | None:
    """Return text in the requested language, falling back to English."""
    if not text:
        return None
    if language in text:
        return text[language]
    if fallback in text:
        return text[fallback]
    return next(iter(text.values()))


def localize_list(items: LocalizedList | None, language: str, fallback: str = DEFAULT_LANGUAGE) -> list[str]:
    """Return a list of strings in the requested language, falling back to English."""
    if not items:
        return []
    if language in items:
        return list(items[language])
    if fallback in items:
        return list(items[fallback])
    return list(next(iter(items.values())))


class AreaName(str, Enum):
    """Body areas that have an examination endpoint."""

    EYES = "eyes"
    FACE = "face"
    MOUTH = "mouth"
    HANDS = "hands"
    MOVEMENT = "movement"
    POSTURE = "posture"
    SKIN = "skin"


class CameraSetup(str, Enum):
    """What must be visible in the camera for a step."""

    FACE = "face"
    FACE_PROFILE = "face_profile"
    BOTH_EYES_CLOSE_UP = "both_eyes_close_up"
    MOUTH_CLOSE_UP = "mouth_close_up"
    BOTH_HANDS = "both_hands"
    ONE_HAND_CLOSE_UP = "one_hand_close_up"
    FINGERS_CLOSE_UP = "fingers_close_up"
    UPPER_BODY_SEATED = "upper_body_seated"
    FULL_BODY_STANDING = "full_body_standing"
    FULL_BODY_SIDE = "full_body_side"
    FULL_BODY_BACK = "full_body_back"
    FULL_BODY_MOVING = "full_body_moving"
    SKIN_CLOSE_UP = "skin_close_up"


class CompletionType(str, Enum):
    """Kinds of step completion criteria."""

    VALID_FRAMES = "valid_frames"
    EVENTS = "events"
    STATE = "state"


class CompletionCriteria(BaseModel):
    """Condition that ends a protocol step."""

    model_config = ConfigDict(extra="forbid")

    type: CompletionType
    min_valid_frames: int = Field(ge=1)
    event: str | None = None
    min_events: int | None = Field(default=None, ge=1)
    state: str | None = None
    complete_on_max_duration: bool = False
    description: str

    @model_validator(mode="after")
    def check_type_fields(self) -> "CompletionCriteria":
        """Ensure event and state criteria carry their required fields."""
        if self.type == CompletionType.EVENTS and (self.event is None or self.min_events is None):
            raise ValueError("events criteria require 'event' and 'min_events'")
        if self.type == CompletionType.STATE and self.state is None:
            raise ValueError("state criteria require 'state'")
        return self


class TimedPrompt(BaseModel):
    """An instruction shown at a given number of seconds into a running step."""

    model_config = ConfigDict(extra="forbid")

    at_seconds: float = Field(ge=0.0)
    text: LocalizedText


class ProtocolStep(BaseModel):
    """One patient-facing step of an examination protocol."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    step_number: int = Field(ge=1)
    title: LocalizedText
    instruction: LocalizedText
    tips: LocalizedList = Field(default_factory=dict)
    safety_note: LocalizedText | None = None
    camera_setup: CameraSetup
    get_ready_seconds: int = Field(default=3, ge=0, le=10)
    min_duration_seconds: float = Field(gt=0.0)
    max_duration_seconds: float = Field(gt=0.0)
    completion_criteria: CompletionCriteria
    active_agents: list[str] = Field(default_factory=list)
    requires_patient_ready: bool = False
    skippable: bool = False
    repeatable: bool = False
    max_repetitions: int = Field(default=1, ge=1)
    allow_motion: bool = False
    captures_colour_reference: bool = False
    prompts: list[TimedPrompt] = Field(default_factory=list)
    repeat_guidance: LocalizedList = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def check_identifier(cls, value: str) -> str:
        """Reject identifiers containing the repetition separator."""
        if "__" in value:
            raise ValueError("step ids must not contain a double underscore")
        return value

    @field_validator("title", "instruction")
    @classmethod
    def require_english(cls, value: LocalizedText) -> LocalizedText:
        """Require English text for titles and instructions."""
        if DEFAULT_LANGUAGE not in value:
            raise ValueError("English ('en') text is required")
        return value

    @model_validator(mode="after")
    def check_consistency(self) -> "ProtocolStep":
        """Validate durations and repetition settings."""
        if self.max_duration_seconds < self.min_duration_seconds:
            raise ValueError(f"step '{self.id}': max_duration_seconds must be >= min_duration_seconds")
        if not self.repeatable and self.max_repetitions != 1:
            raise ValueError(f"step '{self.id}': only repeatable steps may have max_repetitions > 1")
        return self


class AreaProtocol(BaseModel):
    """Full examination protocol for one body area."""

    model_config = ConfigDict(extra="forbid")

    area: AreaName
    version: str
    title: LocalizedText
    description: LocalizedText
    steps: list[ProtocolStep]

    @model_validator(mode="after")
    def check_steps(self) -> "AreaProtocol":
        """Ensure step numbers are sequential and ids are unique."""
        identifiers = [step.id for step in self.steps]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"protocol '{self.area.value}' has duplicate step ids")
        expected_numbers = list(range(1, len(self.steps) + 1))
        actual_numbers = [step.step_number for step in self.steps]
        if actual_numbers != expected_numbers:
            raise ValueError(f"protocol '{self.area.value}' step numbers must be 1..{len(self.steps)} in order")
        return self

    @property
    def total_steps(self) -> int:
        """Number of steps in the protocol."""
        return len(self.steps)

    def step_by_id(self, step_id: str) -> ProtocolStep | None:
        """Return the step with the given id, if present."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None


class ProtocolStepView(BaseModel):
    """Localised, client-facing description of a protocol step."""

    id: str
    step_number: int
    total_steps: int
    title: str
    instruction: str
    tips: list[str]
    safety_note: str | None
    camera_setup: CameraSetup
    get_ready_seconds: int
    min_duration_seconds: float
    max_duration_seconds: float
    requires_patient_ready: bool
    skippable: bool
    repeatable: bool
    max_repetitions: int


class ProtocolResponse(BaseModel):
    """Protocol listing returned by the protocols endpoint."""

    area: AreaName
    version: str
    language: str
    title: str
    description: str
    total_steps: int
    steps: list[ProtocolStepView]


def build_step_view(step: ProtocolStep, total_steps: int, language: str) -> ProtocolStepView:
    """Create a localised view of a protocol step."""
    return ProtocolStepView(
        id=step.id,
        step_number=step.step_number,
        total_steps=total_steps,
        title=localize(step.title, language) or step.id,
        instruction=localize(step.instruction, language) or "",
        tips=localize_list(step.tips, language),
        safety_note=localize(step.safety_note, language),
        camera_setup=step.camera_setup,
        get_ready_seconds=step.get_ready_seconds,
        min_duration_seconds=step.min_duration_seconds,
        max_duration_seconds=step.max_duration_seconds,
        requires_patient_ready=step.requires_patient_ready,
        skippable=step.skippable,
        repeatable=step.repeatable,
        max_repetitions=step.max_repetitions,
    )


def build_protocol_response(protocol: AreaProtocol, language: str) -> ProtocolResponse:
    """Create a localised protocol listing."""
    return ProtocolResponse(
        area=protocol.area,
        version=protocol.version,
        language=language,
        title=localize(protocol.title, language) or protocol.area.value,
        description=localize(protocol.description, language) or "",
        total_steps=protocol.total_steps,
        steps=[build_step_view(step, protocol.total_steps, language) for step in protocol.steps],
    )
