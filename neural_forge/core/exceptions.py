"""Application exception hierarchy."""


class NeuralForgeError(Exception):
    """Base class for all application errors."""

    default_code: str = "neural_forge_error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code


class ConfigurationError(NeuralForgeError):
    """Raised when a configuration file or setting is missing or invalid."""

    default_code = "configuration_error"


class ProtocolError(ConfigurationError):
    """Raised when an examination protocol definition is invalid."""

    default_code = "protocol_error"


class ModelNotAvailableError(NeuralForgeError):
    """Raised when model weights or task files are missing or cannot be loaded."""

    default_code = "model_not_available"


class ModelInferenceError(NeuralForgeError):
    """Raised when a model fails during inference."""

    default_code = "model_inference_error"


class SessionNotFoundError(NeuralForgeError):
    """Raised when a session id does not exist."""

    default_code = "session_not_found"


class SessionStateError(NeuralForgeError):
    """Raised when an operation is not valid for the current session state."""

    default_code = "session_state_error"


class ConsentRequiredError(NeuralForgeError):
    """Raised when a session is requested without explicit consent."""

    default_code = "consent_required"


class FrameRejectedError(NeuralForgeError):
    """Raised when an incoming frame is oversized, malformed or cannot be decoded."""

    default_code = "frame_rejected"


class InvalidMessageError(NeuralForgeError):
    """Raised when a WebSocket message cannot be parsed or validated."""

    default_code = "invalid_message"


class LLMServiceError(NeuralForgeError):
    """Raised when a vLLM server cannot be reached or returns an error."""

    default_code = "llm_service_error"

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseValidationError(NeuralForgeError):
    """Raised when an LLM response does not match the required JSON schema after repair attempts."""

    default_code = "llm_response_invalid"


class ExaminationStopped(Exception):
    """Control-flow signal raised when the patient stops the examination."""


class StepSkipped(Exception):
    """Control-flow signal raised when the patient skips a skippable step."""


class QualityTimeout(Exception):
    """Control-flow signal raised when acceptable camera quality was not reached in time."""

    def __init__(self, issues: list[str], guidance: list[str]) -> None:
        super().__init__("Camera quality was not acceptable within the allowed time.")
        self.issues = issues
        self.guidance = guidance
