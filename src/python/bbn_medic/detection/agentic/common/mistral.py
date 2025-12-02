import autogen
import enum

class TaskTypeToAgentName(enum.Enum):
    Hallucination = "ErrorDetector"
    Omission = "OmissionDetector"


def build_automatic_user(detector_name):
    return autogen.UserProxyAgent(
        name="AutomatedUser",
        system_message=(
            f"You are an automated user providing the original answer to the {detector_name}, then remaining silent."
        ),
        code_execution_config=False,
        human_input_mode="NEVER",
    )
