from abc import ABC, abstractmethod
"""
Interface for calculating prompt scaling.
"""
class PromptScalingMeasureInterface(ABC):

    """
    Given a specific run of the system, go through the logs and calculate statistics on prompt speed. Average time per
    prompt is the bottom line, but we also want other information about the distribution and specifics which alert us
    to long-running outliers.
    """
    @abstractmethod
    def calculate_scaling(self, run_id):
        pass
