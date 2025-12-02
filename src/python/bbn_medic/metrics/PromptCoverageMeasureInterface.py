import typing
from abc import ABC, abstractmethod

from bbn_medic.common.Prompt import Prompt

"""
Interface for calculating prompt coverage.
"""


class PromptCoverageMeasureInterface(ABC):

    @abstractmethod
    def calculate_coverage(self, prompts: typing.List[Prompt]):
        pass
