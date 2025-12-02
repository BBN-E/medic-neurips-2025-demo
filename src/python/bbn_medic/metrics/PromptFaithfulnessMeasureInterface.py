from abc import ABC, abstractmethod

from bbn_medic.common import Prompt

"""
Interface for calculating prompt faithfulness.
"""


class PromptFaithfulnessMeasureInterface(ABC):

    @abstractmethod
    def calculate_faithfulness(self, prompts: list[Prompt], source_prompts: list[Prompt]):
        pass
