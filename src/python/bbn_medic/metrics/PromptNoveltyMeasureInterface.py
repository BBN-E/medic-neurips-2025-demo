from abc import ABC, abstractmethod

from bbn_medic.common import Prompt

"""
Interface for calculating prompt novelty.
"""
class PromptNoveltyMeasureInterface(ABC):

    @abstractmethod
    def calculate_diversity(self, prompts: list[Prompt]):
        pass
