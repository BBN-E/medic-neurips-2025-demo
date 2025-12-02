from abc import ABC, abstractmethod

"""
Interface for calculating the performance of a detector (e.g., hallucination detector or omission detector).
"""


class DetectionMeasureInterface(ABC):

    @abstractmethod
    def calculate_precision_recall(self, detections: list):
        pass
