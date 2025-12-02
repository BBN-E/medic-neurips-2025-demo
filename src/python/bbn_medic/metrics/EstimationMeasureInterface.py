from abc import ABC, abstractmethod

"""
Interface for calculating the performance of an estimator (e.g., confidence estimator)
"""


class EstimationMeasureInterface(ABC):

    @abstractmethod
    def calculate_mean_square_error(self, estimations: list):
        pass
