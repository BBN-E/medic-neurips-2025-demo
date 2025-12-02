import math

from bbn_medic.common import Confidence
from bbn_medic.metrics.EstimationMeasureInterface import EstimationMeasureInterface


class ConfidenceEstimationMeasure(EstimationMeasureInterface):
    '''
    Implements an confidence estimation measure, which compares the human-provided confidence of the chatbot answer
    to the confidence automatically estimated
    '''
    def __init__(self, reference_estimations: list[Confidence] = None):
        self.reference_estimations = reference_estimations
        self.reference_answers_with_confidences = {}
        for estimation in reference_estimations:
                self.reference_answers_with_confidences[estimation.answer_id] = estimation

    def _mean_square_error(self, estimations: list[Confidence]):
        sum_square_errors = 0
        number_of_samples = 0
        for estimation in estimations:
            if estimation.answer_id in self.reference_answers_with_confidences:
                sum_square_errors += (estimation.confidence - self.reference_answers_with_confidences[estimation.answer_id].confidence)**2
                number_of_samples += 1
        mean_square_error = sum_square_errors / number_of_samples
        return mean_square_error

    def calculate_mean_square_error(self, estimations: list[Confidence]):
        mse = self._mean_square_error(estimations)
        return {"mean_square_error": mse}

    def calculate_root_mean_square_error(self, estimations: list[Confidence]):
        mse = self._mean_square_error(estimations)
        return {"root_mean_square_error": math.sqrt(mse)}
