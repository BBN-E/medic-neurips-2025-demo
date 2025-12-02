import json
from bbn_medic.metrics.ConfidenceEstimationMeasure import ConfidenceEstimationMeasure
from bbn_medic.io.io_utils import fopen, JSONLGenerator

class ConfidenceEstimationMetrics:

    def compute(reference_confidences_file,
                estimated_confidences_file,
                output_file,
                compute_root_mean_square_error=True):

        reference_confidences = list(JSONLGenerator.read(reference_confidences_file))
        confidence_estimation_measure_computer = ConfidenceEstimationMeasure(reference_confidences)

        estimated_confidences = list(JSONLGenerator.read(estimated_confidences_file))

        results = {}
        if compute_root_mean_square_error:
            root_mean_square_error = confidence_estimation_measure_computer.calculate_root_mean_square_error(estimated_confidences)
            results = {**results, **root_mean_square_error}

        with open(output_file, "w") as g:
            g.write(json.dumps(results) + "\n")
