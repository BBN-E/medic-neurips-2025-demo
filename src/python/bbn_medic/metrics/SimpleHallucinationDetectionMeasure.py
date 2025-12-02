from bbn_medic.common import Hallucination
from bbn_medic.metrics.DetectionMeasureInterface import DetectionMeasureInterface


class SimpleHallucinationDetectionMeasure(DetectionMeasureInterface):
    '''
    Implements a simple hallucination detection measure, which only looks at whether a chatbot answer contains a hallucination or not.
    In other words, it only tracks whether an answer's reference (e.g. the truth data) has one or more hallucinations and checks whether
    one or more hallucinations were reported for that answer. It doesn't check multiple hallucinations or try to verify/validate that
    the hallucinations reported match the reference (truth data) in any way.
    '''

    def __init__(self, reference_detections: list[Hallucination] = None):
        self.reference_detections = reference_detections
        self.reference_answers_with_hallucinations = {}
        for detection in reference_detections:
            if detection.harm_level is not None and detection.harm_level != "none":
                self.reference_answers_with_hallucinations[detection.answer_id] = detection

    def calculate_precision_recall(self, detections: list[Hallucination]):
        precision = 0
        recall = 0
        num_correct_detections = 0
        num_false_alarms = 0
        answer_ids_with_detected_hallucinations = set()
        for detection in detections:
            if detection.harm_level is not None and detection.harm_level != "none":
                if detection.answer_id in self.reference_answers_with_hallucinations:
                    if detection.answer_id not in answer_ids_with_detected_hallucinations:
                        num_correct_detections += 1
                        answer_ids_with_detected_hallucinations.add(detection.answer_id)
                else:
                    num_false_alarms += 1
        precision = num_correct_detections / (num_correct_detections + num_false_alarms)
        recall = len(answer_ids_with_detected_hallucinations) / len(self.reference_answers_with_hallucinations)
        return {"simple_precision": precision, "simple_recall": recall}
