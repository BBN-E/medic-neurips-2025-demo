import sys
from collections import defaultdict

from bbn_medic.common import Hallucination
from bbn_medic.metrics.DetectionMeasureInterface import DetectionMeasureInterface
from bbn_medic.common.HarmLevels import HarmLevels


class HallucinationDetectionMeasure(DetectionMeasureInterface):
    '''
    Implements a hallucination detection measure, which simply compares the count of hallucinations reported in the reference
    to the count of hallucinations reported in the input, for each chatbot answer. It doesn't try to verify/validate that the
    hallucinations reported match the reference (truth data) in any way.
    '''

    def __init__(self, reference_detections: list[Hallucination] = None):
        self.reference_detections = reference_detections
        self.reference_answers_with_hallucinations = defaultdict(list)
        self.number_of_reference_hallucations_per_answer = defaultdict(lambda: 0)
        self.total_reference_hallucations = 0
        for detection in reference_detections:
            if detection.harm_level is not None and detection.harm_level != "none":
                self.reference_answers_with_hallucinations[detection.answer_id].append(detection)
                self.number_of_reference_hallucations_per_answer[detection.answer_id] += 1
                self.total_reference_hallucations += 1

    def calculate_precision_recall(self, detections: list[Hallucination], harm_level_threshold="very_low", confidence_threshold=0.0):
        precision = 0
        recall = 0
        f1_score = 0
        num_correct_detections = 0
        num_false_alarms = 0
        answer_ids_with_detected_hallucinations = set()
        number_of_hallucations_per_answer = defaultdict(lambda: 0)
        for detection in detections:
            if detection.harm_level is not None and \
               HarmLevels().harm_string_to_harm_level(detection.harm_level) >= HarmLevels().harm_string_to_harm_level(harm_level_threshold) and \
               detection.confidence >= confidence_threshold:
                if detection.answer_id in self.reference_answers_with_hallucinations:
                    number_of_hallucations_per_answer[detection.answer_id] += 1
                else:
                    num_false_alarms += 1
        for answer_id in self.number_of_reference_hallucations_per_answer:
            if number_of_hallucations_per_answer.get(answer_id, 0) >= self.number_of_reference_hallucations_per_answer[answer_id]:
                num_false_alarms += number_of_hallucations_per_answer[answer_id] - self.number_of_reference_hallucations_per_answer[answer_id]
                num_correct_detections += self.number_of_reference_hallucations_per_answer[answer_id]
            else:
                num_correct_detections += number_of_hallucations_per_answer.get(answer_id, 0)
        precision = num_correct_detections / (num_correct_detections + num_false_alarms) if num_correct_detections + num_false_alarms > 0 else 0
        recall = num_correct_detections / self.total_reference_hallucations
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return {"precision": precision, "recall": recall, "f1_score": f1_score}


class SegmentBasedHallucinationDetectionMeasure(DetectionMeasureInterface):
    def __init__(self, reference_detections: list[Hallucination] = None):
        self.reference_detections = reference_detections
        self.reference_segments_with_hallucinations = set()
        for detection in reference_detections:
            if detection.harm_level is not None and detection.harm_level != "none":
                self.reference_segments_with_hallucinations.add((detection.answer_id, detection.segment_id))

    def calculate_precision_recall(self, detections: list[Hallucination], harm_level_threshold="very_low", confidence_threshold=0.0):
        precision = 0
        recall = 0
        f1_score = 0
        correctly_detected_segments = set()
        wrongly_detected_segments = set()
        for detection in detections:
            if detection.harm_level is not None and \
               HarmLevels().harm_string_to_harm_level(detection.harm_level) >= HarmLevels().harm_string_to_harm_level(harm_level_threshold) and \
               detection.confidence >= confidence_threshold:
                if (detection.answer_id, detection.segment_id) in self.reference_segments_with_hallucinations:
                    correctly_detected_segments.add((detection.answer_id, detection.segment_id))
                else:
                    wrongly_detected_segments.add((detection.answer_id, detection.segment_id))
        precision = len(correctly_detected_segments) / (len(correctly_detected_segments) + len(wrongly_detected_segments))
        recall = len(correctly_detected_segments) / len(self.reference_segments_with_hallucinations)
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return {"precision": precision, "recall": recall, "f1_score": f1_score}
