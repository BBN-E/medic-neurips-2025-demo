from bbn_medic.common import Omission
from bbn_medic.metrics.DetectionMeasureInterface import DetectionMeasureInterface
from bbn_medic.common.HarmLevels import HarmLevels


class OmissionDetectionMeasure(DetectionMeasureInterface):
    '''
    Implements an omission detection measure, which only looks at whether a chatbot answer contains an omission or not.
    It doesn't try to match the omissions in the reference (truth data) with the detected ones. This is consistent with
    the description of the omission detection measure in the ARPA-H CARE BAA:
    What percentage of LLM outputs with missing or omitted information is detected with the developed method relative to
    what is detected by human expert?
    '''
    def __init__(self, reference_detections: list[Omission] = None):
        self.reference_detections = reference_detections
        self.reference_answers_with_omissions = {}
        for detection in reference_detections:
            if detection.harm_level is not None and detection.harm_level != "none":
                self.reference_answers_with_omissions[detection.answer_id] = detection

    def calculate_precision_recall(self, detections: list[Omission], harm_level_threshold="very_low", confidence_threshold=0.0):
        precision = 0
        recall = 0
        f1_score = 0
        num_correct_detections = 0
        num_false_alarms = 0
        answer_ids_with_correctly_detected_omissions = set()
        for detection in detections:
            if detection.harm_level is not None and \
               HarmLevels().harm_string_to_harm_level(detection.harm_level) >= HarmLevels().harm_string_to_harm_level(harm_level_threshold) and \
               detection.confidence >= confidence_threshold:
                if detection.answer_id in self.reference_answers_with_omissions:
                    if detection.answer_id not in answer_ids_with_correctly_detected_omissions:
                        num_correct_detections += 1
                        answer_ids_with_correctly_detected_omissions.add(detection.answer_id)
                else:
                    num_false_alarms += 1
        precision = num_correct_detections / (num_correct_detections + num_false_alarms) if num_correct_detections + num_false_alarms > 0 else 0
        recall = len(answer_ids_with_correctly_detected_omissions) / len(self.reference_answers_with_omissions)
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return {"precision": precision, "recall": recall, "f1_score": f1_score}
