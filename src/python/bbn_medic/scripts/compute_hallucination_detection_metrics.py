import argparse
import os

from bbn_medic.metrics.HallucinationDetectionMeasure import HallucinationDetectionMeasure, SegmentBasedHallucinationDetectionMeasure
from bbn_medic.io.io_utils import JSONLGenerator
from bbn_medic.common.HarmLevels import HarmLevels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detected_hallucinations_file', type=str, required=True, help='Input jsonl file with detected hallucinations')
    parser.add_argument('--reference_hallucinations_file', type=str, required=True, help='Input jsonl file with reference (ground truth) hallucinations')
    parser.add_argument('--output_hallucination_detection_scores_file', type=str, required=True, help='Output jsonl file containing precision/recall scores per detection threshold setting')
    parser.add_argument('--harm_levels', nargs='+', help='An optional list of harm levels such as "none" "low" "medium" etc. If missing, a default list of harm levels will be used')
    parser.add_argument('--confidence_thresholds', nargs='+', help='An optional list of confidence thresholds. If missing, the range [0, 1] (inclusive) at 0.1 increments will be used.')
    parser.add_argument('--segment_level', type=bool, default=False, help='If True, hallucination detection metrics will be computed at the segment level instead of the answer level')
    args = parser.parse_args()

    reference_hallucinations = list(JSONLGenerator.read(args.reference_hallucinations_file))
    if args.segment_level:
        hallucination_detection_measure_computer = SegmentBasedHallucinationDetectionMeasure(reference_hallucinations)
    else:
        hallucination_detection_measure_computer = HallucinationDetectionMeasure(reference_hallucinations)
    detected_hallucinations = list(JSONLGenerator.read(args.detected_hallucinations_file))

    if args.harm_levels is None:
        harm_levels = HarmLevels().sorted_list_of_harm_strings()
    else:
        harm_levels = args.harm_levels

    if args.confidence_thresholds is None:
        confidence_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        confidence_thresholds = [float(x) for x in args.confidence_thresholds]

    all_scores = []
    os.makedirs(os.path.dirname(args.output_hallucination_detection_scores_file), exist_ok=True)
    with open(args.output_hallucination_detection_scores_file, "w") as g:
        for harm_level in harm_levels:
            for confidence_threshold in confidence_thresholds:
                precision_recall_f1 = hallucination_detection_measure_computer.calculate_precision_recall(detected_hallucinations, harm_level_threshold=harm_level, confidence_threshold=confidence_threshold)
                all_scores.append({"harm_level_threshold": harm_level, "confidence_threshold": confidence_threshold, "scores": precision_recall_f1})
        all_scores = sorted(all_scores, key=lambda x: x["scores"]["f1_score"], reverse=True)
        for score in all_scores:
            g.write(f"{score}\n")


if __name__ == "__main__":
    main()
