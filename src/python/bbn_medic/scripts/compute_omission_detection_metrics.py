import argparse
import os

from bbn_medic.metrics.OmissionDetectionMeasure import OmissionDetectionMeasure
from bbn_medic.io.io_utils import JSONLGenerator
from bbn_medic.common.HarmLevels import HarmLevels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detected_omissions_file', type=str, required=True, help='Input jsonl file with detected omissions')
    parser.add_argument('--reference_omissions_file', type=str, required=True, help='Input jsonl file with reference (ground truth) omissions')
    parser.add_argument('--output_omission_detection_scores_file', type=str, required=True, help='Output jsonl file containing precision/recall scores per detection threshold setting')
    parser.add_argument('--harm_levels', nargs='+', help='An optional list of harm levels such as ["none", "low", "medium", etc.]. If missing, a default list of harm levels will be used')
    parser.add_argument('--confidence_thresholds', nargs='+', help='An optional list of confidence thresholds. If missing, the range [0, 1] (inclusive) at 0.1 increments will be used.')
    args = parser.parse_args()

    reference_omissions = list(JSONLGenerator.read(args.reference_omissions_file))
    omission_detection_measure_computer = OmissionDetectionMeasure(reference_omissions)
    detected_omissions = list(JSONLGenerator.read(args.detected_omissions_file))

    if args.harm_levels is None:
        harm_levels = HarmLevels().sorted_list_of_harm_strings()
    else:
        harm_levels = args.harm_levels

    if args.confidence_thresholds is None:
        confidence_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        confidence_thresholds = [float(x) for x in args.confidence_thresholds]

    all_scores = []
    os.makedirs(os.path.dirname(args.output_omission_detection_scores_file), exist_ok=True)
    with open(args.output_omission_detection_scores_file, "w") as g:
        for harm_level in harm_levels:
            for confidence_threshold in confidence_thresholds:
                precision_recall_f1 = omission_detection_measure_computer.calculate_precision_recall(detected_omissions, harm_level_threshold=harm_level, confidence_threshold=confidence_threshold)
                all_scores.append({"harm_level_threshold": harm_level, "confidence_threshold": confidence_threshold, "scores": precision_recall_f1})
        all_scores = sorted(all_scores, key=lambda x: x["scores"]["f1_score"], reverse=True)
        for score in all_scores:
            g.write(f"{score}\n")


if __name__ == "__main__":
    main()
