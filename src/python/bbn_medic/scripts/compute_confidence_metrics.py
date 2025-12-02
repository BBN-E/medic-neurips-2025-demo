import argparse

from bbn_medic.metrics.ConfidenceEstimationMetrics import ConfidenceEstimationMetrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_confidences_file', type=str, help='Input jsonl file containing reference confidences')
    parser.add_argument('--estimated_confidences_file', type=str, help='Input jsonl file containing estimated confidences')
    parser.add_argument('--output_file', type=str, help='Output jsonl file containing confidence estimation metrics')

    args = parser.parse_args()

    ConfidenceEstimationMetrics.compute(
        reference_confidences_file=args.reference_confidences_file,
        estimated_confidences_file=args.estimated_confidences_file,
        output_file=args.output_file,
    )

if __name__ == "__main__":
    main()
