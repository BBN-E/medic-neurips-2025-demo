import argparse

from pathlib import Path

from bbn_medic.detection.hallucinations.QABaselineHallucinationDetector import QABaselineHallucinationDetector
from bbn_medic.detection.hallucinations.QASegmentLevelBaselineHallucinationDetector import QASegmentLevelBaselineHallucinationDetector
from bbn_medic.llms import ModelFactory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, required=True, help='Input jsonl file with prompts for the given answers (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--answers_file', type=str, required=True, help='Input jsonl file with answers (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--output_file', type=str, required=True, help="Path to output jsonl (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/nvidia-Llama3-ChatQA-1.5-8B")
    parser.add_argument('--model_temp', type=float, help='Temperature setting for the LLM used to detect hallucinations',
                        default=1.0)
    parser.add_argument("--include_harmless", type=bool, help="whether to include hallucinations where harm to patient "
                                                              "is assessed to be 'none' (default is False)",
                        default=False),
    parser.add_argument("--segment_level_detection", type=bool, default=False,
                        help="If true, it will perform hallucination detection at the level of segment, assuming that the input answers have been segmented")
    parser.add_argument('--window_length', type=int, default=1, help='Number of consecutive segments to use in detection')
    parser.add_argument('--context_length', type=int, help='Number of left (and right) segments to use as context in the detection')
    parser.add_argument('--compression', type=str, choices=['4bit', '8bit'], help="Set to 4bit or 8bit for dynamically compressing the LLM")

    args = parser.parse_args()

    # load model
    model_instance = ModelFactory.load(args.model_path, compression=args.compression)

    if args.segment_level_detection:
        QASegmentLevelBaselineHallucinationDetector.detect(prompts_input_file=Path(args.prompts_file),
                                                         answers_input_file=Path(args.answers_file),
                                                         output_file=Path(args.output_file),
                                                         model_instance=model_instance,
                                                         model_temp=args.model_temp,
                                                         include_harmless=args.include_harmless,
                                                         window_length=args.window_length,
                                                         context_length=args.context_length)
    else:
        QABaselineHallucinationDetector.detect(prompts_input_file=Path(args.prompts_file),
                                             answers_input_file=Path(args.answers_file),
                                             output_file=Path(args.output_file),
                                             model_instance=model_instance,
                                             model_temp=args.model_temp,
                                             include_harmless=args.include_harmless)
