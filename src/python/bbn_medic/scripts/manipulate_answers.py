import argparse
import json

from pathlib import Path

from bbn_medic.llms import ModelFactory
from bbn_medic.generation.AnswerManipulator import AnswerManipulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Input jsonl file with chatbot answers (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--output_file', type=str, required=True, help="Path to output jsonl with modified answers (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument('--hallucinations_reference_file', type=str, help="Path to output jsonl with the generated hallucinations (if hallucinations are requested)")
    parser.add_argument('--omissions_reference_file', type=str, help="Path to output jsonl with the generated omissions (if omissions are requested)")
    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ")
    parser.add_argument('--hallucinations_proportion', type=float, help='Proportion of answers or segments to manipulate for hallucination insertion', default=0.5)
    parser.add_argument('--omissions_proportion', type=float, help='Proportion of answers to manipulate for omission insertion', default=0.5)
    parser.add_argument('--model_temp', type=float, help='Temperature setting for the LLM',
                        default=1.0)
    parser.add_argument('--segment_level', type=bool, default=False, help='Manipulate answers at the segment level (for hallucinations)')
    parser.add_argument('--replace_statements_with_hallucinations', type=bool, default=False, help='If True, hallucinated segments will *replace* existing segments in the Answer. The default is to just add the hallucinated segments into the Answer.')

    args = parser.parse_args()

    # load model
    model_instance = ModelFactory.load(args.model_path)

    print(f"ARGS: {args}")

    AnswerManipulator.generate(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        hallucinations_proportion=args.hallucinations_proportion,
        omissions_proportion=args.omissions_proportion,
        model_instance=model_instance,
        model_temp=args.model_temp,
        hallucinations_reference_file=args.hallucinations_reference_file,
        omissions_reference_file=args.omissions_reference_file,
        segment_level=args.segment_level,
        replace_statements_with_hallucinations=args.replace_statements_with_hallucinations)
