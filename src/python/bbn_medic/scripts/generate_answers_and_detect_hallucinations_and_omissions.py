import argparse
import json

from pathlib import Path

from bbn_medic.llms import ModelFactory
from bbn_medic.common.Prompt import Prompt
from bbn_medic.generation.AnswerGenerator import AnswerGenerator
from bbn_medic.detection.hallucinations.BaselineHallucinationDetector import BaselineHallucinationDetector
from bbn_medic.detection.omissions.BaselineOmissionDetector import BaselineOmissionDetector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ")

    # answers params
    parser.add_argument('--prompts_file', type=str, required=True, help='Input jsonl file with prompts to answer (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--answers_output_file', type=str, required=True, help="Path to output jsonl (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument('--num_answers', type=int, help='Number of answers to generate for each input prompt', default=2)
    parser.add_argument('--model_temp_for_answers', type=float, help='Temperature setting for the LLM', default=1.0)

    # hallucinations params
    parser.add_argument('--hallucinations_output_file', type=str, required=True, help="Path to output hallucinations jsonl (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument('--model_temp_for_hallucinations', type=float, help='Temperature setting for the LLM used to detect hallucinations', default=1.0)
    parser.add_argument("--include_harmless_hallucinations", type=bool, help="whether to include hallucinations where harm to patient "
                                                              "is assessed to be 'none' (default is False)", default=False)

    # omissions params
    parser.add_argument('--omissions_output_file', type=str, required=True, help="Path to output jsonl (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument('--model_temp_for_omissions', type=float, help='Temperature setting for the LLM used to detect omissions', default=1.0)
    parser.add_argument("--include_harmless_omissions", type=bool, help="whether to include omissions where harm to patient "
                                                              "is assessed to be 'none' (default is False)", default=False)

    args = parser.parse_args()

    # load model
    model_instance = ModelFactory.load(args.model_path)

    AnswerGenerator.generate(
        input_file=Path(args.prompts_file),
        output_file=Path(args.answers_output_file),
        model_instance=model_instance,
        num_answers=args.num_answers,
        model_temp=args.model_temp_for_answers)


    BaselineHallucinationDetector.detect(prompts_input_file=Path(args.prompts_file),
                                         answers_input_file=Path(args.answers_output_file),
                                         output_file=Path(args.hallucinations_output_file),
                                         model_instance=model_instance,
                                         model_temp=args.model_temp_for_hallucinations,
                                         include_harmless=args.include_harmless_hallucinations)

    BaselineOmissionDetector.detect(prompts_input_file=Path(args.prompts_file),
                                    answers_input_file=Path(args.answers_output_file),
                                    output_file=Path(args.omissions_output_file),
                                    model_instance=model_instance,
                                    model_temp=args.model_temp_for_omissions,
                                    include_harmless=args.include_harmless_omissions)
