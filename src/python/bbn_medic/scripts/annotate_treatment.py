import argparse
import json

from pathlib import Path

from bbn_medic.llms import ModelFactory
from bbn_medic.common.Prompt import Prompt
from bbn_medic.generation.AnswerGenerator import AnswerGenerator

from bbn_medic.detection.classifications.BaselineTreatmentClassifier import BaselineTreatmentClassifier



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ")

    parser.add_argument('--prompts_file', type=str, required=True, help='Input jsonl file with prompts to answer (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--answers_file', type=str, required=True, help="Path to answers jsonl (or jsonl.gz or jsonl.bz2) file")

    parser.add_argument('--output_file', type=str, required=True, help="Path to output manage jsonl (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument('--model_temp_for_treatment_classification', type=float, help='Temperature setting for the LLM used to detect responses that suggest to "manage"', default=0.1)

    args = parser.parse_args()

    # load model
    model_instance = ModelFactory.load(args.model_path)

    BaselineTreatmentClassifier.detect(prompts_input_file=Path(args.prompts_file),
                                         answers_input_file=Path(args.answers_file),
                                         output_file=Path(args.output_file),
                                         model_instance=model_instance,
                                         model_temp=args.model_temp_for_treatment_classification,
                                        )

