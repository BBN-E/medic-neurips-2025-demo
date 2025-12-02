from pathlib import Path

from bbn_medic.utils.logging_utils import setup_logger
from bbn_medic.generation.PromptGenerator import PromptGenerator
from bbn_medic.llms import ModelFactory

if __name__ == "__main__":
    setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--styles_file", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--model_temp", required=True, type=float)
    parser.add_argument("--num_prompts", required=True, type=int)
    args = parser.parse_args()
    model_instance = ModelFactory.load(args.model_path)

    PromptGenerator.restyle_patient_expression(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        styles_file=Path(args.styles_file),
        model_instance=model_instance,
        model_temp=args.model_temp,
        num_prompts=args.num_prompts
    )
