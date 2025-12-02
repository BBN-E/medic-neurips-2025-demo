import argparse
import json

from pathlib import Path

from bbn_medic.llms import ModelFactory
from bbn_medic.generation.DesireGenerator import DesireGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--desires_file', type=str, required=True, help='Input jsonl file with desires (can be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--disorders_file', type=str, default=None, help='Input jsonl file with disorders (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--default_disorder', type=str, help='Default disorder text')
    parser.add_argument('--output_file', type=str, required=True, help="Path to output jsonl (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ")
    parser.add_argument('--num_new_desires', type=int, help='Number of new desires to generate for each initial desire', default=2)
    parser.add_argument('--model_temp', type=float, help='Temperature setting for the LLM', default=1.0)
    parser.add_argument('--generation_prompts_file', type=str, help='Path to jsonl file containing alternative prompts', default=None)
    parser.add_argument('--max_generation_length', type=int, help='Only keep generations whose length does not exceed this', default=None)

    args = parser.parse_args()

    # load model
    model_instance = ModelFactory.load(args.model_path)
    disorders_file = None if args.disorders_file is None or args.disorders_file == 'None' else Path(args.disorders_file)

    DesireGenerator.generate(
        input_file=Path(args.desires_file),
        disorders_file=disorders_file,
        default_disorder=args.default_disorder,
        output_file=Path(args.output_file),
        model_instance=model_instance,
        num_desires=args.num_new_desires,
        model_temp=args.model_temp,
        desire_generation_prompts_file=args.generation_prompts_file,
        max_generation_length=args.max_generation_length
    )
