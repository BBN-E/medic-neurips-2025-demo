import argparse
import json

from pathlib import Path

from bbn_medic.llms import ModelFactory
from bbn_medic.common.Prompt import Prompt
from bbn_medic.generation.PromptGenerator import PromptGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str, help='Input jsonl file with desires or prompts to use as source material for generating prompts (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--disorders_file', type=str, default=None, help='Input jsonl file with disorders (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--default_disorder', type=str, help='Default disorder text')
    parser.add_argument('--output_file', required=True, type=str, help="Path to output jsonl (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ")
    parser.add_argument('--num_prompts', type=int, help='Number of prompts to generate for each input record', default=2)
    parser.add_argument('--model_temp', type=float, help='Temperature setting for the LLM',
                        default=1.0)
    parser.add_argument('--desire_to_prompt_generation_prompts_file', type=str, help='Path to jsonl file containing alternative prompts for desire-to-prompt generation', default=None)
    parser.add_argument('--prompt_to_prompt_generation_prompts_file', type=str, help='Path to jsonl file containing alternative prompts for prompt-to-prompt generation', default=None)
    parser.add_argument('--preamble_text', type=str, help='String containing preamble text, with \"{disorder_text}\" as an expected variable for the disorder (None will default to \"My question is related to {disorder_text}\")', default=None)
    parser.add_argument('--patient_expression_path', type=str, help='A file contains generated patient expressions', default=None)
    args = parser.parse_args()

    # load model
    model_instance = ModelFactory.load(args.model_path)
    disorders_file = None if args.disorders_file is None or args.disorders_file == 'None' else Path(args.disorders_file)

    if args.patient_expression_path is None:
        PromptGenerator.generate(
            input_file=Path(args.input_file),
            disorders_file=disorders_file,
            default_disorder=args.default_disorder,
            output_file=Path(args.output_file),
            model_instance=model_instance,
            num_prompts=args.num_prompts,
            model_temp=args.model_temp,
            desire_to_prompt_generation_prompts_file=args.desire_to_prompt_generation_prompts_file,
            prompt_to_prompt_generation_prompts_file=args.prompt_to_prompt_generation_prompts_file,
            preamble_text=args.preamble_text)
    else:
        PromptGenerator.generate_v2(
            input_file=Path(args.input_file),
            output_file=Path(args.output_file),
            desire_to_prompt_generation_prompts_file=args.desire_to_prompt_generation_prompts_file,
            model_instance=model_instance,
            model_temp=args.model_temp,
            num_prompts=args.num_prompts,
            patient_expression_path=Path(args.patient_expression_path)
        )
