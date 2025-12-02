import argparse
import json

from pathlib import Path

from bbn_medic.llms import ModelFactory
from bbn_medic.common.Prompt import Prompt
from bbn_medic.generation.PromptGenerator import PromptGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, help='Input jsonl file with prompts to diversify (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--styles_file', type=str, help='Input jsonl file with styles (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--disorders_file', type=str, default=None, help='Input jsonl file with disorders (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--default_disorder', type=str, help='Default disorder text')
    parser.add_argument('--output_file', type=str, help="Path to output jsonl file (can also be jsonl.gz or jsonl.bz2)")
    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ")
    parser.add_argument('--num_prompts', type=int, help='Number of prompts to generate for each input record', default=2)
    parser.add_argument('--model_temp', type=float, help='Temperature setting for the LLM', default=1.0)
    parser.add_argument('--synthetic_prefixes', type=str, help='String containing list of synthetic history prefixes, separated by commas')
    parser.add_argument('--prompt_to_prompt_generation_prompts_file', type=str, help='Path to jsonl file containing alternative prompts for prompt-to-prompt generation', default=None)
    parser.add_argument('--preamble_text', type=str, help='String containing preamble text, with \"{disorder_text}\" as an expected variable for the disorder (None will default to \"My question is related to {disorder_text}\")', default=None)
    args = parser.parse_args()

    # load model and set parameters
    model_instance = ModelFactory.load(args.model_path)
    disorders_file = None if args.disorders_file is None or args.disorders_file == 'None' else Path(args.disorders_file)
    if args.synthetic_prefixes is not None:
        synthetic_prefixes = args.synthetic_prefixes.split(',')
    else:
        synthetic_prefixes = None

    # If file for prompts to generate prompts exists, don't include the default prompt
    default_prompt=True if args.prompt_to_prompt_generation_prompts_file is None else False

    PromptGenerator.generate(
        input_file=args.prompts_file,
        disorders_file=disorders_file,
        default_disorder=args.default_disorder,
        output_file=args.output_file,
        styles_file=args.styles_file,
        model_instance=model_instance,
        num_prompts=args.num_prompts,
        model_temp=args.model_temp,
        use_default_prompt=default_prompt,
        synthetic_prefixes=synthetic_prefixes,
        prompt_to_prompt_generation_prompts_file=args.prompt_to_prompt_generation_prompts_file,
        preamble_text=args.preamble_text)
