import argparse
import json

from pathlib import Path

from bbn_medic.llms import ModelFactory
from bbn_medic.common.Prompt import Prompt
from bbn_medic.generation.DesireGenerator import DesireGenerator
from bbn_medic.generation.PromptGenerator import PromptGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # common params
    parser.add_argument('--disorders_file', type=str, default=None, help='Input jsonl file with disorders (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--default_disorder', type=str, help='Default disorder text')
    
    # desire params
    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ")
    parser.add_argument('--model_temp_for_desires', type=float, help='Temperature setting for the LLM desire generation', default=1.0)
    parser.add_argument('--input_desires_file', type=str, required=True, help='Input jsonl file with desires (can be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--desire_generation_prompts_file', type=str, help='Path to jsonl file containing alternative prompts', default=None)
    parser.add_argument('--output_desires_file', type=str, required=True, help="Path to output desires jsonl (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument('--num_new_desires', type=int, help='Number of new desires to generate for each initial desire', default=2)
    parser.add_argument('--max_generation_length', type=int, help='Only keep desire generations whose length does not exceed this', default=None)


    # standard prompt params
    parser.add_argument('--input_standard_prompts_source_file', required=True, type=str, help='Input jsonl file with desires or prompts to use as source material for generating prompts (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--output_standard_prompts_file', required=True, type=str, help="Path to output prompts jsonl (or jsonl.gz or jsonl.bz2) file", default=1.0)
    parser.add_argument('--num_standard_prompts', type=int, help='Number of prompts to generate for each input record', default=2)
    parser.add_argument('--desire_to_standard_prompt_generation_prompts_file', type=str, help='Path to jsonl file containing alternative prompts for desire-to-prompt generation', default=None)
    parser.add_argument('--prompt_to_standard_prompt_generation_prompts_file', type=str, help='Path to jsonl file containing alternative prompts for prompt-to-prompt generation', default=None)
    parser.add_argument('--model_temp_for_standard_prompts', type=float, help='Temperature setting for the LLM prompt generation', default=1.0)


    # diversified prompt params
    parser.add_argument('--input_diversified_prompts_source_file', required=True, type=str, help='Input jsonl file with desires or prompts to use as source material for generating prompts (can also be jsonl.gz or jsonl.bz2) (diversify step)')
    parser.add_argument('--styles_file', type=str, help='Input jsonl file with styles (can also be jsonl.gz or jsonl.bz2)')

    parser.add_argument('--output_diversified_prompts_file', required=True, type=str, help="Path to output prompts jsonl (or jsonl.gz or jsonl.bz2) file (diversify step)", default=1.0)
    parser.add_argument('--generation_diversified_prompts_file', type=str, help='Path to jsonl file containing alternative prompts (diversify step)', default=None)
    parser.add_argument('--num_diversified_prompts', type=int, help='Number of prompts to generate for each input record (diversify step)', default=2)
    parser.add_argument('--model_temp_for_diversified_prompts', type=float, help='Temperature setting for the LLM prompt generation', default=1.0)


    args = parser.parse_args()

    # load model
    model_instance = ModelFactory.load(args.model_path)
    disorders_file = None if args.disorders_file is None or args.disorders_file == 'None' else Path(args.disorders_file)


    DesireGenerator.generate(
        input_file=Path(args.input_desires_file),
        disorders_file=disorders_file,
        default_disorder=args.default_disorder,
        output_file=Path(args.output_desires_file),
        model_instance=model_instance,
        num_desires=args.num_new_desires,
        model_temp=args.model_temp_for_desires,
        desire_generation_prompts_file=args.desire_generation_prompts_file,
        max_generation_length=args.max_generation_length
    )

    PromptGenerator.generate(
        input_file=Path(args.input_standard_prompts_source_file),
        disorders_file=disorders_file,
        default_disorder=args.default_disorder,
        output_file=Path(args.output_standard_prompts_file),
        model_instance=model_instance,
        num_prompts=args.num_standard_prompts,
        model_temp=args.model_temp_for_standard_prompts,
        desire_to_prompt_generation_prompts_file=args.desire_to_standard_prompt_generation_prompts_file,
        prompt_to_prompt_generation_prompts_file=args.prompt_to_standard_prompt_generation_prompts_file)

    PromptGenerator.generate(
        input_file=Path(args.input_diversified_prompts_source_file),
        disorders_file=disorders_file,
        default_disorder=args.default_disorder,
        output_file=Path(args.output_diversified_prompts_file),
        styles_file=args.styles_file,
        model_instance=model_instance,
        num_prompts=args.num_diversified_prompts,
        model_temp=args.model_temp_for_diversified_prompts,
        use_default_prompt=True)
