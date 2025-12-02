import argparse

from bbn_medic.generation.PromptGenerator import PromptGenerator
from bbn_medic.llms import ModelFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str,
                        help='Input jsonl file with prompts to diversify (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--styles_file', type=str,
                        help='Input jsonl file with styles (can also be jsonl.gz or jsonl.bz2)')

    parser.add_argument('--output_file', type=str, help="Path to output jsonl file (can also be jsonl.gz or jsonl.bz2)")
    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ")
    parser.add_argument('--num_prompts', type=int, help='Number of prompts to generate for each input record',
                        default=2)
    parser.add_argument('--model_temp', type=float, help='Temperature setting for the LLM', default=1.0)
    args = parser.parse_args()

    # load model and set parameters
    model_instance = ModelFactory.load(args.model_path)

    PromptGenerator.restyle_prompt(input_file=args.prompts_file, output_file=args.output_file,
                                   styles_file=args.styles_file, model_instance=model_instance,
                                   model_temp=args.model_temp, num_prompts=args.num_prompts)
