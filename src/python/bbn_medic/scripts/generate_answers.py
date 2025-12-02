import argparse
import json

from pathlib import Path

from bbn_medic.llms import ModelFactory
from bbn_medic.common.Prompt import Prompt
from bbn_medic.generation.AnswerGenerator import AnswerGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, required=True, help='Input jsonl file with prompts to answer (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--output_file', type=str, required=True, help="Path to output jsonl (or jsonl.gz or jsonl.bz2) file")
    parser.add_argument("--model_path", type=str, help="Path to model",
                        default="/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ")
    parser.add_argument('--num_answers', type=int, help='Number of answers to generate for each input prompt', default=2)
    parser.add_argument('--model_temp', type=float, help='Temperature setting for the LLM',
                        default=1.0)
    parser.add_argument('--segment_into_sentences', type=bool, default=False, help="If true, each Answer entry will have a list of sentences in metadata['segments']")
    parser.add_argument('--compression', type=str, choices=['4bit', '8bit'], help="Set to 4bit or 8bit for dynamically compressing the LLM")

    args = parser.parse_args()

    # load model
    model_instance = ModelFactory.load(args.model_path, compression=args.compression)

    AnswerGenerator.generate(
        input_file=Path(args.prompts_file),
        output_file=Path(args.output_file),
        model_instance=model_instance,
        num_answers=args.num_answers,
        model_temp=args.model_temp,
        segment_into_sentences=args.segment_into_sentences)
