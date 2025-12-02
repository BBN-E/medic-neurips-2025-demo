"""Runs tests of io for basic types"""
import os
import random

from pathlib import Path

from bbn_medic.io.DesireCollection import DesireCollection
from bbn_medic.io.io_utils import DesireJSONLGenerator, PromptJSONLGenerator, JSONLGenerator

SAMPLE_DESIRES_PATH = Path(__file__).parent.resolve() / 'kratkiew_0016gen_desires_all.jsonl'
SAMPLE_STANDARD_PROMPTS_PATH = Path(__file__).parent.resolve() / 'kratkiew_0016gen_standard_prompts_00.jsonl'
SAMPLE_DIVERSIFIED_PROMPTS_PATH = Path(__file__).parent.resolve() / 'kratkiew_0016gen_prompts_0_all.jsonl.gz'

SAMPLE_DESIRES_AND_CONCERNS_PATH = Path(__file__).parent.resolve() / '50004_prompt_gen_timing_test_desires_and_prompts.jsonl'
OUTPUT_DIR = Path(__file__).parent.resolve()
N_FILE_SPLITS = 10


def test_read_desires_and_prompts_into_collection():
    desires = DesireJSONLGenerator.read(SAMPLE_DESIRES_PATH)
    standard_prompts = PromptJSONLGenerator.read(SAMPLE_STANDARD_PROMPTS_PATH)
    diversified_prompts = PromptJSONLGenerator.read(SAMPLE_DIVERSIFIED_PROMPTS_PATH)

    desire_collection = DesireCollection(desires, standard_prompts, quiet_mode=True)
    assert len(desire_collection.get_desires()) == 120, "number of desires read in should be 120"
    assert len(desire_collection.get_prompts()) == 400, "number of prompts read at this point should be 400"

    desire_collection.add_prompts(diversified_prompts)
    assert len(desire_collection.get_prompts()) == 4956, "number of prompts read at this point should be 4000"


# this test takes a large jsonl file of interrelated desires and concerns, scrambles the line order and writes into
# N sub-files, and then feeds those files one at a time to the Desire Collection. The expectation is that, at the end
# all the desires and prompts will be properly accounted for in the DesireCollection
def test_read_desires_and_prompts_out_of_order():
    test_files = scramble_and_split_input_file_by_line(SAMPLE_DESIRES_AND_CONCERNS_PATH)
    desire_collection = DesireCollection(quiet_mode=True)
    for f in test_files:
        desire_collection.add_content_from_jsonl(JSONLGenerator.read(f))

    assert len(desire_collection.get_desires()) == 4, "number of desires read in should be 4"
    assert len(desire_collection.get_prompts()) == 36, "number of prompts read in should be 36"

    # walk desires with prompts to validate that all are matched up
    desire_count = 0
    prompt_count = 0
    prompt_id_count = {}
    for desire in desire_collection.get_desires():
        desire_count += 1
        for prompt in desire.prompts:
            prompt_count += 1
            if prompt.id not in prompt_id_count:
                prompt_id_count[prompt.id] = 0
            prompt_id_count[prompt.id] += 1
    assert desire_count == 4, "number of desires matched up should be 4"
    assert prompt_count == 36, "number of prompts read in should be 36"

    for f in test_files:
        os.remove(f)


def scramble_and_split_input_file_by_line(input_file_path):
    # Read all lines from the input file
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    # Shuffle the lines
    random.shuffle(lines)
    # Split the lines into approximately equal parts for each file
    split_lines = [lines[i::N_FILE_SPLITS] for i in range(N_FILE_SPLITS)]
    # Write each part to a separate output file
    output_files = []
    for i, part in enumerate(split_lines):
        output_file = os.path.join(OUTPUT_DIR, f'{input_file_path}.shufflesplit.{i + 1}')
        with open(output_file, 'w') as file:
            file.writelines(part)
        output_files.append(output_file)
    return output_files
