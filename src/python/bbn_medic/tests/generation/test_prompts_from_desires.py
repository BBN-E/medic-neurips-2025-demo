import random
from pathlib import Path
import os

from bbn_medic.llms import ModelFactory
from bbn_medic.generation.PromptGenerator import PromptGenerator
from bbn_medic.io.io_utils import DesireJSONLGenerator, PromptJSONLGenerator

seed = 10
random.seed(seed)


MISTRAL_MODEL_PATH = '/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ'
LLAMA3_MODEL_PATH = '/nfs/nimble/projects/ami/models/nvidia-Llama3-ChatQA-1.5-8B/'
SAMPLE_DESIRES_PATH = Path(__file__).parent.resolve() / 'sample_desires_to_generate_prompts_for.jsonl'
STYLES_PATH = Path(__file__).parent.resolve() / 'styles.jsonl'
TEST_OUTPUT_OVERGENERATED_PROMPTS_TEMP_PATH = Path(__file__).parent.resolve() / 'sample_prompts2.jsonl'

def get_prompt_styles_from_jsonl(jsonl_file, styles_file):

    list_of_prompt_style_string_lengths = []

    for desire_obj in DesireJSONLGenerator.read(jsonl_file):
        for prompt in desire_obj.prompts:
            if prompt.style:
                list_of_prompt_style_string_lengths.append(len(prompt.style))
    return list_of_prompt_style_string_lengths


def generate_prompts_from_desires(model_path):
    num_prompts_to_generate = 3
    model_instance = ModelFactory.load(MISTRAL_MODEL_PATH)
    PromptGenerator.generate(input_file = SAMPLE_DESIRES_PATH,
                             styles_file = STYLES_PATH,
                             output_file = TEST_OUTPUT_OVERGENERATED_PROMPTS_TEMP_PATH,
                             model_instance = model_instance,
                             num_prompts = num_prompts_to_generate,
                             model_temp = 1.0)

    prompts = list(PromptJSONLGenerator.read(TEST_OUTPUT_OVERGENERATED_PROMPTS_TEMP_PATH))

    # SPOT CHECKS:
    assert len(prompts) == 17             # each desire has 9 prompts generated for it (3 for each style),
                                          # 1 gets filtered out by current default filtering settings
    assert prompts[0].metadata["style"] == "7th grade level of education" # spot check styles
    assert prompts[1].metadata["style"] == "7th grade level of education"
    assert prompts[2].metadata["style"] == "7th grade level of education"
    assert prompts[3].metadata["style"] == "8th grade level of education"
    assert prompts[4].metadata["style"] == "8th grade level of education"
    assert prompts[5].metadata["style"] == "8th grade level of education"
    assert prompts[6].metadata["style"] == "14th grade level of education"
    assert prompts[7].metadata["style"] == "14th grade level of education"
    assert prompts[8].metadata["style"] == "14th grade level of education"
    assert prompts[9].metadata["style"] == "7th grade level of education"
    assert prompts[10].metadata["style"] == "7th grade level of education"
    assert prompts[11].metadata["style"] == "7th grade level of education"
    assert prompts[12].metadata["style"] == "8th grade level of education"
    assert prompts[13].metadata["style"] == "8th grade level of education"
    assert prompts[14].metadata["style"] == "14th grade level of education"
    assert prompts[15].metadata["style"] == "14th grade level of education"
    assert prompts[16].metadata["style"] == "14th grade level of education"

    os.remove(TEST_OUTPUT_OVERGENERATED_PROMPTS_TEMP_PATH)


def test_prompts_from_desires_with_mistral():
    generate_prompts_from_desires(MISTRAL_MODEL_PATH)

def test_prompts_from_desires_with_llama3():
    generate_prompts_from_desires(LLAMA3_MODEL_PATH)
    
