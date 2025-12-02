from bbn_medic.llms import ModelFactory
from bbn_medic.generation.DesireGenerator import DesireGenerator
from bbn_medic.io.io_utils import DesireJSONLGenerator
from pathlib import Path
import os

MISTRAL_MODEL_PATH = '/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ'
LLAMA_MODEL_PATH = '/nfs/nimble/projects/ami/models/nvidia-Llama3-ChatQA-1.5-8B/'
SAMPLE_DESIRES_PATH = Path(__file__).parent.resolve() / 'desires.jsonl'
TEST_OUTPUT_OVERGENERATED_PROMPTS_TEMP_PATH = Path(__file__).parent.resolve() / 'overgenerated_desires.jsonl'
MODEL_TEMP = 1.0

def run_desire_overgeneration(model_path):
    num_desires_to_append = 3
    # load model
    model_instance = ModelFactory.load(MISTRAL_MODEL_PATH)

    DesireGenerator.generate(SAMPLE_DESIRES_PATH, TEST_OUTPUT_OVERGENERATED_PROMPTS_TEMP_PATH, model_instance, num_desires_to_append, MODEL_TEMP)
    
    num_input_desires = len(list(DesireJSONLGenerator.read(SAMPLE_DESIRES_PATH)))
    num_output_desires = len(list(DesireJSONLGenerator.read(TEST_OUTPUT_OVERGENERATED_PROMPTS_TEMP_PATH)))

    assert num_output_desires == (num_desires_to_append + 1) * num_input_desires

    os.remove(TEST_OUTPUT_OVERGENERATED_PROMPTS_TEMP_PATH)


def test_desire_overgeneration_mistral():
    run_desire_overgeneration(MISTRAL_MODEL_PATH)

def test_desire_overgeneration_llama():
    run_desire_overgeneration(LLAMA_MODEL_PATH)
    
