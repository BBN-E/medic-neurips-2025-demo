import random
from pathlib import Path
import os

from bbn_medic.generation.AnswerGenerator import AnswerGenerator
from bbn_medic.llms import ModelFactory
from bbn_medic.generation.PromptGenerator import PromptGenerator
from bbn_medic.io.io_utils import AnswerJSONLGenerator

seed = 10
random.seed(seed)


MISTRAL_MODEL_PATH = '/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ'
LLAMA3_MODEL_PATH = '/nfs/nimble/projects/ami/models/nvidia-Llama3-ChatQA-1.5-8B/'
INPUT_PROMPT = Path(__file__).parent.resolve() / 'sample_prompt.jsonl'
TEMP_OUTPUT_ANSWERS_MISTRAL_JSONL = Path(__file__).parent.resolve() / 'test_output_answers_mistral.jsonl'
TEMP_OUTPUT_ANSWERS_LLAMA3L_JSONL = Path(__file__).parent.resolve() / 'test_output_answers_llama3.jsonl'


def generate_answers_for_prompts(model_path, output_path):
    num_answers_to_generate = 3
    model_instance = ModelFactory.load(model_path)
    AnswerGenerator.generate(input_file= INPUT_PROMPT,
                             output_file= output_path,
                             model_instance = model_instance,
                             num_answers=num_answers_to_generate,
                             model_temp=1.0)

    answers = list(AnswerJSONLGenerator.read(output_path))

    # SPOT CHECKS:
    assert len(answers) == 3   # 1 prompt with 3 answers each = 3

    os.remove(output_path)


def test_answers_from_mistral_medical_chatbot():
    generate_answers_for_prompts(MISTRAL_MODEL_PATH, TEMP_OUTPUT_ANSWERS_MISTRAL_JSONL)

def test_answers_from_llama3_medical_chatbot():
    generate_answers_for_prompts(LLAMA3_MODEL_PATH, TEMP_OUTPUT_ANSWERS_LLAMA3L_JSONL)
