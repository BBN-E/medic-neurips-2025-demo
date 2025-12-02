import random
from pathlib import Path
import os

from bbn_medic.detection.hallucinations.BaselineHallucinationDetector import BaselineHallucinationDetector
from bbn_medic.llms import ModelFactory
from bbn_medic.generation.PromptGenerator import PromptGenerator
from bbn_medic.io.io_utils import AnswerJSONLGenerator, JSONLGenerator


LLAMA3_MODEL_PATH = '/nfs/nimble/projects/ami/models/nvidia-Llama3-ChatQA-1.5-8B/'
INPUT_PROMPTS = Path(__file__).parent.resolve() / 'sample_prompt.jsonl'
INPUT_ANSWERS = Path(__file__).parent.resolve() / 'sample_answers.jsonl'
TMP_OUTPUT_DETECTION_LLAMA3_JSONL = Path(__file__).parent.resolve() / 'test_output_hallucination_detection_llama3.jsonl'


def run_detection_on_answers(model_path, output_path, include_harmless=False):
    model_instance = ModelFactory.load(model_path)
    BaselineHallucinationDetector.detect(prompts_input_file=INPUT_PROMPTS,
                                         answers_input_file=INPUT_ANSWERS,
                                         output_file= output_path,
                                         model_instance = model_instance,
                                         model_temp=1.0,
                                         retry_max_attempts=1,
                                         include_harmless=include_harmless)

    hallucinations = list(JSONLGenerator.read(output_path))

    os.remove(output_path)

    return hallucinations


def test_detection_with_llama3():
    hallucinations = run_detection_on_answers(LLAMA3_MODEL_PATH, TMP_OUTPUT_DETECTION_LLAMA3_JSONL)
    assert len(hallucinations) == 3 # llama finds 3 hallucinations with this answer file

def test_detection_with_llama3_including_harmless():
    hallucinations = run_detection_on_answers(LLAMA3_MODEL_PATH, TMP_OUTPUT_DETECTION_LLAMA3_JSONL,
                                              include_harmless=True)
    assert len(hallucinations) == 3  # llama still finds 3 hallucinations when you include assessments rated harmless

