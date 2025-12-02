"""Ensures that Answer constructor retains consistency check"""
import bz2
import filecmp
import gzip
import os

import pytest
from pathlib import Path

from bbn_medic.common.Answer import Answer
from bbn_medic.common.Desire import Desire
from bbn_medic.common.Prompt import Prompt
from bbn_medic.io.io_utils import DesireJSONLGenerator, PromptJSONLGenerator, fopen, AnswerJSONLGenerator
from bbn_medic.io.io_utils import JSONLineVisitor

SAMPLE_ANSWERS_PATH = Path(__file__).parent.resolve() / 'sample_answers.jsonl'
SAMPLE_ANSWERS_TEMP_BZ2 = SAMPLE_ANSWERS_PATH.with_suffix(".json.bz2")
SAMPLE_ANSWERS_TEMP_GZ = SAMPLE_ANSWERS_PATH.with_suffix(".json.gz")

TEST_OUTPUT_ANSWERS_TEMP_PATH = Path(__file__).parent.resolve() / 'test_output_answers.jsonl'
TEST_OUTPUT_ANSWERS_TEMP_BZ2 = TEST_OUTPUT_ANSWERS_TEMP_PATH.with_suffix(".json.bz2")
TEST_OUTPUT_ANSWERS_TEMP_GZ = TEST_OUTPUT_ANSWERS_TEMP_PATH.with_suffix(".json.gz")

TEST_ANSWER1 = Answer("Certainly! It is important to address the root cause of your symptoms and speak with a healthcare provider about adjusting your treatment plan if needed.",
                      "0x6148e921e870e496", "Mistral-7B-Instruct-v0.1-GPTQ", id="0x8b77f7401923607")
TEST_ANSWER2 = Answer("Certainly! Drug adverse reactions occur when a medication interacts with the body in a way that is harmful or causes unwanted side effects.",
                      "0x5c0c8756d08d3d3e", "Mistral-7B-Instruct-v0.1-GPTQ", id="0x81eb3bf4effc28cb")
TEST_ANSWER3 = Answer("Of course! It is important for you to know about any possible side effects of the medication you are taking.",
                      "0x45bc284dc5d5a0c6", "Mistral-7B-Instruct-v0.1-GPTQ", id="0x52ff78fa1b8be9ee")


def test_read_answers_from_jsonl_and_bz2_and_gz_files():
    answer1, answer2, answer3 = AnswerJSONLGenerator.read(SAMPLE_ANSWERS_PATH)
    assert answer1 == TEST_ANSWER1
    assert answer2 == TEST_ANSWER2
    assert answer3 == TEST_ANSWER3

    fopen(SAMPLE_ANSWERS_TEMP_BZ2, 'wb').write(fopen(SAMPLE_ANSWERS_PATH, 'rb').read())
    answer1, answer2, answer3 = AnswerJSONLGenerator.read(SAMPLE_ANSWERS_TEMP_BZ2)
    assert answer1 == TEST_ANSWER1
    assert answer2 == TEST_ANSWER2
    assert answer3 == TEST_ANSWER3
    os.remove(SAMPLE_ANSWERS_TEMP_BZ2)

    with fopen(SAMPLE_ANSWERS_PATH, 'rb') as infile:
        with fopen(SAMPLE_ANSWERS_TEMP_GZ, 'wb') as outfile:
            outfile.writelines(infile)
    answer1, answer2, answer3 = AnswerJSONLGenerator.read(SAMPLE_ANSWERS_TEMP_GZ)
    assert answer1 == TEST_ANSWER1
    assert answer2 == TEST_ANSWER2
    assert answer3 == TEST_ANSWER3
    os.remove(SAMPLE_ANSWERS_TEMP_GZ)


def test_write_answer_jsonl_and_bz2_and_gz():
    answers = [TEST_ANSWER1, TEST_ANSWER2, TEST_ANSWER3]

    AnswerJSONLGenerator.write(TEST_OUTPUT_ANSWERS_TEMP_PATH, answers)
    assert filecmp.cmp(SAMPLE_ANSWERS_PATH, TEST_OUTPUT_ANSWERS_TEMP_PATH)

    AnswerJSONLGenerator.write(TEST_OUTPUT_ANSWERS_TEMP_BZ2, answers)
    assert bz2.decompress(open(TEST_OUTPUT_ANSWERS_TEMP_BZ2, 'rb').read()) == open(SAMPLE_ANSWERS_PATH, 'rb').read()
    os.remove(TEST_OUTPUT_ANSWERS_TEMP_BZ2)

    AnswerJSONLGenerator.write(TEST_OUTPUT_ANSWERS_TEMP_GZ, answers)
    assert gzip.decompress(open(TEST_OUTPUT_ANSWERS_TEMP_GZ, 'rb').read()) == open(SAMPLE_ANSWERS_PATH, 'rb').read()
    os.remove(TEST_OUTPUT_ANSWERS_TEMP_GZ)

    os.remove(TEST_OUTPUT_ANSWERS_TEMP_PATH)
