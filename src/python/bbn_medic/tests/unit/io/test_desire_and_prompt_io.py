"""Runs tests of io for basic types"""
import bz2
import filecmp
import gzip
import json
import os
from json import JSONDecodeError

import pytest
from pathlib import Path

from bbn_medic.common.Desire import Desire
from bbn_medic.common.Prompt import Prompt
from bbn_medic.io.io_utils import DesireJSONLGenerator, PromptJSONLGenerator, fopen
from bbn_medic.io.io_utils import JSONLineVisitor

SAMPLE_PROMPTS_PATH = Path(__file__).parent.resolve() / 'sample_prompts.jsonl'
SAMPLE_PROMPTS_TEMP_BZ2 = SAMPLE_PROMPTS_PATH.with_suffix(".json.bz2")
SAMPLE_PROMPTS_TEMP_GZ = SAMPLE_PROMPTS_PATH.with_suffix(".json.gz")

TEST_OUTPUT_PROMPTS_TEMP_PATH = Path(__file__).parent.resolve() / 'test_output_prompts.jsonl'
TEST_OUTPUT_PROMPTS_TEMP_BZ2 = TEST_OUTPUT_PROMPTS_TEMP_PATH.with_suffix(".json.bz2")
TEST_OUTPUT_PROMPTS_TEMP_GZ = TEST_OUTPUT_PROMPTS_TEMP_PATH.with_suffix(".json.gz")

MALFORMED_PROMPTS_PATH = Path(__file__).parent.resolve() / 'sample_prompts_malformed.jsonl'
ONE_GOOD_PROMPT_AT_END_OF_MALFORMED_PROMPTS = Prompt("one good prompt at end to make testing easier")

SAMPLE_DESIRES_PATH = Path(__file__).parent.resolve() / 'sample_desires.jsonl'
SAMPLE_DESIRES_TEMP_BZ2 = SAMPLE_DESIRES_PATH.with_suffix(".json.bz2")
SAMPLE_DESIRES_TEMP_GZ = SAMPLE_DESIRES_PATH.with_suffix(".json.gz")

TEST_OUTPUT_DESIRES_TEMP_PATH = Path(__file__).parent.resolve() / 'test_output_desires.jsonl'
TEST_OUTPUT_DESIRES_TEMP_BZ2 = TEST_OUTPUT_DESIRES_TEMP_PATH.with_suffix(".json.bz2")
TEST_OUTPUT_DESIRES_TEMP_GZ = TEST_OUTPUT_DESIRES_TEMP_PATH.with_suffix(".json.gz")

TEST_PROMPT1 = Prompt("test prompt #1", id="test-prompt-1", metadata={"style": "7th grade level of education"})
TEST_PROMPT2 = Prompt("test prompt #2", id="test-prompt-2", metadata={"style": "8th grade level of education"})
TEST_PROMPT3 = Prompt("test prompt #3", id="test-prompt-3",
                      metadata={"style": "14th grade level of education", "system_prompt": "test system prompt"})
TEST_PROMPT4 = Prompt("test prompt #4", id="test-prompt-4", metadata={"style": "9th grade level of education"})


def test_read_prompts_from_jsonl_and_bz2_and_gz_files():
    prompt1, prompt2, prompt3 = PromptJSONLGenerator.read(SAMPLE_PROMPTS_PATH)
    assert prompt1 == TEST_PROMPT1
    assert prompt2 == TEST_PROMPT2
    assert prompt3 == TEST_PROMPT3

    fopen(SAMPLE_PROMPTS_TEMP_BZ2, 'wb').write(fopen(SAMPLE_PROMPTS_PATH, 'rb').read())
    prompt1, prompt2, prompt3 = PromptJSONLGenerator.read(SAMPLE_PROMPTS_TEMP_BZ2)
    assert prompt1 == TEST_PROMPT1
    assert prompt2 == TEST_PROMPT2
    assert prompt3 == TEST_PROMPT3
    os.remove(SAMPLE_PROMPTS_TEMP_BZ2)

    with fopen(SAMPLE_PROMPTS_PATH, 'rb') as infile:
        with fopen(SAMPLE_PROMPTS_TEMP_GZ, 'wb') as outfile:
            outfile.writelines(infile)
    prompt1, prompt2, prompt3 = PromptJSONLGenerator.read(SAMPLE_PROMPTS_TEMP_GZ)
    assert prompt1 == TEST_PROMPT1
    assert prompt2 == TEST_PROMPT2
    assert prompt3 == TEST_PROMPT3
    os.remove(SAMPLE_PROMPTS_TEMP_GZ)


def test_skip_malformed_lines():
    generator = PromptJSONLGenerator.read(MALFORMED_PROMPTS_PATH, skip_malformed_lines=True)
    one_good_prompt = next(generator)
    assert one_good_prompt == ONE_GOOD_PROMPT_AT_END_OF_MALFORMED_PROMPTS


def test_exception_on_malformed_lines():
    generator = PromptJSONLGenerator.read(MALFORMED_PROMPTS_PATH, skip_malformed_lines=False)
    with pytest.raises(ValueError) as exception:
        prompt1 = next(generator)
    assert "Prompt does not match expected format: missing 'text' field" in str(exception.value)

    # want to add more malformed examples? it's simpler to put them in other files b/c the reader won't keep reading


def test_write_prompt_jsonl_and_bz2_and_gz():
    prompts = [TEST_PROMPT1, TEST_PROMPT2, TEST_PROMPT3]

    PromptJSONLGenerator.write(TEST_OUTPUT_PROMPTS_TEMP_PATH, prompts)
    assert filecmp.cmp(SAMPLE_PROMPTS_PATH, TEST_OUTPUT_PROMPTS_TEMP_PATH)

    PromptJSONLGenerator.write(TEST_OUTPUT_PROMPTS_TEMP_BZ2, prompts)
    assert bz2.decompress(open(TEST_OUTPUT_PROMPTS_TEMP_BZ2, 'rb').read()) == open(SAMPLE_PROMPTS_PATH, 'rb').read()
    os.remove(TEST_OUTPUT_PROMPTS_TEMP_BZ2)

    PromptJSONLGenerator.write(TEST_OUTPUT_PROMPTS_TEMP_GZ, prompts)
    assert gzip.decompress(open(TEST_OUTPUT_PROMPTS_TEMP_GZ, 'rb').read()) == open(SAMPLE_PROMPTS_PATH, 'rb').read()
    os.remove(TEST_OUTPUT_PROMPTS_TEMP_GZ)

    os.remove(TEST_OUTPUT_PROMPTS_TEMP_PATH)


def test_read_desires_from_jsonl_and_bz2_and_gz_files():
    expected_desire1 = Desire("nutrition", id="test-desire-1", prompts=TEST_PROMPT1)
    expected_desire2 = Desire("nutrition", id="test-desire-2", prompts=TEST_PROMPT2)
    expected_desire3 = Desire("medication", id="test-desire-3", prompts=[TEST_PROMPT3, TEST_PROMPT4])
    expected_desire4 = Desire("medication", id="test-desire-4", )

    desire1, desire2, desire3, desire4 = DesireJSONLGenerator.read(SAMPLE_DESIRES_PATH)
    assert desire1 == expected_desire1
    assert desire2 == expected_desire2
    assert desire3 == expected_desire3
    assert desire4 == expected_desire4

    fopen(SAMPLE_DESIRES_TEMP_BZ2, 'wb').write(fopen(SAMPLE_DESIRES_PATH, 'rb').read())
    desire1, desire2, desire3, desire4 = DesireJSONLGenerator.read(SAMPLE_DESIRES_TEMP_BZ2)
    assert desire1 == expected_desire1
    assert desire2 == expected_desire2
    assert desire3 == expected_desire3
    assert desire4 == expected_desire4
    os.remove(SAMPLE_DESIRES_TEMP_BZ2)

    with fopen(SAMPLE_DESIRES_PATH, 'rb') as infile:
        with fopen(SAMPLE_DESIRES_TEMP_GZ, 'wb') as outfile:
            outfile.writelines(infile)
    desire1, desire2, desire3, desire4 = DesireJSONLGenerator.read(SAMPLE_DESIRES_TEMP_GZ)
    assert desire1 == expected_desire1
    assert desire2 == expected_desire2
    assert desire3 == expected_desire3
    assert desire4 == expected_desire4
    os.remove(SAMPLE_DESIRES_TEMP_GZ)


def test_write_desires_jsonl_and_bz2_and_gz():
    desire1 = Desire("nutrition", id="test-desire-1", prompts=TEST_PROMPT1)
    desire2 = Desire("nutrition", id="test-desire-2", prompts=TEST_PROMPT2)
    desire3 = Desire("medication", id="test-desire-3", prompts=[TEST_PROMPT3, TEST_PROMPT4])
    desire4 = Desire("medication", id="test-desire-4", )
    desires = [desire1, desire2, desire3, desire4]

    DesireJSONLGenerator.write(TEST_OUTPUT_DESIRES_TEMP_PATH, desires)
    assert filecmp.cmp(SAMPLE_DESIRES_PATH, TEST_OUTPUT_DESIRES_TEMP_PATH)

    DesireJSONLGenerator.write(TEST_OUTPUT_DESIRES_TEMP_BZ2, desires)
    assert bz2.decompress(open(TEST_OUTPUT_DESIRES_TEMP_BZ2, 'rb').read()) == open(SAMPLE_DESIRES_PATH, 'rb').read()
    os.remove(TEST_OUTPUT_DESIRES_TEMP_BZ2)

    DesireJSONLGenerator.write(TEST_OUTPUT_DESIRES_TEMP_GZ, desires)
    assert gzip.decompress(open(TEST_OUTPUT_DESIRES_TEMP_GZ, 'rb').read()) == open(SAMPLE_DESIRES_PATH, 'rb').read()
    os.remove(TEST_OUTPUT_DESIRES_TEMP_GZ)

    os.remove(TEST_OUTPUT_DESIRES_TEMP_PATH)
