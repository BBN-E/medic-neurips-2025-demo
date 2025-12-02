"""Ensures that Answer constructor retains consistency check"""

import pytest
from pathlib import Path

from bbn_medic.common.Answer import Answer


def test_exception_on_conflicting_prompt_id_and_metadata():
    with pytest.raises(ValueError) as exception:
        answer = Answer("this is a test answer", "prompt_id", "chatbot", id=None,
                        metadata={"Prompt_metadata": {"id": "different_prompt_id"}}, assign_id_automatically=True)
    assert f"The supplied prompt_id string 'prompt_id' does not match metadata[\"Prompt_metadata\"]" \
           f"[\"id\"]='different_prompt_id'" in str(exception.value)


def test_no_exception_on_matching_prompt_id_and_metadata():
    answer = Answer("this is a test answer", "prompt_id", "chatbot", id=None,
                    metadata={"Prompt_metadata": {"id": "prompt_id"}}, assign_id_automatically=True)
