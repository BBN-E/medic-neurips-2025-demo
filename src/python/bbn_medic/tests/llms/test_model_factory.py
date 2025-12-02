import pytest
from bbn_medic.llms import ModelFactory


def test_model_factory_on_nonexistent_model():
    with pytest.raises(ValueError) as exception:
        ModelFactory.load("unknown_model")
    assert(f"Unknown model type: unknown_model; currently supported models include ['mistral', 'llama']")
