from bbn_medic.llms.MistralLLMInteractor import MistralLLMInteractor
import random

random.seed(10)

def test_mistral():
    model_path = '/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ'
    mistral_instance = MistralLLMInteractor()
    mistral_instance.load(model_path)

    test_prompt = 'In which year was Obama elected president?'
    test_answer = '2008'

    generated_text = mistral_instance.forward(test_prompt, temperature=0.0001, num_return_sequences=1)

    assert generated_text[0] == test_answer
