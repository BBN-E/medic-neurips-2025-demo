from bbn_medic.llms.LlamaLLMInteractor import Llama3LLMInteractor
import random

random.seed(10)

def test_llama():
    model_path = '/nfs/nimble/projects/ami/models/nvidia-Llama3-ChatQA-1.5-8B/'
    llama_instance = Llama3LLMInteractor()
    llama_instance.load(model_path)

    test_prompt = 'In which year was Obama elected president?'

    formatted_prompt = llama_instance.get_formatted_input(test_prompt)
    generated_text = llama_instance.forward(formatted_prompt, temperature=0.0001, num_return_sequences=1)

    assert generated_text[0] == 'Barack Obama was first elected President of the United States in 2008'

