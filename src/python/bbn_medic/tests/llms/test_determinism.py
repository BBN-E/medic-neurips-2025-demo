import pytest

from bbn_medic.llms import ModelFactory
from bbn_medic.llms.LlamaLLMInteractor import Llama3LLMInteractor
from bbn_medic.llms.MistralLLMInteractor import MistralLLMInteractor
from bbn_medic.llms.QwenLLMInteractor import QwenLLMInteractor

test_prompt = 'Please extend the thank you note below:\n\nGood work.'

# These tests pass using the bbn_medic python environment with the following GPUs: A6000, l40, p100, v100-16g, v100-32g

# CAVEAT: this is a very short generation below. Empirically, we found that, if the torch seed is not set, the response
# will vary every time you run it, whereas if it is set, the response is the same across the architectures above.
# However, longer experiments we ran still exposed measurable differences in generations between results generated
# using different GPU architectures, a known phenomenon attested elsewhere.

# Note that reformatting these to add line breaks seems to break the equality assertions in the tests below.
expected_answers = {
    "mistral": """Many thanks.  We'll see you soon!  \nPlease add another phrase or two so they know how much we appreciate their contributions to our project/team: \r\nTheir dedication and attention to detail made a significant difference in the success of this project. We couldn't have done it without them. Their contribution will always be recognized and highly valued by us.""",
    "llama": """Thank you notes are typically more formal than this. A good rule of thumb for extending a thank-you is that it shouldn't just simply be "Great job." Something like this might suffice but again it only has one sentence so I will try my best to modify it without making it look too cliche. 

"Thank you very much for finishing such an important project."

"That was an extremely stressful period for all parties involved"

"I've heard a lot about your hard working ethic as well as always taking pride in doing great quality work"
These may help!""",
    "qwen": """Thanks! I appreciate your efforts in finishing this project on time and under budget, which is a testament to your hard work and dedication.

Thank you again for all of your hard work on completing this project so efficiently and effectively. Your commitment to excellence has not gone unnoticed, as it played an essential role in achieving our goals. It's always refreshing to see someone go above and beyond what was expected.
Certainly! Here’s an extended version of your thank-you note:

---

Dear [Name],

I wanted to take a moment to express my sincere gratitude for your outstanding contributions to our recent project. Your diligent effort and unwavering commitment have truly made a significant impact, helping us complete the task not only on time but also well within budget. This achievement is a direct reflection of your hard work and dedication, and it underscores your exceptional skill set and professionalism.

Your ability to manage multiple tasks with such efficiency and attention to detail did not go unnoticed. You consistently demonstrated a high level of expertise and a proactive approach that kept the team motivated and focused. The quality of your work has been exemplary, and it has greatly contributed to the success of our objectives.

It’s always inspiring to witness such a strong sense of responsibility and enthusiasm. Your willingness to go above and beyond expectations sets a remarkable standard and serves as a model for others. Thank you once again for everything you’ve done. Your contribution is invaluable, and we are incredibly fortunate to have you on our team.

Best regards,

[Your Name]

--- 

Feel free to adjust any part to better fit your personal style or specific details about the project and individual."""
}


# This test is designed to fail if we introduce a new model but don't
# write a determinism test below for it. Requiring one pytest per model
# is neater than writing one test that loops through all models because
# it breaks out the pytest failures more neatly, i.e., by model more neatly.
def test_to_verify_all_models_have_determinism_tests():
    models_we_have_tests_for = list(expected_answers.keys())
    models_we_support = list(ModelFactory.models.keys())
    assert(models_we_support == models_we_have_tests_for)


def test_determinism_llama3():
    model_path = '/nfs/nimble/projects/ami/models/nvidia-Llama3-ChatQA-1.5-8B/'
    llama_instance = Llama3LLMInteractor()
    llama_instance.load(model_path)

    formatted_prompt = llama_instance.get_formatted_input(test_prompt)
    generated_text = llama_instance.forward(formatted_prompt, temperature=1.0, num_return_sequences=1)

    assert generated_text[0] == expected_answers["llama"]


def test_determinism_mistral():
    model_path = '/nfs/nimble/projects/ami/models/Mistral-7B-Instruct-v0.1-GPTQ'
    mistral_instance = MistralLLMInteractor()
    mistral_instance.load(model_path)

    generated_text = mistral_instance.forward(test_prompt, temperature=1.0, num_return_sequences=1)

    assert generated_text[0] == expected_answers["mistral"]

#@pytest.mark.skip(reason="Qwen is really slow and requires 4 GPUs and 48GB memory") # uncomment to skip this test
def test_determinism_qwen():
    model_path = '/nfs/nimble/projects/hiatus/hqiu/hiatus_public/transformers_cache/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842'
    qwen_instance = QwenLLMInteractor()
    qwen_instance.load(model_path)

    generated_text = qwen_instance.forward(test_prompt, temperature=1.0, num_return_sequences=1)
    print(generated_text[0])
    assert generated_text[0] == expected_answers["qwen"]
