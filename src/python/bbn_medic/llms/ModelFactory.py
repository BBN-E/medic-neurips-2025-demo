from bbn_medic.llms.Llama2LLMInteractor import Llama2LLMInteractor
from bbn_medic.llms.LlamaLLMInteractor import Llama3LLMInteractor
from bbn_medic.llms.MistralLLMInteractor import MistralLLMInteractor
# from bbn_medic.llms.BioMistralLLMInteractor import BioMistralLLMInteractor
from bbn_medic.llms.QwenLLMInteractor import QwenLLMInteractor
from bbn_medic.llms.Gemma2LLMInteractor import Gemma2LLMInteractor
# from bbn_medic.llms.MedGemmaLLMInteractor import MedGemmaLLMInteractor
from bbn_medic.llms.MedalpacaInteractor import MedalpacaLLMInteractor
from bbn_medic.llms.GenericNonTemplatedLLMInteractor import GenericNonTemplatedLLMInteractor

models = {
    # "biomistral": lambda: BioMistralLLMInteractor(),
    "mistral": lambda: MistralLLMInteractor(),
    "llama3": lambda: Llama3LLMInteractor(),
    "olmo": lambda: GenericNonTemplatedLLMInteractor(),
    "medalpaca": lambda: MedalpacaLLMInteractor(),
    "llama2": lambda: Llama2LLMInteractor(),
    "qwen": lambda: QwenLLMInteractor(),
    "gemma2": lambda : Gemma2LLMInteractor(),
    "medgemma": lambda : Gemma2LLMInteractor(),
}


def load(model_path_string, compression=None, seed=42):
    model = None
    model_path_search_string = model_path_string.lower().replace('-', '')
    for model_string in models.keys():
        if model_string in model_path_search_string:
            model = models[model_string]()
            model.load(model_path_string, compression=compression)
            print(f"Using model: {model_string}")
            break
    if not model:
        raise ValueError(f"Unknown model type: {model_path_string}; "
                         f"currently supported models include {list(models.keys())}")
    return model
