import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from bbn_medic.llms.LLMInteractorInterface import LLMInteractorInterface


class Gemma2LLMInteractor(LLMInteractorInterface):
    def get_formatted_input(self, text, add_generation_prompt=False):
        messages = [
            {"role": "user", "content": text},
        ]
        formatted_input = self._tokenizer.apply_chat_template(messages,
                                                              tokenize=False,
                                                              add_generation_prompt=add_generation_prompt)
        return formatted_input

    def load(self, model_path_string, compression, seed=42):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path_string,
                                                        cache_dir=None,
                                                        local_files_only=False,
                                                        trust_remote_code=True)

        if compression is None:
            self._model = AutoModelForCausalLM.from_pretrained(model_path_string,
                                                               cache_dir=None,
                                                               local_files_only=False,
                                                               trust_remote_code=True,
                                                               torch_dtype="auto",
                                                               device_map="auto")

        elif compression == "4bit":
            print(f"Using model {model_path_string} with 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config
            )

        elif compression == "8bit":
            print(f"Using model {model_path_string} with 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_use_double_quant=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path_string,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config
            )

        else:
            raise ValueError(f"Unknown compression parameter {compression}")

        torch.manual_seed(seed)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prompt_for_desire_to_chatbot_prompt(self, desire_text, style):
        raise NotImplementedError()

    def prompt_for_chatbot_prompt_to_chatbot_prompt(self, prompt_text, style):
        raise NotImplementedError()

    def prompt_for_desire_to_desire(self, desire_text, style):
        raise NotImplementedError()
    
    def prompt_for_desire_patient_to_chatbot_prompt(self, patient_expression, lang, desire, style):
        raise NotImplementedError()
