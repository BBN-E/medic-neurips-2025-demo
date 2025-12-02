import torch
from bbn_medic.llms.LLMInteractorInterface import LLMInteractorInterface
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings


class Llama2LLMInteractor(LLMInteractorInterface):

    def prompt_for_desire_patient_to_chatbot_prompt(self, patient_expression, lang, desire, style):
        extra_string = "" if style == "standard" else f", who {style},"
        prompt = f'''{self._pre_prompt}. {patient_expression}. A patient{extra_string} would like a medical chatbot to help him/her with the topic {desire}. One possible question the patient might ask the chatbot is:'''
        formatted_prompt = self.get_formatted_input(prompt)
        return formatted_prompt

    def __init__(self):
        super().__init__()
        self._pre_prompt = "'Please give a full and complete answer for the following request. "

    def load(self, model_path, compression=None, seed=42):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._tokenizer.chat_template = "{{ bos_token }}{%- if messages[0]['role'] == 'system' -%}{% set loop_messages = messages[1:] %}{%- else -%}{% set loop_messages = messages %}{% endif %}System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.\n\n{% for message in loop_messages %}{%- if message['role'] == 'user' -%}User: {{ message['content'].strip() + '\n\n' }}{%- else -%}Assistant: {{ message['content'].strip() + '\n\n' }}{%- endif %}{% if loop.last and message['role'] == 'user' %}Assistant:{% endif %}{% endfor %}";
        if compression is None:
            self._model = AutoModelForCausalLM.from_pretrained(model_path,
                                                               torch_dtype=torch.float16,
                                                               device_map="auto")
        elif compression == "4bit":
            print(f"Using model {model_path} with 4-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self._model = AutoModelForCausalLM.from_pretrained(model_path,
                                                               quantization_config=quantization_config)
        elif compression == "8bit":
            print(f"Using model {model_path} with 8-bit quantization")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
            self._model = AutoModelForCausalLM.from_pretrained(model_path,
                                                               quantization_config=quantization_config)
        else:
            raise ValueError(f"Unknown compression parameter {compression}")

        torch.manual_seed(seed)

    def get_formatted_input(self, query, add_generation_prompt=False):

        # TODO: Add a field for context later. If you have a multi-turn conversation, you need to add
        # the previous dialog turns to the prompt so that the LLM is aware of the context of the conversation.
        # We don't need that level of functionality right now.
        formatted_msg = [
            {'role': 'system',
             'content': "This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."}
        ]

        formatted_msg.append({'role': 'user',
                             'content': query})
        formatted_input = self._tokenizer.apply_chat_template(formatted_msg,
                                                              tokenize=False,
                                                              add_generation_prompt=add_generation_prompt)
        return formatted_input

    def prompt_for_desire_to_desire(self, desire_text, style):
        extra_string = "" if style == "standard" else f", who {style},"
        prompt = f'''{self._pre_prompt}A patient{extra_string} would like a medical chatbot to help him/her with the topic of {desire_text}.
            Giving your answer in six words or less, another topic that the patient may want the chatbot to help him/her with is:'''
        formatted_prompt = self.get_formatted_input(prompt)
        return formatted_prompt

    def prompt_for_desire_to_chatbot_prompt(self, desire_text, style):
        extra_string = "" if style == "standard" else f", who {style},"
        prompt = f'''{self._pre_prompt}A patient{extra_string} would like a medical chatbot to help him/her with the topic {desire_text}.
            One possible question the patient might ask the chatbot is:'''
        formatted_prompt = self.get_formatted_input(prompt)
        return formatted_prompt

    def prompt_for_chatbot_prompt_to_chatbot_prompt(self, prompt_text, style):
        extra_string = "" if style == "standard" else f" in the style of someone who {style},"
        prompt = f'''{self._pre_prompt}One possible way to re-write the sentence "{prompt_text}"{extra_string} is: '''
        formatted_prompt = self.get_formatted_input(prompt)
        return formatted_prompt

    def text_to_formatted_prompt(self, text):
        formatted_prompt = self.get_formatted_input(text)
        return formatted_prompt
