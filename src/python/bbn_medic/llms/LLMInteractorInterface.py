from abc import ABC, abstractmethod

import torch

"""
Interface defining functions for interacting with LLMs
"""


class LLMInteractorInterface(ABC):

    def __init__(self, **kwargs) -> None:
        self._model = None
        self._tokenizer = None
        self._pre_prompt = ""

        self._mask = None
        self._use_chinese_logits_processor = False

    def get_model_name(self):
        if not self._model:
            print("No model loaded yet")
            return None
        return self._model.config.model_type

    @abstractmethod
    def load(self, model_path_string, compression, seed=42):
        """
        Desired behavior: after calling this function, model will be loaded and _model and _tokenizer will have non None values
        """
        raise NotImplementedError()

    @abstractmethod
    def prompt_for_desire_to_desire(self, desire_text, style):
        raise NotImplementedError()

    @abstractmethod
    def prompt_for_desire_to_chatbot_prompt(self, desire_text, style):
        raise NotImplementedError()

    @abstractmethod
    def prompt_for_chatbot_prompt_to_chatbot_prompt(self, prompt_text, style):
        raise NotImplementedError()

    @abstractmethod
    def prompt_for_desire_patient_to_chatbot_prompt(self, patient_expression, lang, desire, style):
        raise NotImplementedError()

    @abstractmethod
    def get_formatted_input(self, text, add_generation_prompt=False):
        raise NotImplementedError()

    def forward(self, prompt, num_return_sequences=2, temperature=1.0, max_length=8192, **kwargs):

        def chinese_logits_processor(token_ids, logits):
            # logits_processor default recieve the logits which is the score matrix of each time-step
            """
            A processor to ban Chinese characters
            """
            if self._mask is None:
                # as we don't know where the Chinses tokens locate at which index
                # in the vocabulary but we know how it looks like and the range of it 

                # decode all the tokens in the vocabulary in order 
                token_ids = torch.arange(logits.size(-1))
                decoded_tokens = self._tokenizer.batch_decode(token_ids.unsqueeze(1), skip_special_tokens=True)

                # create a mask tensor to exclude positions of Chinese characters.
                # since this process uses a for loop and is time-consuming, 
                # the result will be stored as a property for later use to ensure it only runs once.
                self._mask = torch.tensor([
                    # loop through each token in the vocabulary and compare it to Chinese characters.
                    any(0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF or 0xF900 <= ord(c) <= 0xFAFF for c in
                        token)
                    for token in decoded_tokens
                ])

            # mask the score by - inf
            logits[:, self._mask] = -float("inf")
            return logits

        logits_processor_list = []
        if self._use_chinese_logits_processor:
            logits_processor_list.append(chinese_logits_processor)

        prompt_wrapped = prompt
        with torch.inference_mode():
            inputs = self._tokenizer(
                prompt_wrapped, return_tensors="pt", truncation=True, padding=False
            )
            inputs.to(self._model.device)
            generation_output = self._model.generate(
                **inputs,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                repetition_penalty=1.2,
                do_sample=False if temperature == 0.0 else True,
                # top_p=0.95,
                # top_k=40,
                max_new_tokens=max_length,
                logits_processor=logits_processor_list
            )

        if num_return_sequences == 1:
            output_text = self._tokenizer.decode(generation_output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            generated_text = [output_text.replace(prompt_wrapped, "").strip()]
        else:
            generated_text = []
            for i in range(num_return_sequences):
                output_text = self._tokenizer.decode(generation_output[i][inputs.input_ids.shape[1]:],
                                                     skip_special_tokens=True)
                generated_text.append(output_text)

        return generated_text
