import asyncio
import logging
from functools import partial
from typing import List, Optional, Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel, BaseMessage, CallbackManagerForLLMRun, \
    AsyncCallbackManagerForLLMRun, ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel

global logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("huggingface_pipeline.py")


class MyHuggingFaceModel(BaseChatModel):
    """
    https://github.com/langchain-ai/langchain/discussions/9596
    https://qwen.readthedocs.io/en/latest/framework/Langchain.html
    """
    max_new_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 0.8
    history_len: int = 0
    huggingface_tokenizer: PreTrainedTokenizerBase
    huggingface_model: PreTrainedModel

    @property
    def param_inject_huggingface_generation(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            # "history_len": self.history_len
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return self.param_inject_huggingface_generation

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        # Implement the logic for generating the response using the HuggingFace model
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _call(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        # Implement the logic for generating the response using the HuggingFace model
        resolved_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                resolved_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                resolved_messages.append({"role": "user", "content": message.content})
            else:
                raise NotImplementedError("Unknown type {}".format(type(message)))

        text = self.huggingface_tokenizer.apply_chat_template(
            resolved_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.huggingface_tokenizer([text], return_tensors="pt").to(self.huggingface_model.device)
        logger.info(self.param_inject_huggingface_generation)
        generated_ids = self.huggingface_model.generate(
            **model_inputs,
            **self.param_inject_huggingface_generation
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.huggingface_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        func = partial(
            self._generate, messages, stop=stop, run_manager=run_manager, **kwargs
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)
