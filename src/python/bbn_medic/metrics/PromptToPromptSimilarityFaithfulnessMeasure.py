import typing

from bbn_medic.common.Patient import AtomicPatientExpression
from bbn_medic.metrics.PromptFaithfulnessMeasureInterface import PromptFaithfulnessMeasureInterface
from sentence_transformers import SentenceTransformer
from bbn_medic.common.Prompt import Prompt
from bbn_medic.utils.dict_utils import recursively_find_in_dict
from abc import ABC

from bbn_medic.io.io_utils import PromptJSONLGenerator
import pdb


class PromptToPromptSimilarityFaithfulnessMeasure(PromptFaithfulnessMeasureInterface):
    def __init__(self, model_path:str, source_prompts: list[Prompt]):
        self.feature_space_model = SentenceTransformer(model_path)
        self.source_prompt_id_to_text = {}
        self.source_prompt_id_to_embedding = {}

        if source_prompts is not None:
            for source_prompt in source_prompts:
                self.source_prompt_id_to_text[source_prompt.id] = source_prompt.text


    def calculate_faithfulness(self, prompts: list[typing.Union[Prompt, AtomicPatientExpression]]):
        faithfulness_list = []
        for prompt in prompts:
            assert isinstance(prompt, Prompt) or isinstance(prompt, AtomicPatientExpression)
            if isinstance(prompt, Prompt):
                prompt_source_id = recursively_find_in_dict(prompt.metadata, 'Prompt_source_id')
            else:
                prompt_source_id = prompt.metadata["AtomicPatientExpression_source_id"]
            if prompt_source_id is None:
                raise ValueError(
                    f"Prompt {prompt} doesn't have 'Desire_source_id' or 'AtomicPatientExpression_source_id' in the metadata")
            if prompt_source_id in self.source_prompt_id_to_embedding:
                source_prompt_embedding = self.source_prompt_id_to_embedding[prompt_source_id]
            else:
                source_prompt_embedding = self.source_prompt_id_to_embedding[prompt_source_id] = self.feature_space_model.encode(self.source_prompt_id_to_text[prompt_source_id])

            prompt_embedding = self.feature_space_model.encode([prompt.text])
            similarity = self.feature_space_model.similarity(source_prompt_embedding, prompt_embedding).numpy()[0][0]
            faithfulness_list.append(similarity.item())
        return faithfulness_list
