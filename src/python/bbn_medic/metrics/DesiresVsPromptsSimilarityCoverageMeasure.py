import typing

from sentence_transformers import SentenceTransformer

from bbn_medic.common.Desire import Desire
from bbn_medic.common.Prompt import Prompt
from bbn_medic.metrics.PromptCoverageMeasureInterface import PromptCoverageMeasureInterface
from bbn_medic.utils.dict_utils import recursively_find_in_dict


class DesiresVsPromptsSimilarityCoverageMeasure(PromptCoverageMeasureInterface):
    def __init__(self, model_path: str, desires: typing.List[Desire]):
        self.feature_space_model = SentenceTransformer(model_path)
        self.desire_id_to_text = {}
        self.desire_id_to_embedding = {}

        # In the following, it is assumed that the number of desires is not astronomical, so, they are kept in memory
        for desire in desires:
            self.desire_id_to_text[desire.id] = desire.text

    def calculate_coverage(self, prompts: typing.List[Prompt]) -> typing.List[float]:
        coverages = []
        for prompt in prompts:
            assert isinstance(prompt, Prompt)
            desire_source_id = recursively_find_in_dict(prompt.metadata, "Desire_source_id")
            if desire_source_id is None:
                raise ValueError(
                    f"Prompt {prompt} doesn't have 'Desire_source_id' or 'AtomicPatientExpression_source_id' in the metadata")
            if desire_source_id in self.desire_id_to_embedding:
                desire_embedding = self.desire_id_to_embedding[desire_source_id]
            else:
                desire_embedding = self.desire_id_to_embedding[desire_source_id] = self.feature_space_model.encode(
                    [self.desire_id_to_text[desire_source_id]])
            prompt_embedding = self.feature_space_model.encode([prompt.text])
            similarity = self.feature_space_model.similarity(desire_embedding, prompt_embedding).numpy()[0][0]
            coverages.append(similarity.item())
        return coverages
