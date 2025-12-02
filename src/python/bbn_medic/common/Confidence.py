"""
A Confidence represents the confidence in the language used in an Answer.
"""
import json
from typing import Any

from bbn_medic.utils.dict_utils import flatten_dict


class Confidence:
    def __init__(self, explanation: str, confidence: float,
                 prompt_id: str, answer_id: str, model_name: str, id: str = None, metadata: dict = None,
                 assign_id_automatically: bool = True):
        """
            Args:
                explanation (str): explanation for the level of confidence estimated
                confidence (float): confidence estimated (between 0 and 1 inclusive)
                prompt_id (str): the id for the prompt the answer being evaluated was generated in response to
                answer_id (str): the id for the answer being evaluated
                id (str): a unique id for this entry
                model_name (str): name of the model used to evaluate the question/answer pair
                metadata (dict): an arbitrary dict containing other information about the Answer
        """
        self.explanation = explanation
        self.confidence = confidence
        self.id = id
        self.prompt_id = prompt_id
        self.answer_id = answer_id
        self.model_name = model_name
        self.metadata = metadata
        self.type = "Confidence"

        # if metadata["Prompt_metadata"]["id"] exists, check that it is identical to prompt_id;
        # we want to ensure these are consistent
        if self.metadata and "Prompt_metadata" in self.metadata and "id" in self.metadata["Prompt_metadata"] and \
                self.prompt_id != self.metadata["Prompt_metadata"]["id"]:
            mismatched_prompt_id_from_metadata = self.metadata["Prompt_metadata"]["id"]
            raise ValueError(f"The supplied prompt_id string '{prompt_id}' does not match "
                             f"metadata[\"Prompt_metadata\"][\"id\"]='{mismatched_prompt_id_from_metadata}'")

        if self.id is None and assign_id_automatically:
            self.id = hex(self.__hash__() & 0xffffffffffffffff)

    def as_list(self):
        return list(self)

    @classmethod
    def from_json(cls, json_data: dict[str, Any], assign_id_automatically: bool = True):
        # check that 'type' is 'Confidence'
        if 'type' not in json_data or json_data["type"] != "Confidence":
            raise ValueError("JSON data does not contain object of type 'Confidence'")

        # pull essential elements from object
        for field in {"explanation", "confidence", "prompt_id", "answer_id",
                      "model_name"}:
            if field not in json_data:
                raise ValueError(f"Confidence does not match expected format: missing '{field}' field")

        metadata = None
        if 'metadata' in json_data and type(json_data["metadata"]) == dict:
            metadata = json_data["metadata"]
        id = None
        if 'id' in json_data and type(json_data["id"]) == str:
            id = json_data["id"]
        return cls(json_data["explanation"],
                   json_data["confidence"], json_data["prompt_id"], json_data["answer_id"],
                   json_data["model_name"], id=id, metadata=metadata, assign_id_automatically=assign_id_automatically)

    def to_json(self):
        # Create a dictionary to JSON-ify
        data = {
            'type': self.type,
            'explanation': self.explanation,
            'confidence': self.confidence,
            'prompt_id': self.prompt_id,
            'answer_id': self.answer_id,
            'model_name': self.model_name
        }
        if self.id is not None:
            data['id'] = self.id
        if self.metadata is not None:
            data['metadata'] = self.metadata
        return json.dumps(data)

    def __hash__(self):
        return hash(tuple((self.explanation, self.id, flatten_dict(self.metadata)))) \
            if self.metadata \
            else hash(tuple((self.explanation, self.id)))

    def __eq__(self, other):
        if not isinstance(other, Confidence):
            return False
        return self.prompt_id == other.prompt_id and self.answer_id == other.answer_id and \
            self.explanation == other.explanation and \
            self.confidence == other.confidence and self.model_name == other.model_name and \
            self.id == other.id and self.metadata == other.metadata

    def __str__(self):
        return self.to_json()
