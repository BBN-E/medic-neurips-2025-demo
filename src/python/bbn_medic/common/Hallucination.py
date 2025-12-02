"""
A Hallucination represents a snippet of text from an Answer which contains a hallucination (e.g., a medical error
likely to cause patient harm).
"""
import json
from typing import Any

from bbn_medic.utils.dict_utils import flatten_dict


class Hallucination:
    def __init__(self, detector_type: str, snippet: str, explanation: str, harm_level: str, confidence: float,
                 prompt_id: str, answer_id: str, model_name: str, id: str = None, metadata: dict = None,
                 assign_id_automatically: bool = True, segment_id: int = None):
        """
            Args:
                detector_type (str): e.g., HallucinationDetector, OmissionDetector
                snippet (str): text snippet from an Answer identified as containing a hallucination
                explanation (str): explanation of why the snippet contains a hallucination
                harm_level (str): level of patient harm (should be 'low', 'medium', or 'high')
                confidence (float): confidence in this hallucination identification (between 0 and 1 inclusive)
                prompt_id (str): the id for the prompt the answer being evaluated was generated in response to
                answer_id (str): the id for the answer being evaluated
                id (str): a unique id for this entry
                model_name (str): name of the model used to evaluate the question/answer pair
                metadata (dict): an arbitrary dict containing other information about the Answer
                segment_id (int): the index of the segment of the answer containing the hallucination (optional)
        """
        self.evaluator_type = detector_type
        self.snippet = snippet
        self.explanation = explanation
        self.harm_level = harm_level
        self.confidence = confidence
        self.id = id
        self.prompt_id = prompt_id
        self.answer_id = answer_id
        self.segment_id = segment_id
        self.model_name = model_name
        self.metadata = metadata
        self.type = "Hallucination"

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
        # check that 'type' is 'Answer'
        if 'type' not in json_data or json_data["type"] != "Hallucination":
            raise ValueError("JSON data does not contain object of type 'Hallucination'")

        # pull essential elements from object
        for field in {"detector_type", "snippet", "explanation", "harm_level", "confidence", "prompt_id", "answer_id",
                      "model_name"}:
            if field not in json_data:
                raise ValueError(f"Hallucination does not match expected format: missing '{field}' field")

        metadata = None
        if 'metadata' in json_data and type(json_data["metadata"]) == dict:
            metadata = json_data["metadata"]
        id = None
        if 'id' in json_data and type(json_data["id"]) == str:
            id = json_data["id"]
        segment_id = None
        if 'segment_id' in json_data and type(json_data["segment_id"]) == int:
            segment_id = json_data["segment_id"]
        return cls(json_data["detector_type"], json_data["snippet"], json_data["explanation"], json_data["harm_level"],
                   json_data["confidence"], json_data["prompt_id"], json_data["answer_id"],
                   json_data["model_name"], id=id, segment_id=segment_id, metadata=metadata, assign_id_automatically=assign_id_automatically)

    def to_json(self):
        # Create a dictionary to JSON-ify
        data = {
            'type': self.type,
            'detector_type': self.evaluator_type,
            'snippet': self.snippet,
            'explanation': self.explanation,
            'harm_level': self.harm_level,
            'confidence': self.confidence,
            'prompt_id': self.prompt_id,
            'answer_id': self.answer_id,
            'model_name': self.model_name,
            'segment_id': self.segment_id
        }
        if self.id is not None:
            data['id'] = self.id
        if self.metadata is not None:
            data['metadata'] = self.metadata
        return json.dumps(data)

    def __hash__(self):
        return hash(tuple((self.snippet, self.explanation, self.id, flatten_dict(self.metadata)))) \
            if self.metadata \
            else hash(tuple((self.snippet, self.explanation, self.id)))

    def __eq__(self, other):
        if not isinstance(other, Hallucination):
            return False
        return self.snippet == other.snippet and self.prompt_id == other.prompt_id and self.answer_id == other.answer_id and \
            self.explanation == other.explanation and self.harm_level == other.harm_level and self.confidence == other.confidence and \
            self.model_name == other.model_name and self.id == other.id and self.metadata == other.metadata and self.segment_id == other.segment_id

    def __str__(self):
        return self.to_json()
