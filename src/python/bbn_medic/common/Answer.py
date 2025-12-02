"""
An Answer represents a response from a medical chatbot when questioned with a Prompt.
"""
import json
from typing import Any

from bbn_medic.utils.dict_utils import flatten_dict


class Answer:
    def __init__(self, text: str, prompt_id: str, chatbot_name: str, id: str = None, metadata: dict = None,
                 assign_id_automatically: bool = True, segment_into_sentences: bool = False, segmenter_object=None):
        """
            Args:
                text (str): string representation of the Answer
                prompt_id (str): the id for the prompt this answer was generated in response to
                id (str): a unique id for this entry
                chatbot_name (str): name of the chatbot used to generate this answer
                metadata (dict): an arbitrary dict containing other information about the Answer
        """
        self.text = text
        self.id = id
        self.prompt_id = prompt_id
        self.chatbot_name = chatbot_name
        self.metadata = metadata
        self.type = "Answer"

        # if metadata["Prompt_metadata"]["id"] exists, check that it is identical to prompt_id;
        # we want to ensure these are consistent
        if self.metadata and "Prompt_metadata" in self.metadata and "id" in self.metadata["Prompt_metadata"] and \
           self.prompt_id != self.metadata["Prompt_metadata"]["id"]:
            mismatched_prompt_id_from_metadata = self.metadata["Prompt_metadata"]["id"]
            raise ValueError(f"The supplied prompt_id string '{prompt_id}' does not match "
                             f"metadata[\"Prompt_metadata\"][\"id\"]='{mismatched_prompt_id_from_metadata}'")

        if self.id is None and assign_id_automatically:
            self.id = hex(self.__hash__() & 0xffffffffffffffff)

        if self.metadata is None:
            self.metadata = {}

        if segment_into_sentences:
            if segmenter_object is not None:
                self.metadata["segments"] = segmenter_object.split_text_into_sentences(self.text)
            else:
                raise ValueError("Please provide a segmenter object")
        elif "segments" not in self.metadata:
            self.metadata["segments"] = [{"text": self.text, "start_char": 0, "end_char": len(self.text)}]

    def as_list(self):
        return list(self)

    @classmethod
    def from_json(cls, json_data: dict[str, Any], **kwargs):
        # check that 'type' is 'Answer'
        if 'type' not in json_data or json_data["type"] != "Answer":
            raise ValueError("JSON data does not contain object of type 'Answer'")

        # pull essential elements from Answer object
        if 'text' not in json_data:
            raise ValueError("Answer does not match expected format: missing 'text' field")
        if 'prompt_id' not in json_data:
            raise ValueError("Answer does not match expected format: missing 'prompt_id' field")
        if 'chatbot_name' not in json_data:
            raise ValueError("Answer does not match expected format: missing 'chatbot_name' field")

        metadata = None
        if 'metadata' in json_data and type(json_data["metadata"]) == dict:
            metadata = json_data["metadata"]
        id = None
        if 'id' in json_data and type(json_data["id"]) == str:
            id = json_data["id"]
        return cls(json_data["text"], json_data["prompt_id"], json_data["chatbot_name"], id=id, metadata=metadata, **kwargs)

    def to_json(self):
        # Create a dictionary to JSON-ify
        data = {
            'type': 'Answer',
            'text': self.text,
            'prompt_id': self.prompt_id,
            'chatbot_name': self.chatbot_name
        }
        if self.id is not None:
            data['id'] = self.id
        if self.metadata is not None:
            data['metadata'] = self.metadata
        return json.dumps(data)

    def __hash__(self):
        return hash(tuple((self.text, self.id, flatten_dict(self.metadata)))) \
            if self.metadata \
            else hash(tuple((self.text, self.id)))

    def __eq__(self, other):
        if not isinstance(other, Answer):
            return False
        return self.text == other.text and self.prompt_id == other.prompt_id and self.chatbot_name == other.chatbot_name and \
            self.id == other.id and self.metadata == other.metadata

    def __str__(self):
        return self.to_json()

    def update_with_segment(self, segment, assign_id_automatically: bool = True):
        prev_end_char = len(self.text)
        self.text += segment
        segment_length = len(segment)
        if self.metadata.get("segments", None) is None:
            self.metadata["segments"] = []
        self.metadata["segments"].append({"text": segment, "start_char": prev_end_char, "end_char": prev_end_char + segment_length})
        if assign_id_automatically:
            self.id = hex(self.__hash__() & 0xffffffffffffffff)
