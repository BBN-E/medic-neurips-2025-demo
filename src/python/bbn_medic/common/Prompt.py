"""
A Prompt represents a question asked to a medical chatbot.
"""
import json
from typing import Any
from bbn_medic.utils.dict_utils import flatten_dict


class Prompt:
    def __init__(self, text: str, id: str = None, metadata: dict = None, assign_id_automatically: bool = True):
        """
            Args:
                id (str): a unique id for this entry
                text (str): string representation of the Prompt
                metadata (dict): an arbitrary dict containing information about the Prompt, such as style, system_prompt, etc.
                    style (str): string description of the style, used to generate the Prompt
                    system_prompt (str, optional): non-default system prompt used to generate this prompt
        """
        self.text = text
        self.id = id
        self.metadata = metadata
        self.type = "Prompt"

        if self.id is None and assign_id_automatically:
            self.id = hex(self.__hash__() & 0xffffffffffffffff)

    def as_list(self):
        return list(self)

    @classmethod
    def from_json(cls, json_data: dict[str, Any], assign_id_automatically: bool = True):
        # check that 'type' is 'Prompt'
        if 'type' not in json_data or json_data["type"] != "Prompt":
            raise ValueError("JSON data does not contain object of type 'Prompt'")

        # pull essential elements from Prompt object
        if 'text' not in json_data:
            raise ValueError("Prompt does not match expected format: missing 'text' field")

        metadata = None
        if 'metadata' in json_data and type(json_data["metadata"]) == dict:
            metadata = json_data["metadata"]
        id = None
        if 'id' in json_data and type(json_data["id"]) == str:
            id = json_data["id"]
        return cls(json_data["text"], id=id, metadata=metadata, assign_id_automatically=assign_id_automatically)

    def to_json(self):
        # Create a dictionary to JSON-ify
        data = {
            'type': 'Prompt',
            'text': self.text
        }
        if self.id is not None:
            data['id'] = self.id
        if self.metadata is not None:
            data['metadata'] = self.metadata
        return json.dumps(data)

    def __hash__(self):
        return hash(tuple((self.text, self.id, flatten_dict(self.metadata)))) if self.metadata else hash(tuple((self.text, self.id)))

    def __eq__(self, other):
        if not isinstance(other, Prompt):
            return False
        return self.text == other.text and self.id == other.id and self.metadata == other.metadata

    def __str__(self):
        return self.to_json()
