"""
A Desire represents a topic area which a user wants to solicit information from a medical chatbot about.
"""
import json
from typing import Any

from bbn_medic.common.Prompt import Prompt
from bbn_medic.utils.dict_utils import flatten_dict


class Desire:
    def __init__(self, text: str, id: str = None, metadata: dict = None, assign_id_automatically: bool = True, prompts: list[Prompt] | Prompt = None):
        """
            Args:
                text (str): string representation of a Desire
                prompts (list[Prompt]|Prompt, optional): Prompt(s) associated with this Desire
        """
        self.text = text
        self.id = id
        self.metadata = metadata
        self.type = "Desire"
        self.prompt_ids = set()

        # if prompts was a single prompt, convert into a list with a single element
        if type(prompts) == Prompt:
            self.prompts = [prompts]
        elif type(prompts) == list and all(isinstance(prompt, Prompt) for prompt in prompts):
            self.prompts = prompts
        elif prompts is None:
            self.prompts = []
        else:
            raise TypeError("prompts, if given, should be a Prompt object or list of Prompt objects")

        if self.id is None and assign_id_automatically:
            self.id = hex(self.__hash__() & 0xffffffffffffffff)

    def as_list(self):
        return list(self)

    @classmethod
    def from_json(cls, json_data: dict[str, Any], assign_id_automatically: bool = True):
        # check that 'type' is 'Desire'
        if 'type' not in json_data or json_data["type"] != "Desire":
            raise ValueError("JSON data does not contain object of type 'Desire'")

        # pull essential elements from Desire JSON object
        if 'text' not in json_data:
            raise ValueError("Desire does not match expected format: missing 'text' field")

        # pull optional elements from Desire JSON object
        metadata = None
        if 'metadata' in json_data and type(json_data["metadata"]) == dict:
            metadata = json_data["metadata"]
        id = None
        if 'id' in json_data and type(json_data["id"]) == str:
            id = json_data["id"]
        prompts = None
        if 'prompts' in json_data:
            prompts = [Prompt.from_json(prompt) for prompt in json_data['prompts']]

        return cls(json_data['text'], id=id, metadata=metadata, assign_id_automatically=assign_id_automatically, prompts=prompts)

    def to_json(self):
        # Create a dictionary to JSON-ify; only include self.prompt if it's not None
        data = {
            'type': 'Desire',
            'text': self.text
        }
        if self.id is not None:
            data['id'] = self.id
        if self.metadata is not None:
            data['metadata'] = self.metadata
        if self.prompts and len(self.prompts) > 0:
            data['prompts'] = [json.loads(prompt.to_json()) for prompt in self.prompts]
        return json.dumps(data)

    def add_prompt(self, prompt):
        if prompt.id in self.prompt_ids:
            # self.prompts.remove(prompt)
            print(f"WARN: adding a duplicate prompt {prompt.id} to desire {self.id}")
        self.prompt_ids.add(prompt.id)
        self.prompts.append(prompt)

    def remove_prompt(self, prompt):
        self.prompts.remove(prompt)
        self.prompt_ids.remove(prompt.id)

    def __hash__(self):
        return hash(tuple((self.text, self.id, flatten_dict(self.metadata), tuple(self.prompts)))) \
            if self.metadata \
            else hash(tuple((self.text, self.id, tuple(self.prompts))))

    def __eq__(self, other):
        if not isinstance(other, Desire):
            return False
        return self.text == other.text and self.id == other.id and self.metadata == other.metadata and self.prompts == other.prompts

    def __str__(self):
        return self.to_json()
