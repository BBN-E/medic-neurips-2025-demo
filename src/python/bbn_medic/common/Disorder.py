"""
A Disorder represents a medical disorder that a chatbot user may be interested in.
"""
import json
from typing import List, Any


class Disorder:
    def __init__(self, text: str, id: str = None, assign_id_automatically: bool = True,
                 alternate_names: list[str] | str = None, relevant_symptom_ids: List[str] = None):
        """
            Args:
                text (str): string representation of a Disorder
                alternate_names (list[str]|str, optional): string(s) associated with this Disorder for matching with eventual diversified prompt

            {"type": "Disorder", "text": "agoraphobia", "alternate_names": ["agoraphobia"], "id": "bbn_mh_disorder_000"}

        """
        self.text = text
        self.id = id
        self.type = "Disorder"

        if relevant_symptom_ids is None:
            relevant_symptom_ids = []
        self.relevant_symptom_ids = relevant_symptom_ids
        # if alternate_names was a single string, convert into a list with a single element
        if isinstance(alternate_names, str):
            self.alternate_names = [alternate_names]
        elif isinstance(alternate_names, list) and all(isinstance(m, str) for m in alternate_names):
            self.alternate_names = alternate_names
        elif alternate_names is None:
            self.alternate_names = []
        else:
            raise TypeError("alternate_names, if given, should be a string or list of strings")

        if self.id is None and assign_id_automatically:
            self.id = hex(self.__hash__() & 0xffffffffffffffff)

    @classmethod
    def from_json(cls, json_data: dict[str, Any], assign_id_automatically: bool = True):
        # check that 'type' is 'Disorder'
        if 'type' not in json_data or json_data["type"] != "Disorder":
            raise ValueError("JSON data does not contain object of type 'Disorder'")

        # pull essential elements from Disorder JSON object
        if 'text' not in json_data:
            raise ValueError("Disorder does not match expected format: missing 'text' field")

        # pull optional elements from Disorder JSON object
        id = None
        if 'id' in json_data and type(json_data["id"]) == str:
            id = json_data["id"]
        alternate_names = None
        if 'alternate_names' in json_data:
            alternate_names = [m for m in json_data['alternate_names']]

        return cls(json_data['text'], id=id, assign_id_automatically=assign_id_automatically,
                   alternate_names=alternate_names, relevant_symptom_ids=json_data.get("relevant_symptom_ids", None))

    def to_json(self):
        # Create a dictionary to JSON-ify; only include self.alternate_names if it's not None
        data = {
            'type': 'Disorder',
            'text': self.text
        }
        if self.id is not None:
            data['id'] = self.id
        if self.alternate_names and len(self.alternate_names) > 0:
            data['alternate_names'] = [m for m in self.alternate_names]
        if self.relevant_symptom_ids is not None and len(self.relevant_symptom_ids) > 0:
            data['relevant_symptom_ids'] = [m for m in self.relevant_symptom_ids]
        return json.dumps(data)

    def __hash__(self):
        return hash(tuple((self.text, self.id, tuple(self.alternate_names))))

    def __eq__(self, other):
        if not isinstance(other, Disorder):
            return False
        return self.text == other.text and self.id == other.id and self.alternate_names == other.alternate_names and self.relevant_symptom_ids == other.relevant_symptom_ids

    def __str__(self):
        return self.to_json()
