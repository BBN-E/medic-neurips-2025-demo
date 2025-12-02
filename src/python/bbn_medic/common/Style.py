"""
A Style contains a text representation of a style.
Style objects are used to prompt an LLM to adopt a certain style_id when asking questions, etc.
"""
import json
from typing import Any


class Style:
    def __init__(self, id: str, text: str):
        """
            Args:
                text (str): string representation of the Style
        """
        self.id = id
        self.text = text

    @classmethod
    def from_json(cls, json_data: dict[str, Any]):
        if 'type' not in json_data or json_data["type"] != "Style":
            raise ValueError("JSON data does not contain object of type 'Style'")

        if 'text' not in json_data:
            raise ValueError("Style does not match expected format: missing 'text' field")

        return cls(id=json_data['id'], text=json_data['text'])

    def to_json(self):
        data = {
            'id': self.id,
            'type': 'Style',
            'text': self.text
        }
        return json.dumps(data)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Style):
            return False
        return self.id == other.id and self.text == other.text

    def __str__(self):
        return self.to_json()
