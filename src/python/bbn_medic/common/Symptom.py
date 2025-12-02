import typing

import pydantic

from bbn_medic.utils.text_utils import get_id_from_text


class Symptom(pydantic.BaseModel):
    """
    This is different from Symptom classes under Patient because it's considered as generic and can be referenced by Disorder Object
    """
    id: str
    type: typing.Literal['Symptom'] = pydantic.Field(default="Symptom")
    text: str  # Here, we don't store semantically similar strings but single string

    @staticmethod
    def from_text(text: str, iid: typing.Optional[str] = None):
        if iid is None:
            iid = get_id_from_text(text)
        return Symptom(text=text, id=iid)
