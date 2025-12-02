import hashlib
import re

from spacy.language import Language
from unidecode import unidecode


@Language.component('custom_seg_on_numbered_list')
def custom_seg(doc):
    boundary = re.compile('^[0-9][0-9]?$')
    prev = doc[0].text
    length = len(doc)
    for index, token in enumerate(doc):
        if token.text == '.' and boundary.match(prev) and index != length - 1:
            doc[index + 1].sent_start = False
        prev = token.text
    return doc


@Language.component('custom_seg_on_newlines')
def custom_seg(doc):
    prev = doc[0].text
    length = len(doc)
    for index, token in enumerate(doc):
        if token.text != '\n' and prev == '\n':
            doc[index].sent_start = True
        prev = token.text
    return doc


@Language.component('custom_seg_on_newlines_and_numbered_list')
def custom_seg(doc):
    boundary = re.compile('^[0-9][0-9]?$')
    prev = doc[0].text
    length = len(doc)
    for index, token in enumerate(doc):
        if token.text == '.' and boundary.match(prev) and index != length - 1:
            doc[index + 1].sent_start = False
        elif token.text != '\n' and prev == '\n':
            doc[index].sent_start = True
        prev = token.text
    return doc


spacy_sentencizer_custom_rules = [
    ("new_line", r"\n"),
    ("numbered_list", r"^\d+\.\s")
]


def post_process_text(text, apply_unidecode=False, characters_to_replace_with_space=(), only_keep_first_question=False,
                      max_length=None, remove_leading_spaces=False, remove_trailing_spaces=False, remove_cot_tags=True,
                      remove_leading_assistant=True, reduced_to_single_space=True, only_keep_first_question_or_declarative=False):
    if apply_unidecode:
        text = unidecode(text)
    for c in characters_to_replace_with_space:
        text = text.replace(c, ' ')
    # reduce to single spaces
    if reduced_to_single_space:
        text = " ".join(text.split())
    # The following block removes "stuff" after the first question (ending in a ?) has been generated
    if only_keep_first_question:
        match = re.findall(r'^([^?]+\?)', text)
        text = match[0] if len(match) > 0 else None
    if only_keep_first_question_or_declarative:
        match = re.findall(r'^([\w ]+[?.])', text)
        text = match[0] if len(match) > 0 else None
    if max_length is not None:
        words = text.split()
        if len(words) > max_length:
            text = None
    if type(text) is str and remove_leading_spaces:
        text = text.lstrip()
    if type(text) is str and remove_trailing_spaces:
        text = text.rstrip()
    # Remove "assistant"
    if isinstance(text, str) and remove_leading_assistant and text.startswith("assistant "):
        text = text[len("assistant "):]
    # Remove </think>
    if isinstance(text, str) and remove_cot_tags:
        left_think = text.find("</think>")
        if left_think >= 0:
            text = text[left_think + len("</think>"):]
    return text


def post_process_desire_text(text, regex_of_characters_to_replace_with_space='[^A-Za-z0-9\', -]', max_length=None):
    # TODO merge this with post_process_text
    text = post_process_text(text, max_length=max_length, remove_leading_spaces=True, remove_trailing_spaces=True)
    if text is None:
        return None  # this text has been filtered out and should be skipped

    text = re.sub(regex_of_characters_to_replace_with_space, ' ', text)
    text = " ".join(text.split())
    return text


def get_id_from_text(text, mask=0xffffffffffffffff, return_hex=False):
    h = hashlib.sha256(text.encode("utf-8", errors="ignore"))
    hex_num = hex(int(h.hexdigest(), 16) & mask)
    return str(hex_num) if return_hex is False else hex_num
