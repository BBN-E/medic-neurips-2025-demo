import bz2
import codecs
import gzip
import json
import mmap
import os
import pickle
import shutil
import typing
from abc import ABC
from pathlib import Path
from warnings import warn

from bbn_medic.common.Answer import Answer
from bbn_medic.common.Confidence import Confidence
from bbn_medic.common.Desire import Desire
from bbn_medic.common.Disorder import Disorder
from bbn_medic.common.Hallucination import Hallucination
from bbn_medic.common.Omission import Omission
from bbn_medic.common.Patient import Patient, PatientExpression, AtomicPatientExpression
from bbn_medic.common.Prompt import Prompt
from bbn_medic.common.Style import Style
from bbn_medic.common.Symptom import Symptom

"""
Code for reading and writing MEDIC basic types uses this interface
"""

type_to_pydantic_cls = {
    "Patient": Patient,
    "PatientExpression": PatientExpression,
    "AtomicPatientExpression": AtomicPatientExpression,
    "Symptom": Symptom,
}


class GeneratorInterface(ABC):
    """Returns basic types one at a time, iterator style. """

    @staticmethod
    def read(input_file: Path, skip_malformed_lines=False):
        pass

    @staticmethod
    def write(output_file: Path, list):
        pass


class JSONLGenerator(GeneratorInterface):
    @staticmethod
    def read_all(input_file: Path, skip_malformed_lines=False, **optional_args) -> typing.List[
        Desire | Prompt | Disorder | Patient | PatientExpression | Symptom | AtomicPatientExpression]:
        return list(JSONLGenerator.read(input_file, skip_malformed_lines, **optional_args))

    @staticmethod
    def read(input_file: Path, skip_malformed_lines=False, **optional_args) -> typing.Iterable[
        Desire | Prompt | Disorder | Patient | PatientExpression | Symptom | AtomicPatientExpression]:
        if str(input_file).endswith(".txt") or str(input_file).endswith(".txt.gz"):
            for line in fopen(input_file):
                for obj in JSONLGenerator.read(line.strip(), skip_malformed_lines, **optional_args):
                    yield obj
        else:
            line_number = 1
            for line in fopen(input_file):
                try:
                    j = json.loads(line)
                    if "type" in j:
                        if j["type"] == "Prompt":
                            obj = Prompt.from_json(j, **optional_args)
                        elif j["type"] == "Desire":
                            obj = Desire.from_json(j, **optional_args)
                        elif j["type"] == "Answer":
                            obj = Answer.from_json(j, **optional_args)
                        elif j["type"] == "Hallucination":
                            obj = Hallucination.from_json(j, **optional_args)
                        elif j["type"] == "Omission":
                            obj = Omission.from_json(j, **optional_args)
                        elif j["type"] == "Style":
                            obj = Style.from_json(j, **optional_args)
                        elif j["type"] == "Hallucination":
                            obj = Hallucination.from_json(j, **optional_args)
                        elif j["type"] == "Omission":
                            obj = Omission.from_json(j, **optional_args)
                        elif j["type"] == "Confidence":
                            obj = Confidence.from_json(j, **optional_args)
                        elif j["type"] == "Disorder":
                            obj = Disorder.from_json(j, **optional_args)
                        elif j['type'] in type_to_pydantic_cls:
                            obj = type_to_pydantic_cls[j['type']](**j)
                        else:
                            obj = j
                    line_number += 1
                    yield obj
                except Exception as e:
                    if skip_malformed_lines:
                        warn(f"Skipping malformed line {line_number} in desires file {input_file}")
                    else:
                        raise e

    @staticmethod
    def write(output_jsonl_file: Path, objects: list):
        with fopen(output_jsonl_file, 'w') as file:
            for obj in objects:
                if obj.type in type_to_pydantic_cls:
                    file.write(f"{json.dumps(obj.model_dump(), ensure_ascii=False)}\n")
                else:
                    file.write(f"{obj.to_json()}\n")


# TODO remove if JSONLGenerator makes obsolete
class AnswerJSONLGenerator(GeneratorInterface):
    @staticmethod
    def read(input_jsonl_file: Path, skip_malformed_lines=False) -> Answer:
        line_number = 1
        for line in fopen(input_jsonl_file):
            try:
                j = json.loads(line)
                line_number += 1
                answer = Answer.from_json(j)
                yield answer
            except Exception as e:
                if skip_malformed_lines:
                    warn(f"Skipping malformed line {line_number} in answers file {input_jsonl_file}")
                else:
                    raise e

    @staticmethod
    def write(output_jsonl_file: Path, answers: list[Answer]):
        with fopen(output_jsonl_file, 'w') as file:
            for answer in answers:
                file.write(f"{answer.to_json()}\n")


# TODO remove if JSONLGenerator makes obsolete
class DesireJSONLGenerator(GeneratorInterface):
    @staticmethod
    def read(input_jsonl_file: Path, skip_malformed_lines=False) -> Desire:
        line_number = 1
        for line in fopen(input_jsonl_file):
            try:
                j = json.loads(line)
                line_number += 1
                desire = Desire.from_json(j)
                yield desire
            except Exception as e:
                if skip_malformed_lines:
                    warn(f"Skipping malformed line {line_number} in desires file {input_jsonl_file}")
                else:
                    raise e

    @staticmethod
    def write(output_jsonl_file: Path, desires: list[Desire]):
        with fopen(output_jsonl_file, 'w') as file:
            for desire in desires:
                file.write(f"{desire.to_json()}\n")


# TODO remove if JSONLGenerator makes obsolete
class PromptJSONLGenerator(GeneratorInterface):
    @staticmethod
    def read(input_jsonl_file: Path, skip_malformed_lines=False) -> Prompt:
        line_number = 1
        for line in fopen(input_jsonl_file):
            try:
                j = json.loads(line)
                line_number += 1
                yield Prompt.from_json(j)
            except Exception as e:
                if skip_malformed_lines:
                    print(f"Skipping malformed line {line_number} in prompts file {input_jsonl_file}")
                else:
                    raise e

    @staticmethod
    def write(output_jsonl_file: Path, prompts: list[Prompt]):
        with fopen(output_jsonl_file, 'w') as file:
            for prompt in prompts:
                file.write(f"{prompt.to_json()}\n")


# TODO remove if JSONLGenerator makes obsolete
class StyleJSONLGenerator(GeneratorInterface):
    @staticmethod
    def read(input_jsonl_file: Path, skip_malformed_lines=False) -> Style:
        line_number = 1
        for line in fopen(input_jsonl_file):
            try:
                j = json.loads(line)
                line_number += 1
                yield Style.from_json(j)
            except Exception as e:
                if skip_malformed_lines:
                    print(f"Skipping malformed line {line_number} in prompts file {input_jsonl_file}")
                else:
                    raise e

    @staticmethod
    def write(output_jsonl_file: Path, styles: list[Style]):
        with fopen(output_jsonl_file, 'w') as file:
            for style in styles:
                file.write(f"{style.to_json()}\n")


def read_file_to_set(filename):
    ret = set()
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            ret.add(line.strip())
    return ret


def dict2string(dictionary: dict, title: str):
    """Turns a dictionary into something pretty to print or write.

    Args:
        dictionary (dict): Dictionary with the data.
        title (str): Pretty title to give the collection of data.
    """
    message = '\n\t' + title + '\n'
    info = ['\t\t' + str(key) + '=' + str(value) + '\n' for key, value in list(dictionary.items())]
    info_str = ''.join(info)
    message += info_str

    return message


def serialize_to_pickle(obj, output_path):
    with gzip.open(output_path, mode="wb") as wfp:
        pickle.dump(obj, wfp)


def deserialize_from_pickle(input_path):
    with gzip.open(input_path, mode="rb") as rfp:
        return pickle.load(rfp)


def fopen(filename: str | Path, mode='rt', encoding='utf-8', **kwargs):
    '''Drop-in replacement for built in open() so that .gz and .bz2 files can be
    handled transparently. If filename is '-', standard input will be used.

    Since we are mostly dealing with text files UTF-8 encoding is used by default.
    '''

    gz_condition = filename.endswith(".gz") if type(filename) == str else filename.suffix.endswith(".gz")
    bz_condition = filename.endswith(".bz2") if type(filename) == str else filename.suffix.endswith(".bz2")

    if gz_condition:
        _fopen = gzip.open
        if 'b' not in mode and 't' not in mode:
            mode = mode + 't'  # 'rb' is the default for gzip and bz2
    elif bz_condition:
        _fopen = bz2.open
        if 'b' not in mode and 't' not in mode:
            mode = mode + 't'
    else:
        _fopen = open
    if 'b' in mode:
        return _fopen(filename, mode=mode, **kwargs)
    else:
        return _fopen(filename, mode=mode, encoding=encoding, **kwargs)


def join_files(list_of_files_to_join, output_file):
    with open(output_file, 'wb') as wfd:
        for f in list_of_files_to_join:
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)


class JSONLineVisitor():
    """
    Assuming you have a huge jsonl file and you want random access to any line of that file
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.mmap_handle = None
        self.line_key_to_line_info = list()  # key is line key and value is a tuple of (num_bytes_to_seek, num_bytes_to_read)

    def init_mmap_handle(self):
        with fopen(self.file_path, 'rb') as fp:
            self.mmap_handle = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)

    def release_mmap_handle(self):
        self.mmap_handle.close()
        self.mmap_handle = None

    def build_index(self):
        num_bytes_to_seek = 0
        self.line_key_to_line_info.clear()
        self.mmap_handle.seek(0, os.SEEK_SET)
        line = self.mmap_handle.readline()
        line_idx = 0
        while len(line) > 0:
            num_bytes_to_read = len(line)
            self.line_key_to_line_info.append((num_bytes_to_seek, num_bytes_to_read))
            num_bytes_to_seek += num_bytes_to_read
            line_idx += 1
            line = self.mmap_handle.readline()
        assert line_idx == len(self.line_key_to_line_info)

    def access_line(self, line_no):
        num_bytes_to_seek, num_bytes_to_read = self.line_key_to_line_info[line_no]
        self.mmap_handle.seek(num_bytes_to_seek, os.SEEK_SET)
        line = self.mmap_handle.read(num_bytes_to_read).decode("utf-8")
        json_line = json.loads(line)
        return json_line

    def length(self):
        return len(self.line_key_to_line_info)

    def __len__(self):
        return self.length()

    def __getitem__(self, item):
        return self.access_line(item)

    def __iter__(self):
        return iter(self.access_line(i) for i in range(self.length()))
