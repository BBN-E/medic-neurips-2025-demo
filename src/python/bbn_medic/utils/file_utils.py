# This file was imported from pycube, hycube release R2022_07_06_1
import gzip
import bz2
import re
import pickle
import os
import io
import sys
from typing import List


def read_list(filename: str) -> List[str]:
    """Reads all lines from a file, stripping newlines """
    with fopen(filename, 'r') as lst:
        return [x.strip() for x in lst.readlines()]


def fopen(filename, mode='rt', encoding='utf-8', **kwargs):
    '''Drop-in replacement for built in open() so that .gz and .bz2 files can be
    handled transparently. If filename is '-', standard input will be used.

    Since we are mostly dealing with text files UTF-8 encoding is used by default.
    '''

    if filename == '-':
        if 'w' in mode:
            return io.TextIOWrapper(sys.stdout.buffer, encoding=encoding)
        else:
            return io.TextIOWrapper(sys.stdin.buffer, encoding=encoding)

    if filename.endswith(".gz"):
        _fopen = gzip.open
        if 'b' not in mode and 't' not in mode:
            mode = mode + 't'  # 'rb' is the default for gzip and bz2
    elif filename.endswith(".bz2"):
        _fopen = bz2.open
        if 'b' not in mode and 't' not in mode:
            mode = mode + 't'
    else:
        _fopen = open
    if 'b' in mode:
        return _fopen(filename, mode=mode, **kwargs)
    else:
        return _fopen(filename, mode=mode, encoding=encoding, **kwargs)


def handle_output_file(file_path, create_containing_dirs=False, confirm_overwrite=False):
    """
    Does certain sanity checks for an output file; optionally asks the user to confirm if overwriting the output file is
    acceptable.
    :param file_path: Path to output file.
    :param create_containing_dirs: If True, creates the non-existent directories in the path to output file.
    :param confirm_overwrite: If True, asks the user to confirm if overwriting the output file is acceptable.
    :return:
    # noqa
    """
    if os.path.isdir(file_path):
        raise ValueError(file_path + " is a directory!")
    if not create_containing_dirs:
        dir_exists = os.path.isdir(os.path.dirname(file_path))
        if not dir_exists:
            raise ValueError("Directory " + os.path.dirname(file_path) + "does not exist!")
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        return
    if confirm_overwrite:
        decision = '?'
        while decision not in ['y', 'n']:
            decision = input("File " + file_path + " already exists. OK to overwrite?(y/n)")
        if decision == 'n':
            raise ValueError("Not OK to rewrite existing file!")
        else:
            return  # OK to rewrite existing file
    return  # OK to rewrite existing file


def read_trans(fileobj):
    """
    A generator to read trans file.
    :param fileobj: an opened file object for reading trans file
    return: the following dictionary for each line:
    {
    text: content of the trans text
    id: id string
    }

    To parse id string into individual components of GUID use bbn_text_segment.split_guid()
    """
    sent_id_re = re.compile('^(?P<text>.*)\s+\((?P<id>\S+)\)$')
    line_no = 1
    for line in fileobj:
        line = line.rstrip()  # strip() would make it impossible to have empty trans output
        m = sent_id_re.match(line)
        if m is None:
            raise RuntimeError(f'Invalid trans format at line {line_no}')
        yield {
            'text': m.group('text'),
            'id': m.group('id')
        }
        line_no += 1


def is_xml(filename):
    return re.search(r"(\.xml$)|(\.xml\.gz$)|(\.xml\.bz2$)", filename) is not None


def is_json(filename):
    return re.search(r"(\.json$)|(\.json\.gz$)|(\.json\.bz2$)", filename) is not None


def segment_generator(files, stdin_type='json'):
    '''Return a segment generator that open and read in segments from a list of files.
    Both BBN XML segment and the BBN JSON segment are supported based on the filename suffix.

    If the file list is empty or if filename is '-' standard input (STDIN) will
    be used. When multiple '-' is given as filenames only the first is used as
    STDIN. The rest is discarded.
    If an input file has neither a xml or json extension, it is assumed to be a file containing
    a list of XML or JSON files (but not other filelists), one per line

    Parameters:
    files: list of filenames to read from. An empty list or a filename of '-' means reading from standard input.
    stdin_type: Accepts three values: "xml", "json" and "filelist".
                "filelist" means that the input file just contains paths to XML or JSON files

    Example:
    for segment in segment_generator(['input.xml.gz', 'input.json.gz']):
        print(segment)
    '''
    try:
        import bbn_text_segment
        import json_segment
    except ImportError:
        from . import bbn_text_segment
        from . import json_segment

    if not isinstance(files, list):
        raise ValueError(f"'files' argument must be a list of filenames but {files} is given")

    if not files:
        files = ('-',)

    if files.count('-') > 1:
        print(f"Ignoring multiple '-' used as input filenames. Only the first will be used as STDIN.", file=sys.stderr)
        stdin_seen = False
        files2 = []
        for fn in files:
            if fn == '-':
                if not stdin_seen:
                    files2.append(fn)
                    stdin_seen = True
            else:
                files2.append(fn)
        files = files2

    if stdin_type not in {"xml", "json", "filelist"}:
        raise ValueError("Please give a stdin_type from the set {'xml', 'json', 'filelist'}")

    for fn in files:
        input_is_xml = is_xml(fn) or (fn == '-' and stdin_type == "xml")
        input_is_json = is_json(fn) or (fn == '-' and stdin_type == "json")

        if input_is_xml:
            for segment in bbn_text_segment.BBNSegmentReader(fn):
                yield segment.segment()
        elif input_is_json:
            fobj = fopen(fn)
            for segment in json_segment.read_json(fobj):
                yield segment
        else:
            # input is a filelist
            with fopen(fn) as fobj:
                filelist = [l.strip() for l in fobj]
                if not all([is_xml(f) or is_json(f) for f in filelist]):
                    raise ValueError("Please provide a filelist that contains only XML or JSON files")
                for segment in segment_generator(filelist):
                    yield segment


def document_generator(files, doc_subset=None, doc_id_field="DOCUMENT_ID", stdin_type='json'):
    '''
    Returns a document at a time. A document is a list of consecutive segments.
    The code assumes that segments that belong to the same document are consecutive
    the input.

    Parameters:
    files: Same as in segment_generator
    doc_subset: If not None, documents with a document id not in this list (or set) will be ignored
    doc_id_field: The segment field to use to match document ids ("DOCUMENT_ID" by default)
    stdin_type: Same as in segment_generator
    '''

    document = []

    current_doc_id = None
    for segment in segment_generator(files, stdin_type):
        if doc_subset is not None and segment[doc_id_field] not in doc_subset:
            continue
        if segment[doc_id_field] != current_doc_id:
            if current_doc_id is not None:
                yield document
                document = []
            current_doc_id = segment[doc_id_field]
        document.append(segment)
    if len(document) > 0:
        yield document


def extract_lines_from_documents(paths_to_docs, segment_field=None):
    '''
    This function returns a list of lines, extracted from *all* given documents
    When a given document filename ends in .txt or .txt.gz, it just reads the file directly
    Otherwise, when the document is a json or xml document (or a filelist containing such documents)
    it expects a valid segment_field
    '''
    lines = []
    for path in paths_to_docs:
        if path.endswith(".txt") or path.endswith(".txt.gz"):
            with fopen(path) as f:
                sentences = [l.strip() for l in f]
        else:
            if segment_field is None:
                raise ValueError("Please provide a valid segment_field")
            sentences = [seg[segment_field] for seg in segment_generator(path)]
        lines.extend(sentences)

    return lines


def field_value_generator(files, segment_field):
    '''
    This function returns the requested field value segment-by-segment
    '''

    for segment in segment_generator(files):
        yield segment[segment_field]


class FieldValueGenerator:
    """Iterator that returns segment field values."""

    def __init__(self, files, segment_field, return_list_of_tokens=False):
        self.files = files
        self.segment_field = segment_field
        self.return_list_of_tokens = return_list_of_tokens
        self.generator = field_value_generator(self.files, self.segment_field)

    def __iter__(self):
        return self

    def __next__(self):
        if self.return_list_of_tokens:
            return next(self.generator).split()
        else:
            return next(self.generator)
