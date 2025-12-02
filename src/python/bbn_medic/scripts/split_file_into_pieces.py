# This file was imported from pycube, hycube release R2022_07_06_1
# Initial modification between that version and this one: support for output_file_suffix below
import os
import sys
import logging

from bbn_medic.utils import file_utils
from bbn_medic.utils import read_params
from bbn_medic.utils.read_params import PARAM_TYPES

global logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("split_file_into_pieces.py")


def main():
    """
    Splits a file or list of files into pieces, line-by-line. This script exists
    because the unix command `split` does not allow an option to output a total
    of `n` splits, which is needed if we want to eagerly construct a runjobs
    dependency graph.

    Params:
        {
            file (required): the input file(s). Can be an array or a single string.
            num_pieces (required): an integer specifying the number of pieces to
                split the input into
            output_file_prefix (required): an arbitrary prefix (filepath) used for
                the outputs
            output_file_suffix (optional): an arbitrary suffix (string) used for the outputs
            starting_index (optional): The starting index for the output suffix (default: 0)
            gzip_subsets (optional): flag to compress output (default: 0)
        }
    """
    param_file = sys.argv[1]
    params = read_params.Params(param_file, required_params={'file', 'num_pieces', 'output_file_prefix'})

    starting_index = int(params.get("starting_index", 0))
    gzip_subsets = params.get("gzip_subsets", 0)
    output_file_suffix = params.get("output_file_suffix", "")

    if isinstance(params['file'], str):
        files = [params['file']]
    elif isinstance(params['file'], list):
        files = params['file']

    split_file(files, params.num_pieces, params.output_file_prefix, output_file_suffix, starting_index, gzip_subsets)


def split_file(files_to_split, num_pieces, output_file_prefix, output_file_suffix="", starting_index=0, gzip_subsets=0):
    if isinstance(files_to_split, str):
        files_to_split = [ files_to_split ]

    logger.info('Going through data once to count the total number of lines.')
    num_lines = 0
    for f in files_to_split:
        with file_utils.fopen(f, 'r') as f:
            for i, l in enumerate(f):
                pass
            num_lines += (i + 1)

    elems_per_subset = int(num_lines / num_pieces)
    residual = num_lines - elems_per_subset * num_pieces
    os.makedirs(os.path.dirname(output_file_prefix), exist_ok=True)
    logger.info(f'Splitting data with {num_lines} lines into {num_pieces} pieces each with {elems_per_subset} lines ...')
    open_args = {}
    if gzip_subsets:
        # default gzip.open() compress level is 9. Since this script is
        # usually used to split large files, we use faster compress level
        # There is usually a 4X speed improvement going from compress level 9 to 1, here we use 3.
        open_args = {'compresslevel': 3}
        output_files = [(output_file_prefix + str(x + starting_index) + output_file_suffix
                         + ".gz") for x in range(num_pieces)]
    else:
        output_files = [(output_file_prefix + str(x + starting_index)+ output_file_suffix)
                        for x in range(num_pieces)]

    # write an equal number of lines to each output file
    curr = 0
    num_files_written = 0
    outf = file_utils.fopen(output_files.pop(0), "w", **open_args)
    while files_to_split:
        inf = file_utils.fopen(files_to_split.pop(0), 'r')
        for line in inf:
            outf.write(line)
            curr += 1
            if (num_files_written >= residual and curr == elems_per_subset) or (num_files_written < residual and curr == (elems_per_subset + 1)):
                outf.close()
                num_files_written += 1
                if output_files:
                    outf = file_utils.fopen(output_files.pop(0), "w")
                else:
                    break
                curr = 0
        inf.close()
    # make sure output_files is now empty
    # assert len(output_files) == 0, "not all output files used! remaining: {}".format(output_files)
    # @hqiu I think we should allow empty batch file here
    for f in output_files:
        with file_utils.fopen(f,'w', **open_args) as fp:
            pass

if __name__ == '__main__':
    main()
