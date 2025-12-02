import sys

from bbn_medic.common.Desire import Desire
from  bbn_medic.utils.file_utils import read_list
from bbn_medic.io.io_utils import fopen

if __name__ == "__main__":
    """
    This script converts externally provided desires to jsonl, 
    to be used in lieu of overgenerating desires.
    See the information on mode 2 on the followinfg collab page: 
      https://collab.bbn.com/confluence/display/ARPAHCRE/prompt_generation_figure_3+experiment
    """

    for infile in sys.argv[1:]:
        outfile = infile.replace(".txt", ".jsonl")

        with fopen(outfile, "w") as g:

            for line in read_list(infile):
                text, desireid = line.split('\t')
                desire = Desire(text=text,
                                id=desireid)
                g.write(desire.to_json() + "\n")
