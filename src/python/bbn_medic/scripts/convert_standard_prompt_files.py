import sys

from bbn_medic.common.Prompt import Prompt
from  bbn_medic.utils.file_utils import read_list
from bbn_medic.io.io_utils import fopen

if __name__ == "__main__":

    """
    This script converts externally provided standard prompts to jsonl, 
    to be used in lieu of generating prompts from  desires.
    See the information on mode 2 on the followinfg collab page: 
      https://collab.bbn.com/confluence/display/ARPAHCRE/prompt_generation_figure_3+experiment
    """

    for infile in sys.argv[1:]:
        outfile = infile.replace(".txt", ".jsonl")


        desire_index = -1
        prev_desire = ""
        with fopen(outfile, "w") as g:

            for line in read_list(infile):

                promptid, text, desireid = line.split('\t')

                if desireid != prev_desire:
                    if prev_desire != "":
                        h.close
                    desire_index = desire_index + 1
                    h = fopen("prompts_d" + str(desire_index) + ".jsonl", "w")
                    prev_desire = desireid
                    
                
                prompt = Prompt(text=text,
                                id=promptid,
                                metadata={"Desire_metadata": None,
                                          "Desire_source_id": desireid,
                                          "style": "standard",
                                          "generation_prompt_id": "default_prompt"})
                g.write(prompt.to_json() + "\n")
                h.write(prompt.to_json() + "\n")
        h.close()

