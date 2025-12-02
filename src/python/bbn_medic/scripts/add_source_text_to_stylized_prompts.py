import sys

from bbn_medic.io.io_utils import JSONLGenerator, fopen

if __name__ == "__main__":

    """
    This script adds the full prompt source text to stylized prompts. To do this, 
    it looks in the metadata of each stylized prompt, gets the source prompt ID, 
    searches through the standard prompts file to look up the full text of the 
    standard version of the prompt, and then writes out a version of the stylized 
    prompt file with the full text. 

    Our normal workflow doesn't include the full text of the source prompt because 
    it makes the files unnecessarily large, but when releasing stylized prompts to 
    collaborators it makes things simpler to include the full standard prompt text 
    alongside the stylized version.

    See the section entitled 'Adding Source Text to Stylized Prompts' on the following collab page for how to run:
      https://collab.bbn.com/confluence/display/ARPAHCRE/prompt_generation_figure_3+experiment      
    """

    filtered_stylized_prompts_file = sys.argv[1]
    standard_prompts_file = sys.argv[2]
    output_file = sys.argv[3]

    source_text = dict()
    for obj in JSONLGenerator.read(standard_prompts_file):
        source_text[obj.id] = obj.text

    with fopen(output_file, "w") as g:
        for obj in JSONLGenerator.read(filtered_stylized_prompts_file):
            if "Prompt_source_id" in obj.metadata:
                source_id = obj.metadata["Prompt_source_id"]
                obj.metadata["Prompt_source_text"] = source_text[source_id]
            g.write(obj.to_json() + "\n")
