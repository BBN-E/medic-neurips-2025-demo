import sys
from bbn_medic.io.io_utils import JSONLGenerator, fopen

"""
Reads a jsonl file and writes out the type and text fields. Intended for 
comparing desire and prompt files that may have differing ID and other 
fields.
"""

def writeout(in_file):
    out_file = f"{in_file}.text"
    with fopen(out_file, "w") as g:
        for obj in JSONLGenerator.read(in_file):
            if hasattr(obj, 'type') and hasattr(obj, 'text'):
                g.write(f"{obj.type} {obj.text}\n")

            
if __name__ == "__main__":
    writeout(sys.argv[1])
