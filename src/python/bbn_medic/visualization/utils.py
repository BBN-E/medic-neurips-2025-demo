import sys
import json
import os
import subprocess

from bbn_medic.io.io_utils import fopen


class CaptureStderr:

    _logfile = None

    def start(output_path):
        CaptureStderr.logfile = open(f"{output_path}stderr.log", "w")
        sys.stderr.write = CaptureStderr.logfile.write

    def flush():
        if CaptureStderr.logfile is not None:
            CaptureStderr.logfile.flush()

    def get_logfile():
        if CaptureStderr.logfile is None:
            
            raise("CaptureStderr not initialized; call CaptureStderr.start(output_path) first")
        
        return CaptureStderr.logfile.name

def setup_filtering(prompts_filelist, standard_prompts_file_with_metrics,
                    diversified_prompts_file_with_metrics, filtered_weights_file,
                    perplexity, faithfulness, coverage):
    with fopen(prompts_filelist, "w") as g:
        g.write(f"{standard_prompts_file_with_metrics}\n")
        g.write(f"{diversified_prompts_file_with_metrics}\n")
    with fopen(filtered_weights_file, "w") as g:
        g.write(f"perplexity {perplexity}\n")
        g.write(f"faithfulness {faithfulness}\n")
        g.write(f"coverage {coverage}\n")

def load_configuration(config_file):
    output_path = ""
    model_parent_directory = ""

    for line in fopen(config_file):
        try:
            j = json.loads(line)

            if "output_path" in j:
                output_path = j["output_path"]
            if "model_parent_directory" in j:
                model_parent_directory = j["model_parent_directory"]

        except Exception as e:
            raise Exception("Malformed config file")
                
    if output_path == "":
        raise Exception("Output path not in config file")
    if model_parent_directory == "":
        raise Exception("Model path not in config file")
    return (output_path, model_parent_directory)


def line_count(infile):
    if os.path.exists(infile) and os.path.isfile(infile):
        return int(subprocess.check_output(['wc', '-l', infile]).split()[0].decode('UTF-8'))
    else:
        return None
    
