import json
import sys
from collections import defaultdict


# This script expects a jsonl file which is sorted by threshold. Here's an example one-liner for how to sort:
#
# # use this version to inspect the scores
# zcat mh/filtered_prompts.thr_0.3.jsonl.gz | grep '"faithfulness"' | perl -e 'while(<>){if(m/"perplexity": (\S+),/){$ppl=$1;} if(m/"coverage": (\S+),/){$cov=$1;} if(m/"faithfulness": (\S+)\}\}/){$faith=$1;} print (-0.001*$ppl+0.5*$cov+0.5*$faith, " $_");}' | sort -rg -k 1,1 > mh/filtered_prompts.thr_0.3.sorted-by-score.txt
#
# # this version of the command emits valid jsonl, suitable as input to this script
# $ zcat mh/filtered_prompts.thr_0.3.jsonl.gz | grep '"faithfulness"' | perl -e 'while(<>){if(m/"perplexity": (\S+),/){$ppl=$1;} if(m/"coverage": (\S+),/){$cov=$1;} if(m/"faithfulness": (\S+)\}\}/){$faith=$1;} print (-0.001*$ppl+0.5*$cov+0.5*$faith, " $_");}' | sort -rg -k 1,1  | awk '{$1=""; sub(/^ /, ""); print}' > mh/filtered_prompts.thr_0.3.sorted.jsonl 

def parse_jsonl_file(filepath, num_examples_to_collect=3):
    collected_restylings  = defaultdict(lambda: defaultdict(list))
    with open(filepath, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            try:
                data = json.loads(line)

                # Safely get the nested fields
                metadata = data.get("metadata", {})
                prompt_source_id = metadata.get("Prompt_source_id")
                style = metadata.get("style")

                # Only add if we haven't yet collected enough
                if len(collected_restylings[prompt_source_id][style]) < num_examples_to_collect:
                    collected_restylings[prompt_source_id][style].append(data)
                    #print(data)

                #print(f"Line {line_num}:")
                #print(f"  Prompt_source_id: {prompt_source_id}")
                #print(f"  style: {style}")
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
    return collected_restylings


def save_collected_data_as_jsonl(collected_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt_source_id, styles in collected_data.items():
            for style, entries in styles.items():
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')        

                    
if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: python collect_n_restylings.py prompts.jsonl output.jsonl [Optional: num-examples-to-collect]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_examples_to_collect = 3
    if len(sys.argv) == 4:
        num_examples_to_collect = int(sys.argv[3])
            
    collected_restylings = parse_jsonl_file(input_file, num_examples_to_collect)
    save_collected_data_as_jsonl(collected_restylings, output_file)

    for prompt_source_id, styles in collected_restylings.items():
        for style, entries in styles.items():
            print(f"{prompt_source_id} / {style}: {len(entries)} example(s) collected")
