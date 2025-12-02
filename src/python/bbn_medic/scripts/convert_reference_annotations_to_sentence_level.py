import json
import re
import sys
from tqdm import tqdm

from bbn_medic.io.io_utils import JSONLGenerator, fopen


def find_span_intersection(start, end, sorted_spans):
    lb = 0
    ub = len(sorted_spans)
    while(lb < ub - 1):
        mid = int((lb + ub) / 2)
        if start > sorted_spans[mid]["start_char"]:
            lb = mid
        elif start < sorted_spans[mid]["start_char"]:
            ub = mid
        else:
            lb = mid
            ub = lb
    # The above code makes sure that start is between sorted_spans[lb] and sorted_spans[lb + 1]
    for ub in range(lb, len(sorted_spans)):
        if end <= sorted_spans[ub]["end_char"]:
            break
    return lb, ub + 1


def main():
    annotations_json = sys.argv[1]
    answers_file = sys.argv[2]  # jsonl or filelist containing pointers to jsonl files
    annotations_output_jsonl = sys.argv[3]

    answer_text_to_answer_id = {}
    answer_id_to_answer_segments = {}
    for obj in tqdm(JSONLGenerator.read(answers_file, segment_into_sentences=True), desc="Loading answers file"):
        if obj.type != "Answer":
            raise ValueError(f"Unsupported type {obj.type} in file {answers_file}")
        answer_id = obj.id
        answer_text = obj.text
        answer_text_to_answer_id[answer_text] = answer_id
        if "segments" in obj.metadata:
            answer_id_to_answer_segments[answer_id] = obj.metadata["segments"]
        else:
            raise ValueError(f"Answer with id {answer_id} isn't segmented into sentences")

    with fopen(annotations_json) as f, fopen(annotations_output_jsonl, "w") as g:
        annotations_obj = json.load(f)
        for entry in tqdm(annotations_obj, desc="Processing original annotations"):
            segment_info = {}
            response = entry["data"]["response"]
            if response not in answer_text_to_answer_id:
                continue
            answer_id = answer_text_to_answer_id[response]
            segments = answer_id_to_answer_segments[answer_id]
            sorted_segment_spans = sorted(segments, key=lambda x: x["start_char"])
            for segment in segments:
                segment_info[(segment["start_char"], segment["end_char"])] = segment
            for completion in entry["completions"]:
                for result in completion["result"]:
                    if "value" not in result:
                        continue
                    if "labels" not in result["value"]:
                        continue
                    harm_level = result["value"]["labels"][0].lower()
                    snippet_start = result["value"]["start"]
                    snippet_end = result["value"]["end"]
                    affected_segment_span_indices = find_span_intersection(snippet_start, snippet_end, sorted_segment_spans)
                    for span_index in range(affected_segment_span_indices[0], affected_segment_span_indices[1]):
                        segment_str = segment_info[(sorted_segment_spans[span_index]["start_char"], sorted_segment_spans[span_index]["end_char"])]["text"]
                        # The following if statement gets rid of badly segmented "sentences" containing a single '*'
                        if segment_str == "*" or re.search(r'^[0-9][0-9]*[.:]\s*$', segment_str):
                            continue
                        hallucination_entry = {
                            "type": "Hallucination",
                            "detector_type": "annotation",
                            "snippet": segment_str,
                            "harm_level": harm_level,
                            "answer_id": answer_id,
                            "segment_id": span_index,
                            "explanation": None,
                            "prompt_id": None,
                            "confidence": None,
                            "model_name": None,
                            "metadata": {
                                "segment_info": sorted_segment_spans[span_index]
                            }
                        }
                        g.write(json.dumps(hallucination_entry) + "\n")


if __name__ == "__main__":
    main()
