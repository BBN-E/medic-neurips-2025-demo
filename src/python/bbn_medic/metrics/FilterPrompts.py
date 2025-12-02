import json
import os
import sys

from tqdm.auto import tqdm

from bbn_medic.common.Patient import AtomicPatientExpression
from bbn_medic.common.Prompt import Prompt
from bbn_medic.io.io_utils import fopen, JSONLGenerator
from bbn_medic.utils.text_utils import get_id_from_text


class FilterPrompts:
    @staticmethod
    def filter(weight_file,
               output_file,
               prompts_file,
               ignore_non_existent_metrics,
               renormalize_weights,
               threshold,
               remove_duplicates,
               progress_tag=""):

        weights = {}
        if weight_file is not None:
            with fopen(weight_file) as f:
                for line in f:
                    fields = line.split()
                    weights[fields[0]] = float(fields[1])

        all_prompt_texts = set()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        bloom_filter_mask = 0xffffffffff
        with fopen(output_file, "w") as g:
            if progress_tag != "":
                tag = f"Filtering prompts for {progress_tag}"
            else:
                tag = "Filtering prompts"
            for prompt in tqdm(JSONLGenerator.read_all(prompts_file), file=sys.stdout, desc=tag):
                assert isinstance(prompt, Prompt) or isinstance(prompt, AtomicPatientExpression)
                total_score = 0
                total_weight = 0
                for metric in weights:
                    if metric in prompt.metadata:
                        total_score += weights[metric] * prompt.metadata[metric]
                        total_weight += weights[metric]
                    elif not bool(ignore_non_existent_metrics):
                        raise ValueError(f"Metric {metric} is missing from the prompt with id {prompt.id}")
                if renormalize_weights and total_weight != 0:
                    total_score /= total_weight
                if total_score >= threshold:
                    if remove_duplicates:
                        text_hash = get_id_from_text(prompt.text, mask=bloom_filter_mask, return_hex=True)
                    else:
                        text_hash = "DUMMY_ID" # Use same id for all string
                    if text_hash not in all_prompt_texts:
                        if isinstance(prompt, Prompt):
                            g.write(prompt.to_json() + "\n")
                        elif isinstance(prompt, AtomicPatientExpression):
                            g.write(f"{json.dumps(prompt.model_dump(), ensure_ascii=False, sort_keys=True)}\n")
                        if remove_duplicates:
                            all_prompt_texts.add(text_hash)
