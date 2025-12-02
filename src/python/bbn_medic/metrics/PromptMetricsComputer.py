import json
import os
import sys
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from bbn_medic.common.Patient import AtomicPatientExpression
from bbn_medic.common.Prompt import Prompt
from bbn_medic.io.io_utils import fopen, JSONLGenerator
from bbn_medic.metrics.DesiresVsPromptsSimilarityCoverageMeasure import DesiresVsPromptsSimilarityCoverageMeasure
from bbn_medic.metrics.GPT2Perplexity import GPT2Perplexity
from bbn_medic.metrics.PromptDiversityMeasures import PromptDiversityMeasures
from bbn_medic.metrics.PromptToPromptSimilarityFaithfulnessMeasure import PromptToPromptSimilarityFaithfulnessMeasure
from bbn_medic.visualization.utils import line_count


class PromptMetricsComputer:
    @staticmethod
    def compute_metrics(
            prompts_file,
            output_file=None,
            perplexity_model=None,
            coverage_model=None,
            faithfulness_model=None,
            compute_compression_ratio=False,
            desires_file=None,
            source_prompts_file=None,
            output_file_with_summarized_metrics=None,
            compute_ngram_diversity=False,
            max_ngram_order=4
    ):
        compute_per_prompt_metrics = False

        perplexity_computer = None
        if perplexity_model is not None:
            if "gpt2" in perplexity_model:
                perplexity_computer = GPT2Perplexity(perplexity_model)
                compute_per_prompt_metrics = True
            else:
                raise ValueError(f"Model {perplexity_model} not supported")

        coverage_computer = None
        if coverage_model is not None:
            if desires_file is not None:
                desires = list(JSONLGenerator.read(desires_file))
                coverage_computer = DesiresVsPromptsSimilarityCoverageMeasure(coverage_model, desires)
            else:
                raise ValueError("The coverage metric cannot be computed without a desires_file")
            compute_per_prompt_metrics = True

        faithfulness_computer = None
        if faithfulness_model is not None:
            if source_prompts_file is not None:
                source_prompts = list(JSONLGenerator.read(source_prompts_file))
                faithfulness_computer = PromptToPromptSimilarityFaithfulnessMeasure(faithfulness_model, source_prompts)
            else:
                raise ValueError("The faithfulness metric cannot be computed without a source_prompts_file")
            compute_per_prompt_metrics = True

        if compute_per_prompt_metrics:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with fopen(output_file, "w") as g:
                num_lines = line_count(prompts_file)
                for prompt in tqdm(JSONLGenerator.read(prompts_file), file=sys.stdout, desc="Computing metrics",
                                   total=num_lines):
                    assert isinstance(prompt, Prompt) or isinstance(prompt, AtomicPatientExpression)
                    if perplexity_computer:
                        perplexity = perplexity_computer.calculate_perplexity([prompt])
                        prompt.metadata["perplexity"] = float(perplexity[0])
                    if coverage_computer:
                        coverage = coverage_computer.calculate_coverage([prompt])
                        prompt.metadata["coverage"] = coverage[0]
                    if faithfulness_computer:
                        faithfulness = faithfulness_computer.calculate_faithfulness([prompt])
                        prompt.metadata["faithfulness"] = faithfulness[0]
                    if isinstance(prompt, AtomicPatientExpression):
                        g.write(f"{json.dumps(prompt.model_dump(), ensure_ascii=False)}\n")
                    elif isinstance(prompt, Prompt):
                        g.write(f"{prompt.to_json()}\n")
                    else:
                        raise NotImplementedError()

        if output_file_with_summarized_metrics is not None:
            os.makedirs(os.path.dirname(output_file_with_summarized_metrics), exist_ok=True)
            with fopen(output_file_with_summarized_metrics, "w") as g:
                prompt_lists = list(JSONLGenerator.read(prompts_file))

                if compute_compression_ratio or compute_ngram_diversity:
                    params = defaultdict(dict)
                    if compute_compression_ratio:
                        params["metrics_to_compute"]["compression_ratio"] = True
                    if compute_ngram_diversity:
                        params["metrics_to_compute"]["ngram_diversity"] = max_ngram_order
                    prompt_diversity_computer = PromptDiversityMeasures()
                    diversity_metrics = prompt_diversity_computer.calculate_diversity(prompt_lists, params)
                else:
                    diversity_metrics = {}

                diversity_metrics['number_of_prompts'] = len(prompt_lists)
                # If perplexity scores exist in the prompt already, compute mean, median, sd
                if all('perplexity' in p.metadata for p in prompt_lists):
                    all_values = [p.metadata['perplexity'] for p in prompt_lists]
                    diversity_metrics['mean_perplexity'] = np.mean(all_values)
                    diversity_metrics['median_perplexity'] = np.median(all_values)
                    diversity_metrics['sd_perplexity'] = np.std(all_values)

                # If coverage scores exist in the prompt already, compute mean, median, sd
                if all('coverage' in p.metadata for p in prompt_lists):
                    all_values = [p.metadata['coverage'] for p in prompt_lists]
                    diversity_metrics['mean_coverage'] = np.mean(all_values)
                    diversity_metrics['median_coverage'] = np.median(all_values)
                    diversity_metrics['sd_coverage'] = np.std(all_values)

                # If faithfulness scores exist in the prompt already, compute mean, median, sd
                if any('faithfulness' in p.metadata for p in prompt_lists):
                    all_values = [p.metadata['faithfulness'] for p in prompt_lists if 'faithfulness' in p.metadata]
                    diversity_metrics['mean_faithfulness'] = np.mean(all_values)
                    diversity_metrics['median_faithfulness'] = np.median(all_values)
                    diversity_metrics['sd_faithfulness'] = np.std(all_values)

                g.write(json.dumps(diversity_metrics) + "\n")
