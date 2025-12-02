from diversity import compression_ratio, homogenization_score, ngram_diversity_score
from bbn_medic.metrics.PromptDiversityMeasureInterface import PromptDiversityMeasureInterface
from bbn_medic.common import Prompt


class PromptDiversityMeasures(PromptDiversityMeasureInterface):
    def __init__(self):
        pass

    def calculate_diversity(self, prompts: list[Prompt], params=None):
        prompt_list = [prompt.text for prompt in prompts]

        all_metrics = {}
        if params is None or params is not None and "compression_ratio" in params.get("metrics_to_compute", {}):
            # Inverse of data compressibility. The lower the score, the more diverse the data
            cr = compression_ratio(prompt_list, 'gzip')
            all_metrics["compression_ratio"] = cr

        if params is None or params is not None and "ngram_diversity" in params.get("metrics_to_compute", {}):
            # Type-token ratio but extended to n-grams. n=4 here
            nds = ngram_diversity_score(prompt_list, params["metrics_to_compute"]["ngram_diversity"])
            all_metrics["ngram_diversity"] = nds

        return all_metrics
