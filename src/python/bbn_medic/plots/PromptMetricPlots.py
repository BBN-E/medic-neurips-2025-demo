import json
import matplotlib.pyplot as plt


class PromptMetricPlots:
    def __init__(self, summarized_metrics_files: list):
        self.summarized_metrics = []
        for metrics_file in summarized_metrics_files:
            with open(metrics_file) as f:
                json_line = f.readline()
                metrics = json.loads(json_line)
                self.summarized_metrics.append(metrics)

    def plot_metrics_vs_number_of_prompts(self):
        # We will create 4 plots, with the y-axis of each showing the following:
        # perplexity, coverage, faithfulness, compression_ratio

        sorted_summarized_metrics = sorted(self.summarized_metrics, key=lambda m: m['number_of_prompts'])
        number_of_prompts = [m['number_of_prompts'] for m in sorted_summarized_metrics]
        perplexity_values = [m['mean_perplexity'] for m in sorted_summarized_metrics]
        coverage_values = [m['mean_coverage'] for m in sorted_summarized_metrics]
        faithfulness_values = [m['mean_faithfulness'] for m in sorted_summarized_metrics]
        compression_ratio_values = [m['compression_ratio'] for m in sorted_summarized_metrics]

        plt.figure()
        plt.subplot(221)
        plt.plot(number_of_prompts, perplexity_values)
        plt.title('perplexity vs # prompts')
        plt.grid(True)

        plt.subplot(222)
        plt.plot(number_of_prompts, coverage_values)
        plt.title('coverage vs # prompts')
        plt.grid(True)

        plt.subplot(223)
        plt.plot(number_of_prompts, faithfulness_values)
        plt.title('faithfulness vs # prompts')
        plt.grid(True)

        plt.subplot(224)
        plt.plot(number_of_prompts, compression_ratio_values)
        plt.title('compression_ratio vs # prompts')
        plt.grid(True)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()
