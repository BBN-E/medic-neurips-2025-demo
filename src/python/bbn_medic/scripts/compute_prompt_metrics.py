import argparse

from bbn_medic.metrics.PromptMetricsComputer import PromptMetricsComputer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, help='Input jsonl file containing prompts (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--desires_file', type=str, default=None, help='Input jsonl file containing desires (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--source_prompts_file', type=str, default=None, help='Input jsonl file containing source prompts for the faithfulness metric (can also be jsonl.gz or jsonl.bz2)')
    parser.add_argument('--coverage_model', type=str, default=None, help='The model to use for computing embeddings used in the coverage metric')
    parser.add_argument('--faithfulness_model', type=str, default=None, help='The model to use for computing embeddings used in the faithfulness metric')
    parser.add_argument('--perplexity_model', type=str, default=None, help='The model to use for computing perplexity')
    parser.add_argument('--compute_compression_ratio', action='store_true', default=False, help='Compute compression ratio and other related metrics')
    parser.add_argument('--compute_ngram_diversity', action='store_true', default=False, help='Compute ngram diversity metric')
    parser.add_argument('--max_ngram_order', type=int, default=4, help='The maximum order of the ngram diversity metric')
    parser.add_argument('--output_file', type=str, help='Output jsonl file containing the coverage of the prompts')
    parser.add_argument('--output_file_with_summarized_metrics', type=str, default=None,
                        help='Output file where summary metrics (computed over the entire set of prompts) are saved')

    args = parser.parse_args()

    PromptMetricsComputer.compute_metrics(
        prompts_file=args.prompts_file,
        desires_file=args.desires_file,
        source_prompts_file=args.source_prompts_file,
        coverage_model=args.coverage_model,
        faithfulness_model=args.faithfulness_model,
        perplexity_model=args.perplexity_model,
        compute_compression_ratio=args.compute_compression_ratio,
        compute_ngram_diversity=args.compute_ngram_diversity,
        max_ngram_order=args.max_ngram_order,
        output_file=args.output_file,
        output_file_with_summarized_metrics=args.output_file_with_summarized_metrics
    )

if __name__ == "__main__":
    main()
