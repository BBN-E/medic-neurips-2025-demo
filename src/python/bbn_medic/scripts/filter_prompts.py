import json
import argparse

from bbn_medic.metrics.FilterPrompts import FilterPrompts
from bbn_medic.io.io_utils import fopen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_file', type=str, help='Input jsonl file (or filelist) with prompts that contain metrics', default=None)
    parser.add_argument('--output_file', type=str, help="Path to the output jsonl file, with the filtered prompts", default=None)
    parser.add_argument('--weight_file', type=str, help="Path to a two-column file containing the coefficients for scoring each prompt", default=None)
    parser.add_argument('--threshold', type=float, help='Threshold determining the minimum score for keeping a prompt', default=None)
    parser.add_argument('--ignore_non_existent_metrics', type=bool, help='If False, the code will crash if the prompt does not contain a metric for which we have a weight', default=True)
    parser.add_argument('--renormalize_weights', type=bool, help='If True, the computed score will be divided by the sum of the weights, to normalize to 1', default=True)
    parser.add_argument('--remove_duplicates', type=bool, help='If True, the final file will have all duplicate prompts removed', default=False)
    parser.add_argument('--param_file', type=str, help='Parameter file containing all the above parameters', default=None)

    args = parser.parse_args()

    if args.param_file is not None:
        with fopen(args.param_file) as f:
            args_dict = json.load(f)
            args.weight_file = args_dict["weight_file"]
            args.output_file = args_dict["output_file"]
            args.prompts_file = args_dict["prompts_file"]
            args.ignore_non_existent_metrics = args_dict.get("ignore_non_existent_metrics", True)
            args.threshold = args_dict["threshold"]
            args.remove_duplicates = args_dict.get("remove_duplicates", False)


    FilterPrompts.filter(weight_file=args.weight_file,
                         output_file=args.output_file,
                         prompts_file=args.prompts_file,
                         ignore_non_existent_metrics=args.ignore_non_existent_metrics,
                         renormalize_weights=args.renormalize_weights,
                         threshold=args.threshold,
                         remove_duplicates=args.remove_duplicates)


    
