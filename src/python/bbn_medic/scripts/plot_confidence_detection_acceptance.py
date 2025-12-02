from bbn_medic.plots.ConfidenceDetectionAcceptancePlots import main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition_name_to_detection_jsonl_path", type=str, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.condition_name_to_detection_jsonl_path, args.title, args.output_path)
