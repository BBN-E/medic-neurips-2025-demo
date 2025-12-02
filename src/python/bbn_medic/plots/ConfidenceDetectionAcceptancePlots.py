import json
import os

import matplotlib.pyplot as plt


def plot_cnt(scale, model_name_to_cnt, y_text, title, output_path):
    # markings = {
    #     "marker": ["o", "s", "^"],
    #     "linestyle": ["-", "--", "-."]
    # }
    plt.figure(figsize=(8, 5))
    for idx, (model_name, cnt_array) in enumerate(sorted(model_name_to_cnt.items(), key=lambda x: x[0])):
        plt.plot(scale, cnt_array, linewidth=2,
                 label=model_name)
    plt.xlabel("Confidence threshold")
    plt.ylabel(f"Accepted {y_text}")
    plt.title(title)
    # plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main(condition_name_to_detection_jsonl_path, title, output_path):
    step = 10
    scale = [i / step for i in range(0, step + 1, 1)]
    condition_name_to_bin_cnt = {}
    with open(condition_name_to_detection_jsonl_path) as fp:
        for condition_name, detection_jsonl_path in json.load(fp).items():
            bin_cnt_list = condition_name_to_bin_cnt.setdefault(condition_name, [0 for _ in range(len(scale))])
            with open(detection_jsonl_path) as fp2:
                for i in fp2:
                    j = json.loads(i)
                    confidence = j['confidence']
                    for idx, s in enumerate(scale):
                        if confidence >= s:
                            bin_cnt_list[idx] += 1
                        else:
                            break
    plot_cnt(scale, condition_name_to_bin_cnt, title, title,
             output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition_name_to_detection_jsonl_path", type=str, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.condition_name_to_detection_jsonl_path, args.title, args.output_path)
