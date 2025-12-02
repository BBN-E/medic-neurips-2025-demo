import datetime
import os.path
import pathlib
import re

from bbn_medic.scripts.compute_prompt_generation_time_statistics import add_header_row_for_timing_summary, \
    add_row_to_timing_summary, write_to_csv


def parse_start_end_time_from_single_logfile(logfile_path):
    start_time_re = re.compile(r"\+\+\+\+\+\+\+ newxg started at ([\w :]+) on machine ([\w.-]+) \+\+\+\+\+\+\+")
    end_time_re = re.compile(
        r"\+\+\+\+\+\+\+ newxg finished successfully at ([\w :]+) on machine ([\w.-]+) \+\+\+\+\+\+\+")
    first_row = None
    last_row = None
    with open(logfile_path, 'r') as file:
        for idx, line in enumerate(file):
            if first_row is None and "newxg started" in line:
                first_row = line
            if "newxg finished" in line:
                last_row = line
    start_match = start_time_re.match(first_row)
    end_match = None
    if last_row is not None:
        end_match = end_time_re.match(last_row)
    start_time = datetime.datetime.strptime(start_match.group(1), "%a %b  %d %H:%M:%S %Y")
    if end_match is None:
        return start_time, None
    end_time = datetime.datetime.strptime(end_match.group(1), "%a %b  %d %H:%M:%S %Y")
    return start_time, end_time


def sum_up_num_of_prompts(counting_list_path):
    total_number_of_prompts = 0
    with open(counting_list_path, 'r') as file:
        for i in file:
            i = i.strip()
            with open(i) as fp2:
                total_number_of_prompts += int(fp2.read())
    return total_number_of_prompts


def main(exp_dir, exp, ending_job_name, prompt_counting_list_path, output_summary_csv):
    total_number_of_prompts = sum_up_num_of_prompts(prompt_counting_list_path)
    logfiles_dir = pathlib.Path(exp_dir) / "logfiles"
    ending_log_path = logfiles_dir / os.path.dirname(ending_job_name) / f"{exp}-{os.path.basename(ending_job_name)}.log"
    _, cutoff_time = parse_start_end_time_from_single_logfile(ending_log_path)
    assert cutoff_time is not None
    total_seconds = 0
    for root, dirs, files in os.walk(logfiles_dir):
        for file in files:
            if file.endswith(".log"):
                start_time, end_time = parse_start_end_time_from_single_logfile(os.path.join(root, file))
                if start_time > cutoff_time:
                    continue
                if end_time is None:
                    raise ValueError(f"Bug file {os.path.join(root, file)}")
                total_seconds += (end_time - start_time).total_seconds()
    # compute total number of prompts divided by total time to get the avg time per prompt
    seconds_per_prompt = total_seconds / total_number_of_prompts
    prompts_per_second = total_number_of_prompts / total_seconds
    timing_summary = []
    add_header_row_for_timing_summary(timing_summary)
    add_row_to_timing_summary(timing_summary, datetime.timedelta(seconds=total_seconds), total_seconds,
                              total_number_of_prompts,
                              seconds_per_prompt, prompts_per_second)
    write_to_csv(timing_summary, output_summary_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--ending_job_name", type=str, required=True)
    parser.add_argument("--prompt_counting_list_path", type=str, required=True)
    parser.add_argument("--output_summary_csv", type=str, required=True)
    args = parser.parse_args()
    main(args.exp_dir, args.exp, args.ending_job_name, args.prompt_counting_list_path, args.output_summary_csv)
