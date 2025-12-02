import argparse
import csv
import subprocess
from datetime import datetime, timedelta
import numpy as np
import os
import pandas
import sys
from pathlib import Path


def get_earliest_and_latest_files(directory):
    # Get a list of all files in the directory with their full paths
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if not files:
        return None  # Return None if the directory is empty

    # Find the file with the earliest creation time
    earliest_file = min(files, key=os.path.getctime)
    latest_file = max(files, key=os.path.getctime)
    return earliest_file, latest_file


def parse_start_time_from_logfile(log_file_path):
    """
    Reads the first line of a file, extracts a datetime string,
    and returns it as a datetime object.

    :param log_file_path: Path to the file containing the datetime string.
    :return: A datetime object parsed from the first line of the file.
    """
    try:
        with open(log_file_path, 'r') as file:
            # Read the first line of the file
            first_line_date_time = file.readline().strip().split()[5:9]
            #print(log_file_path, first_line_date_time)
            return parse_time_from_line(first_line_date_time)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return None


def parse_end_time_from_logfile(log_file_path):
    """
    Reads through a file to find the "newxg finished" marker, extracts a datetime string,
    and returns it as a datetime object.

    :param log_file_path: Path to the logfile containing the datetime string.
    :return: A datetime object parsed from the last lines of the file.
    """
    start_time = None
    machine_str = None
    end_time = None
    with open(log_file_path) as fp:
        for i in fp:
            i = i.strip()
            if "newxg finished" in i:
                if "successfully at" in i:
                    time_str = i.split("successfully at")[1].split("on machine")[0].strip()
                else:
                    print(f"warning: job in {log_file_path} did not finish successfully; returning end time anyway")
                    time_str = i.split("finished")[1].split("+++++++")[0].strip()
                end_time = datetime.strptime(time_str, "%a %b  %d %H:%M:%S %Y")
    return end_time


# expected formats:
# +++++++ newxg started at Wed Dec 18 02:45:33 2024 on machine hcmp-007 +++++++
# +++++++ newxg finished successfully at Wed Dec 18 02:46:59 2024 on machine hcmp-007 +++++++
# +++++++ newxg finished Wed Dec 25 00:06:52 2024 +++++++
def parse_time_from_line(logfile_line_containing_date_time):
    # Define the expected format of the datetime string
    datetime_format = "%b %d %H:%M:%S %Y"
    # Parse the datetime string into a datetime object
    extracted_datetime = datetime.strptime(" ".join(logfile_line_containing_date_time), datetime_format)
    return extracted_datetime


# Since the subprocess approach only works on Unix, there is a failure check and callout to python-only
def get_total_line_count_for_files_in_dir(directory, filename_suffix, exclusion_substring):

    command = f"( find {directory} -name '*{filename_suffix}' ! -name '*{exclusion_substring}' -print0 | xargs -0 cat ) | wc -l"
    #print(f"running command {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)        

    # Check if the command was successful
    if result.returncode != 0:
        print("WARNING: error attempting to run Unix subprocess; falling back to slower python-only approach")
        return get_total_line_count_for_files_in_dir_python_only(directory, filename_suffix, exclusion_substring)
    total_lines = int(result.stdout.strip())
    return total_lines

def get_total_line_count_for_files_in_dir_python_only(directory, filename_suffix, exclusion_substring):
    line_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(filename_suffix) and exclusion_substring not in file:
                file_path = os.path.join(root, file)
                # Open each file and count its lines
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count += sum(1 for _ in f)
    return line_count


def get_files_in_directory(directory, file_pattern=None):
    #print("get_files_in_directory: ", directory, file_pattern)
    files = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            if file_pattern is None or file_pattern in file_name:
                files.append(file_path)
    return files


def get_latest_edited_file(directory, exclusion_substrings=[]):
    # Get all files in the directory with their last modified time
    files_with_times = [
        (file, os.path.getmtime(os.path.join(directory, file)))
        for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file))
        and not any(suffix in file for suffix in exclusion_substrings)
    ]

    # Sort the files by last modified time in descending order
    files_with_times.sort(key=lambda x: x[1], reverse=True)

    # Display the files with their last modified times
    #for file, mtime in files_with_times:
    #    print(f"{file} - Last edited: {datetime.fromtimestamp(mtime)}")

    # Get the most recently edited file
    if files_with_times:
        latest_file = files_with_times[0]
        #print(f"Most recently edited file: {latest_file[0]} at {datetime.fromtimestamp(latest_file[1])}")
    else:
        print(f"warning: get_latest_edited_file found no files found in directory {directory}")
    return latest_file[0]

def add_header_row_for_timing_summary(data):
    # Add rows to a 2D list
    data.append(["TotalTime", "TotalSeconds", "TotalPrompts", "SecondsPerPrompt", "PromptsPerSecond"])


def add_row_to_timing_summary(data, total_time, total_seconds, total_prompts, seconds_per_prompt, prompts_per_second):
    data.append([total_time, total_seconds, total_prompts, seconds_per_prompt, prompts_per_second])


def add_header_row_for_stage_summary(data):
    # Add rows to the 2D list
    data.append(["Stage", "Min", "Max", "Median", "Mean", "StdDev"])


def add_header_row_for_stage_details(data):
    # Add rows to the 2D list
    data.append(["Stage", "Filename",
                 "StartTime", "EndTime", "Duration", "DurationSeconds"])  # Adding a header row])  # Adding a header row



def add_row_of_file_times(data, logfile_name_substring, f, file_start_time, file_end_time):
    # ["Category", "Min", "Max", "Mean", "StdDev", "Filename","StartTime", "EndTime", "Duration", "DurationSeconds"]
    readable_duration = file_end_time - file_start_time

    # Calculate duration in seconds
    duration_in_seconds = readable_duration.total_seconds()

    # ["Category", "Filename","StartTime", "EndTime", "Duration", "DurationSeconds"]
    data.append([logfile_name_substring, f,
                 file_start_time, file_end_time, readable_duration, duration_in_seconds])


def add_row_of_stage_times(data, logfile_name_substring, min, max, median, mean, stddev):
    # ["Stage", "Min", "Max", "Median", "Mean", "StdDev"]
    data.append([logfile_name_substring,min,max,median,mean,stddev])


def write_to_csv(data, filename):
    # make sure directory structure exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    df = pandas.DataFrame(data)
    df.to_csv(filename, index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_dir', type=str, required=True, help='Input experiment directory')
    parser.add_argument('--output_summary_csv', type=str, required=True,
                        help='Output csv file containing timing report summary for this experiment')
    parser.add_argument('--output_stage_summary_csv', type=str, required=False,
                        help='Output csv file containing timing report summary for this experiment by stage')
    parser.add_argument('--output_stage_details_csv', type=str, required=False,
                        help='Output csv file containing timing report breakdowns for this experiment by file')
    parser.add_argument('--logfile_substring_for_diversify_stage', type=str, required=False, default='diversify_prompts',
                        help='Substring to match logfiles containing times for the diversify step of prompt generation')
    parser.add_argument('--logfile_substring_for_standard_prompt_stage', type=str, required=False, default='generate_standard_prompts',
                        help='Substring to match logfiles containing times for the standard prompt generation step of prompt generation')
    parser.add_argument('--logfile_substring_for_desire_overgeneration_stage', type=str, required=False, default='overgenerate_desires',
                        help='Substring to match logfiles containing times for the desire overgeneration step of prompt generation')
    parser.add_argument('--dirname_for_prompt_outputs', type=str, required=False, default='prompts',
                        help='Name of expts directory containing all the overgenerated prompts')

    args = parser.parse_args()

    expt_path = Path(args.expt_dir)
    expt_name = expt_path.name


    # collect total number of prompts output by the system 
    total_number_of_prompts = get_total_line_count_for_files_in_dir(expt_path / "expts" / args.dirname_for_prompt_outputs , 
                                                                    filename_suffix=".jsonl", exclusion_substring="with_metrics.jsonl");

    if total_number_of_prompts < 1:
        sys.exit(f"ERROR: no prompts found in {expt_path}/expts/{args.dirname_for_prompt_outputs}")

    # get start and end times of entire expt
    expt_logfiles = expt_path / "logfiles"

    # earliest log file should be the "-prep.log"
    expt_prep_log_file = get_files_in_directory(expt_path / "logfiles", "-prep.log")
    if len(expt_prep_log_file) < 1:
        sys.exit(f"ERROR: no *-prep.log file found in {expt_path}/logfiles")
    earliest_log_file = expt_prep_log_file[0]

    # latest log file is determined by timestamp
    latest_log_file = get_latest_edited_file(expt_path / "logfiles", exclusion_substrings=["metrics_of_prompts","timing.log"])
    latest_log_file = expt_logfiles / latest_log_file

    expt_start_time = parse_start_time_from_logfile(earliest_log_file)
    expt_end_time = parse_end_time_from_logfile(latest_log_file)
    expt_duration = expt_end_time - expt_start_time

    # compute total number of prompts divided by total time to get the avg time per prompt
    seconds_per_prompt = expt_duration.total_seconds() / total_number_of_prompts
    prompts_per_second = total_number_of_prompts / expt_duration.total_seconds()
    timing_summary = []
    add_header_row_for_timing_summary(timing_summary)
    add_row_to_timing_summary(timing_summary, expt_duration, expt_duration.total_seconds(), total_number_of_prompts,
                              seconds_per_prompt, prompts_per_second)
    write_to_csv(timing_summary, args.output_summary_csv)

    logfile_name_substrings_of_interest = []
    # collect start/end for each inference run into an array for:
    # - desire generation
    logfile_name_substrings_of_interest.append(args.logfile_substring_for_desire_overgeneration_stage)
    # - prompt generation (generate_prompts)
    logfile_name_substrings_of_interest.append(args.logfile_substring_for_standard_prompt_stage)
    # - prompt diversification (diversify_prompts)
    logfile_name_substrings_of_interest.append(args.logfile_substring_for_diversify_stage)



    if not (args.output_stage_summary_csv or args.output_stage_details_csv):
        # no need to compute the other stats since we won't write them out
        exit(0)

    stage_summary = []
    add_header_row_for_stage_summary(stage_summary)
    stage_details = []
    add_header_row_for_stage_details(stage_details)

    for logfile_name_substring in logfile_name_substrings_of_interest:
        # grab all the logfiles containing this substring
        for f in get_files_in_directory(expt_logfiles, logfile_name_substring):
            # go through each logfile, extracting the start and end time
            file_start_time = parse_start_time_from_logfile(f)
            #print(f"{f} start time: {file_start_time}")
            file_end_time = parse_end_time_from_logfile(f)
            #print(f"{f}   end time: {file_end_time}")

            # store the start time, end time, and time difference (in seconds) in columns
            add_row_of_file_times(stage_details, logfile_name_substring, f, file_start_time, file_end_time)

    # compute the min, max, median, mean, and stddev
    df = pandas.DataFrame(stage_details)
    # set column names equal to values in row index position 0
    df.columns = df.iloc[0]
    # remove first row from DataFrame
    df = df[1:]
    #print(df)

    for category in logfile_name_substrings_of_interest:
        df_filtered = df[df["Stage"] == category]
        min = np.min(df_filtered["DurationSeconds"])
        max = np.max(df_filtered["DurationSeconds"])
        median = np.median(df_filtered["DurationSeconds"])
        mean = np.mean(df_filtered["DurationSeconds"])
        stddev = np.std(df_filtered["DurationSeconds"])
        # add all to the output array (which we will later write to csv)

        add_row_of_stage_times(stage_summary, category, min, max, median, mean, stddev)


    if args.output_stage_summary_csv:
        write_to_csv(stage_summary, args.output_stage_summary_csv)
    if args.output_stage_details_csv:
        write_to_csv(stage_details, args.output_stage_details_csv)


if __name__ == "__main__":
    main()
