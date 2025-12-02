import pandas as pd
import json

from bbn_medic.io.io_utils import DesireJSONLGenerator, JSONLGenerator, fopen


class DisplayTables:

    _show_output = True

    # Dict for storing prompt text by ID
    _prompt_text_dict = dict()

    _metrics_file_paths = []

    def show_output(show, notebook_path):
        DisplayTables._show_output = show
        DisplayTables._notebook_path = notebook_path

    def _show_short_status():
        print("Done")

    # Function for storing text by ID
    def store_text_by_id(data_file):
        data = JSONLGenerator.read(data_file)
        for datum in data:
            DisplayTables._prompt_text_dict[datum.id] = datum.text

    def _common_table_styles():
        table_styles = [
            {'selector': 'th',
             'props': [
                 ('background-color', 'gray'),
                 ('color', 'white'),
                 ('border-color', 'black'),
                 ('border-style ', 'solid'),
                 ('border-width','1px'),
                 ('text-align', 'left')]},
            {'selector': 'td',
             'props': [
                 ('border-color', 'black'),
                 ('border-style ', 'solid'),
                 ('border-width','1px'),
                 ('text-align', 'left')]},                            
            {'selector': '.row_heading',
             'props': [('display', 'none')]},
            {'selector': '.blank.level0',
             'props': [('display', 'none')]}
        ]
        return table_styles


    def _display_output(data_file, jgen_class, fields, labels):
        """
        General method for displaying json desires or prompt info in a table in a jupyter notebook
        """
        # Read data and prep formatting

        if not DisplayTables._show_output:
            return DisplayTables._show_short_status()
        
        data = jgen_class.read(data_file)
        values = []
        right_justified_cols = []
        right_justified_col_index = 2
        four_places = set(["Coverage", "Faithfulness"])
        for datum in data:
            datum_vals = []
            for field, label in zip(fields, labels):
                # get the value
                if field == "source_prompt_text_from_dict":
                    val = DisplayTables._prompt_text_dict[datum.metadata["Prompt_source_id"]]
                elif isinstance(field, list):
                    # field[0] is a dict and field[1] is the key
                    info = getattr(datum, field[0])
                    key = field[1]
                    if key in info:
                        val = info[key]
                    else: 
                        val = "n/a"
                else:
                    # field is a simple value
                    val = getattr(datum, field)
                # format float values
                if label == "Perplexity":
                    if isinstance(val, float):
                        val = f"{val:.1f}"
                    right_justified_cols.append(right_justified_col_index)
                elif label == "Confidence":
                    if isinstance(val, float):
                        val = f"{val:.2f}"
                    right_justified_cols.append(right_justified_col_index)
                elif label in four_places: 
                    if isinstance(val, float):
                        val = f"{val:.4f}"
                    right_justified_cols.append(right_justified_col_index)
                    #right_justified_col_index += 1
                    #val = str(round(val, 4))
                right_justified_col_index += 1
                # store the value
                datum_vals.append(val)
            values.append(datum_vals)

        # Set up the style of the output tables
        table_styles = DisplayTables._common_table_styles()
        for col in right_justified_cols:
            table_styles.append({'selector': f"td:nth-child({col})", 'props': [('text-align', 'right')]}),
    
        # Set up data frame
        df = pd.DataFrame(values, columns=labels)
        df = df.style.set_table_styles(table_styles)
        return df

    def display_desires(overgenerated_desires_file):
        return DisplayTables._display_output(overgenerated_desires_file, DesireJSONLGenerator, 
                              ["id", "text"], 
                              ["Desire ID", "Desire Text"])


    def display_standard_prompts(standard_prompts_file):
        return DisplayTables._display_output(standard_prompts_file, JSONLGenerator, 
                              ["id", ["metadata", "Desire_source_id"], "text"], 
                              ["Prompt ID", "Source Desire ID", "Prompt Text"])

    
    def display_standard_prompts_with_metrics(standard_prompts_file_with_metrics):
        return DisplayTables._display_output(standard_prompts_file_with_metrics, JSONLGenerator, 
                               ["id", ["metadata", "Desire_source_id"], "text", ["metadata", "coverage"], ["metadata", "perplexity"]], 
                               ["Prompt ID", "Source Desire ID", "Prompt Text", "Coverage", "Perplexity"])


    def display_diversified_prompts(diversified_prompts_file):
        return DisplayTables._display_output(diversified_prompts_file, JSONLGenerator, 
                               ["id", ["metadata", "Prompt_source_id"], "source_prompt_text_from_dict", ["metadata", "style"], "text"], 
                               ["Diversified Prompt ID", "Source Prompt ID", "Source Prompt Text", "Style", "Diversified Prompt Text"])


    def display_diversified_prompts_with_metrics(diversified_prompts_file_with_metrics):
        return DisplayTables._display_output(diversified_prompts_file_with_metrics, JSONLGenerator, 
                               ["id", ["metadata", "Prompt_source_id"], "source_prompt_text_from_dict", ["metadata", "style"], "text", ["metadata", "coverage"], ["metadata", "faithfulness"], ["metadata", "perplexity"]], 
                               ["Diversified Prompt ID", "Source Prompt ID", "Source Prompt Text", "Style", "Diversified Prompt Text", "Coverage", "Faithfulness", "Perplexity"])

    def display_filtered_prompts(filtered_prompts_file):
        return DisplayTables._display_output(filtered_prompts_file, JSONLGenerator, 
                               ["id", "text", ["metadata", "coverage"], ["metadata", "faithfulness"], ["metadata", "perplexity"]], 
                               ["Prompt ID", "Prompt Text", "Coverage", "Faithfulness", "Perplexity"])


    def display_answers(answers_file):
        return DisplayTables._display_output(answers_file, JSONLGenerator, 
                               ["id", "prompt_id", "source_prompt_text_from_dict", "text"], 
                               ["Answer ID", "Prompt ID", "Prompt Text", "Answer Text"])

    def display_manipulated_answers(answers_file):
        return DisplayTables._display_output(answers_file, JSONLGenerator, 
                               ["id", ["metadata", "Answer_source_id"], "prompt_id", ["metadata", "change_type"], "text"], 
                               ["Modified Answer ID", "Answer ID", "Prompt ID", "Modification", "Modified Answer Text"])

    def display_hallucinations(hallucinations_file):
        return DisplayTables._display_output(hallucinations_file, JSONLGenerator, 
                               ["id", "answer_id", "snippet", "explanation", "harm_level", "confidence"], 
                               ["Hallucination ID", "Answer ID", "Answer Snippet", "Hallucination Explanation", "Harm Level", "Confidence"])

    def display_omissions(omissions_file):
        return DisplayTables._display_output(omissions_file, JSONLGenerator, 
                               ["id", "answer_id", "explanation", "harm_level", "confidence"], 
                               ["Omission ID", "Answer ID", "Omission Explanation", "Harm Level", "Confidence"])

    def display_confidences(confidences_file):
        return DisplayTables._display_output(confidences_file, JSONLGenerator, 
                               ["id", "answer_id", "explanation", "confidence"], 
                               ["Confidence ID", "Answer ID", "Confidence Explanation", "Confidence"])

    def display_summary_metrics(infile, tag=None):

        if tag is not None:
            DisplayTables._metrics_file_paths.append([tag, infile])

        if not DisplayTables._show_output:
            return DisplayTables._show_short_status()

        table_styles = DisplayTables._common_table_styles()
        table_styles.append({'selector': 'td:nth-child(3)', 'props': [('text-align', 'right')]})

        out = []
        with fopen(infile) as f:
            for line in f:
                j = json.loads(line)
                for key in j.keys():
                    out.append([key, j[key]])
        df = pd.DataFrame(out, columns=["Metric", "Value"])
        df = df.style.set_table_styles(table_styles)
        return df

    def display_metrics_file_paths():
        print(f"Notebook path:\n{DisplayTables._notebook_path}")
        for line in DisplayTables._metrics_file_paths:
            print(f"{line[0]}:\n{line[1]}")
