import copy
import json
import logging
import os
import string
import sys
import typing

from tqdm.auto import tqdm

from bbn_medic.common.Desire import Desire
from bbn_medic.common.Disorder import Disorder
from bbn_medic.common.Patient import AtomicPatientExpression, PatientExpression
from bbn_medic.common.Prompt import Prompt
from bbn_medic.common.Style import Style
from bbn_medic.io.io_utils import StyleJSONLGenerator, JSONLGenerator, fopen
from bbn_medic.utils.text_utils import post_process_text

logger = logging.getLogger("bbn_medic.generation.PromptGenerator")


class PromptGenerator:
    @staticmethod
    def generate(input_file, output_file, model_instance, num_prompts, model_temp,
                 default_disorder,
                 styles_file=None,
                 desire_to_prompt_generation_prompts_file=None,
                 prompt_to_prompt_generation_prompts_file=None,
                 disorders_file=None,
                 use_default_prompt=False,
                 synthetic_prefixes=None,
                 preamble_text=None):

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if synthetic_prefixes is None:
            synthetic_prefixes = [""]

        # Styles
        styles = []
        if styles_file:
            for style_obj in StyleJSONLGenerator.read(styles_file):
                styles.append(style_obj)

        if len(styles) == 0:
            styles = [Style(id="dummy", text="standard")]

        # Desire to prompt generation prompts
        # TODO: Make the following two prompt loadings done by an object
        if use_default_prompt:
            desire_to_prompt_generation_prompt_objs = [
                Prompt(text=None, id="default_prompt", metadata={"default_prompt": True})]
        else:
            desire_to_prompt_generation_prompt_objs = []
        if desire_to_prompt_generation_prompts_file is not None:
            for obj in JSONLGenerator.read(desire_to_prompt_generation_prompts_file):
                desire_to_prompt_generation_prompt_objs.append(obj)

        # Prompt to prompt generation prompts
        if use_default_prompt:
            prompt_to_prompt_generation_prompt_objs = [
                Prompt(text=None, id="default_prompt", metadata={"default_prompt": True})]
        else:
            # If no default prompt, make sure there's some other input for generation or else there's a problem
            if prompt_to_prompt_generation_prompts_file is None and desire_to_prompt_generation_prompts_file is None:
                raise ValueError(
                    f"Default prompt set to false, but prompt_to_prompt_generation_prompts_file and desire_to_prompt_generation_prompts_file are both None.")
            prompt_to_prompt_generation_prompt_objs = []
        if prompt_to_prompt_generation_prompts_file is not None:
            for obj in JSONLGenerator.read(prompt_to_prompt_generation_prompts_file):
                prompt_to_prompt_generation_prompt_objs.append(obj)

        # Load disorders
        # Everything is already associated with a disorder (if used), so no need to loop over them
        disorder_info = dict()
        if disorders_file is not None:
            for obj in JSONLGenerator.read(disorders_file):
                disorder_info[obj.id] = obj
        else:
            disorder_info["default_disorder"] = Disorder(text=default_disorder, id="default_disorder")

        # Loop over desires or prompts
        with fopen(output_file, "w") as g:
            for obj in tqdm(JSONLGenerator.read_all(input_file), file=sys.stdout, desc="Generating prompts"):
                id = obj.id
                source_text = obj.text
                obj_type = obj.type
                if obj_type == "Prompt":
                    prompt_generation_prompt_objs = prompt_to_prompt_generation_prompt_objs
                elif obj_type == "Desire":
                    prompt_generation_prompt_objs = desire_to_prompt_generation_prompt_objs
                else:
                    raise ValueError(f"Unsupported type {obj_type}")

                generated_prompt_objects = []
                for prompt_generation_prompt_obj in prompt_generation_prompt_objs:
                    for style_obj in styles:
                        prompt_style = style_obj.text
                        if obj_type == "Prompt":
                            if prompt_generation_prompt_obj.text is None:
                                if prompt_generation_prompt_obj.metadata.get("default_prompt", False) is True:
                                    llm_prompt_text = model_instance.prompt_for_chatbot_prompt_to_chatbot_prompt(
                                        source_text, prompt_style)
                                else:
                                    raise ValueError(f"Unsupported given prompt")
                            else:
                                llm_prompt_text = model_instance.text_to_formatted_prompt(
                                    prompt_generation_prompt_obj.text.format(source_text=source_text,
                                                                             prompt_style=prompt_style))

                        elif obj_type == "Desire":
                            if prompt_generation_prompt_obj.text is None:
                                if prompt_generation_prompt_obj.metadata.get("default_prompt", False) is True:
                                    llm_prompt_text = model_instance.prompt_for_desire_to_chatbot_prompt(source_text,
                                                                                                         prompt_style)
                                else:
                                    raise ValueError(f"Unsupported given prompt")
                            else:
                                if not disorder_info:
                                    prompt_generation_prompt = prompt_generation_prompt_obj.text.format(
                                        source_text=source_text, prompt_style=prompt_style)
                                else:
                                    prompt_generation_prompt = prompt_generation_prompt_obj.text.format(
                                        source_text=source_text, prompt_style=prompt_style,
                                        disorder_text=disorder_info[obj.metadata["disorder_id"]].text)
                                llm_prompt_text = model_instance.text_to_formatted_prompt(prompt_generation_prompt)
                        generated_text = model_instance.forward(llm_prompt_text, temperature=model_temp,
                                                                num_return_sequences=num_prompts)
                        generated_prompts = [post_process_text(x,
                                                               characters_to_replace_with_space={'"', '\n'},
                                                               only_keep_first_question=True,
                                                               remove_leading_spaces=True, remove_trailing_spaces=True)
                                             for x in generated_text]
                        generated_prompts = [x for x in generated_prompts if x is not None]
                        for x in generated_prompts:
                            x_lower = x.lower()

                            # drop prompts that contain "rewrite"
                            # TODO change this to a more flexible, non-hardcoded filtering process
                            if "rewrite" in x_lower:
                                continue

                            if disorders_file is not None:
                                # make sure disorder is in prompt but try not to duplicate mention of it
                                if obj_type == "Prompt":
                                    disorder_id = obj.metadata["Desire_metadata"]["disorder_id"]
                                else:
                                    disorder_id = obj.metadata["disorder_id"]

                                disorder_obj = disorder_info[disorder_id]
                                present = False
                                # normaize prompt for alternate_names
                                x_norm = " ".join(x_lower.split())
                                for alt_name in disorder_obj.alternate_names:
                                    if alt_name in x_norm:
                                        present = True
                                        break
                                if not present:
                                    # We use a default if this is None (to make it easier to call from the scripts)
                                    if preamble_text is None:
                                        preamble_text = "My question is related to {disorder_text}"

                                    if preamble_text != "":
                                        preamble = preamble_text.format(disorder_text=disorder_obj.text)
                                        if preamble[-1] in string.punctuation:
                                            x = f"{preamble} {x}"
                                        else:
                                            x = f"{preamble}. {x}"
                            else:
                                disorder_text = disorder_info["default_disorder"].text
                                preamble = f"My question is related to {disorder_text}"
                                if preamble[-1] in string.punctuation:
                                    x = f"{preamble} {x}"
                                else:
                                    x = f"{preamble}. {x}"

                            for synth in synthetic_prefixes:
                                metadata = {f"{obj_type}_metadata": obj.metadata,
                                            f"{obj_type}_source_id": id,
                                            "style": style_obj,
                                            "generation_prompt_id": prompt_generation_prompt_obj.id}
                                metadata['original_prompt'] = llm_prompt_text
                                generated_prompt_objects.append(Prompt(text=f"{synth}{x}",
                                                                       metadata=metadata))
                for generated_prompt in generated_prompt_objects:
                    g.write(generated_prompt.to_json() + "\n")

    @staticmethod
    def generate_v2(input_file, output_file, desire_to_prompt_generation_prompts_file, model_instance, model_temp,
                    num_prompts, patient_expression_path):
        desire_to_prompt_generation_prompt_objs = []
        for obj in JSONLGenerator.read(desire_to_prompt_generation_prompts_file):
            desire_to_prompt_generation_prompt_objs.append(obj)
        patient_expressions = JSONLGenerator.read_all(patient_expression_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        generated_prompts_raw_counts = 0
        generated_prompts_after_post_processing_counts = 0
        generated_prompts_after_deduplication_counts = 0
        # Given two different ways of saying "I'm 24 years old," the LLM may choose to output the same question 
        # because the meaning remains the same. We want to avoid outputting multiple question strings that are 
        # identical.
        generated_patient_id_prompt_set = set()  
        input_desires = JSONLGenerator.read_all(input_file)
        estimate_total = len(input_desires) * len(desire_to_prompt_generation_prompt_objs) * len(patient_expressions)
        with fopen(output_file, "w") as g, tqdm(total=estimate_total, desc="Generating prompts") as pbar:
            for obj in input_desires:
                assert isinstance(obj, Desire)
                for desire_to_prompt_generation_prompt_obj in desire_to_prompt_generation_prompt_objs:
                    for patient_expression in patient_expressions:
                        assert isinstance(patient_expression, PatientExpression)
                        prompt = desire_to_prompt_generation_prompt_obj.text.format(
                            patient_expression=patient_expression.text,
                            desire_text=obj.text,
                        )
                        llm_prompt_text = model_instance.text_to_formatted_prompt(prompt)
                        generated_text = model_instance.forward(llm_prompt_text, temperature=model_temp,
                                                                num_return_sequences=num_prompts)
                        generated_prompts_raw_counts += len(generated_text)
                        generated_prompts = [post_process_text(x,
                                                               characters_to_replace_with_space={'"', '\n'},
                                                               only_keep_first_question=True,
                                                               remove_leading_spaces=True,
                                                               remove_trailing_spaces=True)
                                             for x in generated_text]
                        generated_prompts = [x for x in generated_prompts if x is not None]
                        generated_prompts_after_post_processing_counts += len(generated_prompts)
                        pbar.update(1)
                        for generated_prompt in generated_prompts:
                            deduplication_key = (patient_expression.patient_id, "standard", generated_prompt)
                            if deduplication_key not in generated_patient_id_prompt_set:
                                generated_patient_id_prompt_set.add(deduplication_key)
                                metadata_d = {
                                    f"{type(obj).__name__}_metadata": obj.metadata,
                                    f"{type(obj).__name__}_source_id": obj.id,

                                    "desire_generation": obj.metadata,
                                    "prompt_generation": {
                                        "source_id": obj.id,
                                        "source_field": "desire_generation",
                                        "desire_to_prompt_generation_prompt": prompt,
                                        "style_id": "standard",
                                        "patient_expression_metadata": {
                                            "id": patient_expression.id,
                                            "patient_id": patient_expression.patient_id,
                                            "meaning_id": patient_expression.meaning_id,
                                            "style_id": patient_expression.style_id,
                                            "complete_meaning_id": patient_expression.complete_meaning_id
                                        }
                                    },
                                    "last_stage_key": "prompt_generation"
                                }
                                generated_prompt_obj = Prompt(text=generated_prompt, metadata=metadata_d)
                                g.write(generated_prompt_obj.to_json() + "\n")
                                generated_prompts_after_deduplication_counts += 1
        logger.info(f"Total generated raw prompts count is:\t{generated_prompts_raw_counts}")
        logger.info(
            f"Total generated prompts count after post processing is:\t{generated_prompts_after_post_processing_counts}")
        logger.info(
            f"Total generated prompts count after deduplication is:\t{generated_prompts_after_deduplication_counts}")

    @staticmethod
    def restyle_generic(input_text_objs, styles, model_instance, model_temp, num_return_sequences,
                        characters_to_replace_with_space, only_keep_first_question,
                        only_keep_first_question_or_declarative, remove_leading_spaces, remove_trailing_spaces,
                        output_handling_callback):
        memo_to_generation = {}
        total = len(input_text_objs) * len(styles)
        with tqdm(total=total, desc="Restyling prompts") as pbar:
            for style in styles:
                assert isinstance(style, Style)
                for input_text_obj in input_text_objs:
                    assert isinstance(input_text_obj, Prompt) or isinstance(input_text_obj, AtomicPatientExpression)
                    input_text = input_text_obj.text
                    memo_key = (input_text, style.text)
                    if memo_key not in memo_to_generation:
                        if len(input_text.strip()) > 0:  # Making sure we don't send empty to the LLM, which generates garbage
                            llm_prompt_text = model_instance.prompt_for_chatbot_prompt_to_chatbot_prompt(
                                input_text,
                                style.text)
                            generated_text = model_instance.forward(llm_prompt_text, temperature=model_temp,
                                                                    num_return_sequences=num_return_sequences)
                            generated_prompts = [post_process_text(x,
                                                                   characters_to_replace_with_space=characters_to_replace_with_space,
                                                                   only_keep_first_question=only_keep_first_question,
                                                                   only_keep_first_question_or_declarative=only_keep_first_question_or_declarative,
                                                                   remove_leading_spaces=remove_leading_spaces,
                                                                   remove_trailing_spaces=remove_trailing_spaces)
                                                 for x in generated_text]
                            generated_prompts = [x for x in generated_prompts if x is not None]
                            memo_to_generation[memo_key] = generated_prompts, llm_prompt_text
                        else:
                            memo_to_generation[memo_key] = [""], ""
                    generated_prompts, llm_prompt_text = memo_to_generation[memo_key]
                    output_handling_callback(input_text_obj, llm_prompt_text, style, generated_prompts)
                    pbar.update(1)

    @staticmethod
    def restyle_patient_expression(input_file, output_file, styles_file, model_instance, model_temp, num_prompts):
        atomic_patient_expressions = JSONLGenerator.read_all(input_file)
        styles = JSONLGenerator.read_all(styles_file)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with fopen(output_file, 'w') as wfp:
            def handle_restyle_output(atomic_patient_expression: AtomicPatientExpression, llm_prompt_text: str,
                                      style: Style,
                                      output_restyled_texts: typing.List[str]):
                for idx, text in enumerate(output_restyled_texts):
                    new_atomic_expression = AtomicPatientExpression(
                        id=f"{'|'.join(atomic_patient_expression.id.split('|')[:-1])}|{style.id}^{idx}",
                        # The original id generated at Patient.py is in {meaning_id}|{style_id}^{style_idx} format. Here we preserve meaning id, and replace style_id and idx with new style id and idx, so one can use "meaning_id" as key to find same ancient patient expression.
                        meaning_id=atomic_patient_expression.meaning_id,
                        style_id=style.id, text=text)
                    new_atomic_expression.metadata['AtomicPatientExpression_source_id'] = atomic_patient_expression.id
                    wfp.write(f"{json.dumps(new_atomic_expression.model_dump(), ensure_ascii=False)}\n")

            PromptGenerator.restyle_generic(atomic_patient_expressions, styles, model_instance, model_temp, num_prompts,
                                            characters_to_replace_with_space={'"', '\n'},
                                            only_keep_first_question=False,
                                            only_keep_first_question_or_declarative=True, remove_leading_spaces=True,
                                            remove_trailing_spaces=True,
                                            output_handling_callback=handle_restyle_output)

    @staticmethod
    def restyle_prompt(input_file, output_file, styles_file, model_instance, model_temp, num_prompts):
        styles = JSONLGenerator.read_all(styles_file)
        original_prompts = JSONLGenerator.read_all(input_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with fopen(output_file, 'w') as wfp:
            def handle_restyle_output(original_prompt: Prompt, llm_prompt_text: str, style: Style,
                                      output_restyled_texts: typing.List[str]):
                for idx, generated_prompt in enumerate(output_restyled_texts):
                    new_prompt = copy.deepcopy(original_prompt)
                    metadata = new_prompt.metadata
                    metadata[f"{type(original_prompt).__name__}_metadata"] = original_prompt.metadata
                    metadata[f"{type(original_prompt).__name__}_source_id"] = original_prompt.id
                    metadata['restyle_prompt'] = {
                        "source_id": original_prompt.id,
                        "source_field": "prompt_generation",
                        "prompt_restyling_prompt": llm_prompt_text,
                        "style_id": style.id
                    }
                    metadata['last_stage_key'] = 'restyle_prompt'
                    new_prompt.text = generated_prompt
                    new_prompt.id = f"{new_prompt.id}|{style.id}^{idx}"
                    new_prompt.metadata = metadata
                    wfp.write(new_prompt.to_json() + "\n")

            PromptGenerator.restyle_generic(original_prompts, styles, model_instance, model_temp, num_prompts,
                                            characters_to_replace_with_space={'"', '\n'},
                                            only_keep_first_question=True,
                                            only_keep_first_question_or_declarative=False, remove_leading_spaces=True,
                                            remove_trailing_spaces=True,
                                            output_handling_callback=handle_restyle_output)
