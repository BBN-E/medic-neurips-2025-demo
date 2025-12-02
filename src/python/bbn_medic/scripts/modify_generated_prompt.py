import copy
import os

from tqdm.auto import tqdm

from bbn_medic.common.Patient import PatientExpression
from bbn_medic.common.Prompt import Prompt
from bbn_medic.io.io_utils import JSONLGenerator
from bbn_medic.utils.file_utils import fopen

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--modifier_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--patient_expression_file", type=str, required=False)
    parser.add_argument("--use_case", type=str, required=False)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    assert args.modifier_name == "PatientExpressionModifier"

    allowed_patient_expressions = JSONLGenerator.read_all(args.patient_expression_file)
    style_to_patient_id_to_patient_expressions = dict()
    for allowed_patient_expression in allowed_patient_expressions:
        assert isinstance(allowed_patient_expression, PatientExpression)
        style = allowed_patient_expression.style_id
        patient_id = allowed_patient_expression.patient_id
        style_to_patient_id_to_patient_expressions.setdefault(style, dict()).setdefault(patient_id, []).append(
            allowed_patient_expression)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output_prompt_texts = set()
    all_prompts = JSONLGenerator.read_all(args.input_file)
    with fopen(args.output_file, "w") as g:
        for prompt_obj in tqdm(all_prompts):
            assert isinstance(prompt_obj, Prompt)
            # Case 1: Original prompt, without patient expression, but with use_case
            modified_text = f"{args.use_case} {prompt_obj.text}".strip()
            if modified_text not in output_prompt_texts:
                output_prompt_texts.add(modified_text)
                modified_prompt_obj = copy.deepcopy(prompt_obj)
                modified_prompt_obj.text = modified_text
                modified_prompt_obj.id = f"{prompt_obj.id}#case_1"
                modified_prompt_obj.metadata['attach_patient_expression'] = {
                    'attached_patient_expression_metadata': None,
                    "prompt_key": modified_prompt_obj.metadata['last_stage_key']
                }
                modified_prompt_obj.metadata['last_stage_key'] = 'attach_patient_expression'
                g.write(modified_prompt_obj.to_json() + "\n")
            # Case 2: When the question is restyled, the patient expression can come in question's style
            if prompt_obj.metadata[prompt_obj.metadata['last_stage_key']]['style_id'] != "standard":
                question_style = prompt_obj.metadata[prompt_obj.metadata['last_stage_key']]['style_id']
                original_patient_id = prompt_obj.metadata['prompt_generation']['patient_expression_metadata'][
                    'patient_id']
                local_id = 0
                for patient_expression in style_to_patient_id_to_patient_expressions.get(question_style, dict()).get(
                        original_patient_id, ()):
                    assert isinstance(patient_expression, PatientExpression)
                    modified_text = f"{patient_expression.text} {args.use_case} {prompt_obj.text}".strip()
                    if modified_text not in output_prompt_texts:
                        output_prompt_texts.add(modified_text)
                        modified_prompt_obj = copy.deepcopy(prompt_obj)
                        modified_prompt_obj.text = modified_text
                        modified_prompt_obj.id = f"{prompt_obj.id}#case_2#{local_id}"
                        modified_prompt_obj.metadata['attach_patient_expression'] = {
                            'attached_patient_expression_metadata': {
                                "id": patient_expression.id,
                                "patient_id": patient_expression.patient_id,
                                "meaning_id": patient_expression.meaning_id,
                                "style_id": patient_expression.style_id,
                                "complete_meaning_id": patient_expression.complete_meaning_id
                            },
                            "prompt_key": modified_prompt_obj.metadata['last_stage_key']
                        }
                        modified_prompt_obj.metadata['last_stage_key'] = 'attach_patient_expression'
                        g.write(modified_prompt_obj.to_json() + "\n")
                        local_id += 1
            # Case 3: No matter of how the question is styled, the patient expression can be in standard style
            original_patient_id = prompt_obj.metadata['prompt_generation']['patient_expression_metadata']['patient_id']
            local_id = 0
            for patient_expression in style_to_patient_id_to_patient_expressions.get('standard', dict()).get(
                    original_patient_id, ()):
                assert isinstance(patient_expression, PatientExpression)
                modified_text = f"{patient_expression.text} {args.use_case} {prompt_obj.text}".strip()
                if modified_text not in output_prompt_texts:
                    output_prompt_texts.add(modified_text)
                    modified_prompt_obj = copy.deepcopy(prompt_obj)
                    modified_prompt_obj.text = modified_text
                    modified_prompt_obj.id = f"{prompt_obj.id}#case_3#{local_id}"
                    modified_prompt_obj.metadata['attach_patient_expression'] = {
                        'attached_patient_expression_metadata': {
                            "id": patient_expression.id,
                            "patient_id": patient_expression.patient_id,
                            "meaning_id": patient_expression.meaning_id,
                            "style_id": patient_expression.style_id,
                            "complete_meaning_id": patient_expression.complete_meaning_id
                        },
                        "prompt_key": modified_prompt_obj.metadata['last_stage_key']
                    }
                    modified_prompt_obj.metadata['last_stage_key'] = 'attach_patient_expression'
                    g.write(modified_prompt_obj.to_json() + "\n")
                    local_id += 1
