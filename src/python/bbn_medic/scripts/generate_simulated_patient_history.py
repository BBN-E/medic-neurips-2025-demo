import itertools
import json

from bbn_medic.common.Patient import Patient, logger
from bbn_medic.io.io_utils import JSONLGenerator
from bbn_medic.utils.file_utils import fopen
from bbn_medic.utils.logging_utils import setup_logger
from bbn_medic.utils.text_utils import get_id_from_text


def main(disorders_file_path, symptoms_file_path, additional_generic_patient_expression_path,
         output_simulated_patient_jsonl_path):
    # TODO: Consider medication from /nfs/alletra/projects/care/data/Final_Mental_Health_Disorders_with_Casual_Descriptions.csv
    disorders = JSONLGenerator.read_all(disorders_file_path)
    disorder_id_to_disorder = {i.id: i for i in disorders}
    symptoms = JSONLGenerator.read_all(symptoms_file_path)
    symptom_id_to_symptom = {i.id: i for i in symptoms}
    with fopen(additional_generic_patient_expression_path) as in_patient_attribute_combo_file:
        additional_generic_patient_expression = json.load(in_patient_attribute_combo_file)
    if len(additional_generic_patient_expression.keys()) < 1:
        logger.warning(
            "There's no permutable in the additional patient expression config. To make sure the program can run end to end, we'll provide an empty dict for continuing.")
        additional_generic_patient_expression["EMPTY_DICT_PLACEHOLDER"] = ["DUMMY_VALUE"]
    fixed_combos = []
    for disorder_id, disorder in disorder_id_to_disorder.items():
        relevant_symptoms_patient_format = []
        for symptom_id in disorder.relevant_symptom_ids:
            symptom = symptom_id_to_symptom[symptom_id]
            relevant_symptoms_patient_format.append({
                "description": [symptom.text]
            })
        fixed_combos.append({
            "medical_histories": {
                "category": "medical_history",
                "items": [disorder.text]
            },
            "unstructured_symptoms": {"unstructured_symptoms": relevant_symptoms_patient_format}
        })
    if len(fixed_combos) < 1:
        logger.warning(
            "There's no fix combo provided. To make sure the program can run end to end, we'll provide an empty dict for continuing.")
        fixed_combos.append({})
    field_names = sorted(additional_generic_patient_expression.keys())
    buckets = [additional_generic_patient_expression[i] for i in field_names]
    new_patients = list()
    expected_num_patients = 1
    for i in buckets:
        expected_num_patients *= len(i)
    expected_num_patients *= len(fixed_combos)
    logger.info(f"We're expecting {expected_num_patients} patients.")
    actual_num_patients = 0
    for combo, fixed_combo in itertools.product(itertools.product(*buckets), fixed_combos):
        k_d = {i: j for i, j in zip(field_names, combo)}
        if "EMPTY_DICT_PLACEHOLDER" in k_d:
            k_d.pop("EMPTY_DICT_PLACEHOLDER")
        k_d.update(fixed_combo)
        patient_id = get_id_from_text(json.dumps(k_d, sort_keys=True, ensure_ascii=False))
        k_d['patient_id'] = patient_id
        new_patients.append(Patient(**k_d))
    with fopen(output_simulated_patient_jsonl_path, 'w') as wfp:
        for patient in new_patients:
            wfp.write(f"{json.dumps(patient.model_dump(), ensure_ascii=False)}\n")
            actual_num_patients += 1
    logger.info(f"We're generating {expected_num_patients} patients.")


if __name__ == "__main__":
    setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--disorders_file_path", type=str, required=True)
    parser.add_argument("--symptoms_file_path", type=str, required=True)
    parser.add_argument("--additional_generic_patient_expression_path", type=str, required=True)
    parser.add_argument("--output_simulated_patient_jsonl_path", type=str, required=True)
    args = parser.parse_args()
    main(args.disorders_file_path, args.symptoms_file_path, args.additional_generic_patient_expression_path,
         args.output_simulated_patient_jsonl_path)
