import json

from bbn_medic.utils.logging_utils import setup_logger
from bbn_medic.common.Patient import Patient
from bbn_medic.io.io_utils import fopen


def main(in_patients_jsonl, lang, output_patient_expression_jsonl):
    with fopen(in_patients_jsonl) as fp, fopen(output_patient_expression_jsonl, 'w') as fout:
        for i in fp:
            j = json.loads(i)
            patient = Patient(**j)
            patient_expressions = patient.convert_to_complete_patient_descriptions(lang=lang)
            for patient_expression in patient_expressions:
                fout.write(f"{json.dumps(patient_expression.model_dump())}\n")


if __name__ == "__main__":
    setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_patients_jsonl", required=True, type=str)
    parser.add_argument("--lang", required=True, type=str)
    parser.add_argument("--out_patient_expression_jsonl", required=True, type=str)
    args = parser.parse_args()
    main(args.in_patients_jsonl, args.lang, args.out_patient_expression_jsonl)
