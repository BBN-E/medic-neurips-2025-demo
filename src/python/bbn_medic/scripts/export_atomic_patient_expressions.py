import json
import os

from bbn_medic.io.io_utils import JSONLGenerator, fopen

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    existed_atomic_patient_expression = set()
    with fopen(args.output_file, "w") as output_file:
        for patient_expression in JSONLGenerator.read(args.input_file):
            for atomic_patient_expression in patient_expression.atomic_patient_expressions:
                output_str = json.dumps(atomic_patient_expression.model_dump(), ensure_ascii=False, sort_keys=True)
                if output_str not in existed_atomic_patient_expression:
                    existed_atomic_patient_expression.add(output_str)
                    output_file.write(f"{output_str}\n")
