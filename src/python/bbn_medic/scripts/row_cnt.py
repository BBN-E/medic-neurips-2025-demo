import os

from bbn_medic.io.io_utils import JSONLGenerator, fopen


def main(input_file, output_file):
    cnt = 0
    for idx, row in enumerate(JSONLGenerator.read(input_file)):
        cnt += 1
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with fopen(output_file, 'w') as out_file:
        out_file.write(f'{cnt}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
