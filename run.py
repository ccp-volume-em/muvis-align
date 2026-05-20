import argparse
import yaml

from src.muvis_align.Pipeline import Pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f'muvis-align')
    parser.add_argument('params',
                        help='The parameters file',
                        default='resources/params.yml')

    args = parser.parse_args()
    print(f'Parameters file: {args.params}')
    with open(args.params, 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    pipeline = Pipeline(params)
    pipeline.run()
