import os.path
import yaml

from src.muvis_align.Pipeline import Pipeline


def test(resource_files):
    for resource_file in resource_files:
        with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
            params = yaml.safe_load(file)

        pipeline = Pipeline(params)
        pipeline.run()


if __name__ == "__main__":
    resource_files = [
        'params_test_2d.yml',
        #'params_test_2d_overlay.yml',
    ]
    test(resource_files)
