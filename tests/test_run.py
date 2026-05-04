import os.path
import pytest
import yaml

from src.muvis_align.Pipeline import Pipeline


@pytest.mark.parametrize(
    'resource_file',
    [
        'params_test_2d.yml',
        'params_test_2d_overlay.yml',
    ],
)
def test(resource_file):
    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    pipeline = Pipeline(params)
    pipeline.run()


if __name__ == "__main__":
    test('params_test_2d.yml')
