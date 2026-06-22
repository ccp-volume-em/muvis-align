import os.path
import pytest
import yaml
from multiview_stitcher import msi_utils

from src.muvis_align.MVSRegistration import MVSRegistration
from src.muvis_align.Pipeline import Pipeline


@pytest.mark.parametrize(
    'resource_file',
    [
        'params_test_2d.yml',
        'params_test_2d2.yml',
        'params_test_2d_overlay.yml',
    ],
)
def test(resource_file):
    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    pipeline = Pipeline(params)
    pipeline.run()


@pytest.mark.parametrize(
    'resource_file',
    [
        'params_test_2d.yml',
        'params_test_2d2.yml',
        'params_test_2d_overlay.yml',
    ],
)
def test2(resource_file):
    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    operation_params = params['operations'][0]
    reg_params = operation_params['registration']
    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_sims()
    reg.preprocess(reg.sims)
    reg.register_pairs(reg.sims, params=reg_params)
    msims = [msi_utils.get_msim_from_sim(sim) for sim in reg.sims]
    reg.register_global(reg.sims, msims, params=reg_params)
    reg.fuse(reg.sims, output_filename='output')


if __name__ == "__main__":
    for filename in [
        'params_test_2d.yml',
        'params_test_2d2.yml',
        'params_test_2d_overlay.yml',
    ]:
        test2(filename)
