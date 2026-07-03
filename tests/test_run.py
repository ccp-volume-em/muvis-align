import os.path
import pytest
import yaml
from multiview_stitcher import msi_utils

from src.muvis_align.MVSRegistration import MVSRegistration
from src.muvis_align.Pipeline import Pipeline


test_filenames = [
    'params_test_2d.yml',
    'params_test_2d2.yml',
    'params_test_2d_overlay.yml',
]

@pytest.mark.parametrize(
    'resource_file', test_filenames,
)
def test(resource_file):
    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    pipeline = Pipeline(params)
    pipeline.run()


@pytest.mark.parametrize(
    "resource_file", test_filenames
)
def test2(resource_file):
    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    operation_params = params['operations'][0]
    reg_params = operation_params['registration']
    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_sims()
    reg_sims, reg_indices, _ = reg.preprocess(reg.sims)
    #reg.register_full(reg.sims, reg_sims, register_indices=reg_indices, register_params=reg_params)
    reg.register_pairs(reg.sims, reg_sims, params=reg_params)
    msims = [msi_utils.get_msim_from_sim(sim) for sim in reg.sims]
    reg.register_global(reg.sims, msims, params=reg_params)
    reg.fuse(reg.sims, output_filename='output')


if __name__ == "__main__":
    for filename in test_filenames:
        print()
        print()
        print('TESTING:', filename)
        print()
        test(filename)
