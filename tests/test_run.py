import os.path
import pytest
import yaml
from multiview_stitcher import msi_utils

from muvis_align.MVSRegistration import MVSRegistration
from muvis_align.Pipeline import Pipeline


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
    'resource_file', test_filenames,
)
def test_source_msims_match_sims(resource_file):
    from multiview_stitcher import spatial_image_utils as si_utils

    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    operation_params = params['operations'][0]
    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    sims = reg.init_data()

    assert len(reg.source_msims) == len(sims)
    for sim, msim in zip(sims, reg.source_msims):
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        assert len(scale_keys) >= 1
        sim0 = msi_utils.get_sim_from_msim(msim, scale=scale_keys[0])

        # the msim's highest-resolution scale must carry exactly the same geometry as the sim
        # that init_data() produces for the (possibly rescaled) working resolution
        assert sim0.dims == sim.dims
        assert sim0.shape == sim.shape
        assert si_utils.get_origin_from_sim(sim0) == si_utils.get_origin_from_sim(sim)
        assert si_utils.get_spacing_from_sim(sim0) == si_utils.get_spacing_from_sim(sim)

        affine_sim = si_utils.get_affine_from_sim(sim, reg.source_transform_key)
        affine_msim = si_utils.get_affine_from_sim(sim0, reg.source_transform_key)
        assert (affine_sim.values == affine_msim.values).all()

        # lower pyramid levels shrink and their spacing grows accordingly
        prev_sim = sim0
        for scale_key in scale_keys[1:]:
            level_sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
            for dim in si_utils.get_spatial_dims_from_sim(level_sim):
                assert level_sim.sizes[dim] <= prev_sim.sizes[dim]
            prev_sim = level_sim


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
    reg.init_data()
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
