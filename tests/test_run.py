import os.path

import numpy as np
import pytest
import yaml
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.MVSRegistration import MVSRegistration, RegState
from muvis_align.Pipeline import Pipeline
from muvis_align.image.util import get_msim_transform_keys, wrap_sims_as_msims
from muvis_align.util import operation_to_past_participle
from tests.data_builders import registration_from_resource

test_filenames = [
    'params_test_2d.yml',
    'params_test_2d2.yml',
    'params_test_2d_overlay.yml',
]


@pytest.mark.parametrize('resource_file', test_filenames)
def test(resource_file, tmp_path):
    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)
    params['operations'][0]['output']['path'] = tmp_path.as_posix() + '/'

    pipeline = Pipeline(params)
    pipeline.run()


def registered_from_resource(output_path):
    """params_test_2d registered by phase correlation: the tests using it check plumbing, not the method."""
    reg, operation_params = registration_from_resource('params_test_2d.yml', output_path)
    reg_params = dict(operation_params['registration'], method='phase_correlation')
    reg.preprocess(reg.msims)
    return reg, operation_params, reg_params


@pytest.mark.parametrize('preprocess_kwargs', [
    {},
    {'gaussian_sigma': 3, 'normalisation': 'global'},
    {'normalisation': 'individual'},
], ids=['none', 'gaussian+global-norm', 'individual-norm'])
def test_register_msims_keep_every_shrinking_pyramid_level(preprocess_kwargs, tmp_path):
    # preprocess() keeps msims-in/msims-out per pyramid level for the steps that generalise per level
    reg, _ = registration_from_resource('params_test_2d.yml', tmp_path)

    for msim in reg.msims:
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        assert scale_keys[0] == 'scale0'
        level_sims = [msi_utils.get_sim_from_msim(msim, scale=scale_key) for scale_key in scale_keys]
        for finer, coarser in zip(level_sims, level_sims[1:]):
            assert all(coarser.sizes[dim] <= finer.sizes[dim] for dim in si_utils.get_spatial_dims_from_sim(coarser))

    reg.preprocess(reg.msims, **preprocess_kwargs)

    assert len(reg.register_msims) == len(reg.msims)
    for msim in reg.register_msims:
        assert len(msi_utils.get_sorted_scale_keys(msim)) == len(reg.msims[0].children)


def test_register_global_writes_every_level_and_fuses_as_a_trivial_wrap_does(tmp_path):
    """register_global puts the transform on self.msims at every scale, not only on the msims it is given; and fusing
    the real pyramid gives what fusing a trivial single-level wrap of the same registered sims does."""
    reg, _, reg_params = registered_from_resource(tmp_path)
    reg.register_pairs(reg.register_msims, params=reg_params)
    pair_msims = wrap_sims_as_msims([msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in reg.msims])
    reg.register_global(pair_msims, params=reg_params)

    assert len(reg.msims) == len(pair_msims)
    for pair_msim, msim in zip(pair_msims, reg.msims):
        affine_pair = si_utils.get_affine_from_sim(msi_utils.get_sim_from_msim(pair_msim, scale='scale0'),
                                                   reg.reg_transform_key)
        for scale_key in msi_utils.get_sorted_scale_keys(msim):
            level_sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
            assert (affine_pair.values == si_utils.get_affine_from_sim(level_sim, reg.reg_transform_key).values).all()

    # taken before fusing, which promotes reg.msims' transforms to 3D in place
    trivial_msims = wrap_sims_as_msims([msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in reg.msims])
    fused_trivial, _ = reg.fuse(trivial_msims, output_filename='fused_trivial')
    fused_pyramid, _ = reg.fuse(reg.msims, output_filename='fused_pyramid')

    trivial = np.asarray(msi_utils.get_sim_from_msim(fused_trivial, scale='scale0').data)
    pyramid = np.asarray(msi_utils.get_sim_from_msim(fused_pyramid, scale='scale0').data)
    np.testing.assert_array_equal(trivial, pyramid)


def test_fuse_channel_overlay_real_pyramid_matches_trivial_wrap():
    # several extra_metadata channels take fuse()'s own combine-as-channels path: it too must not
    # depend on fusing the real pyramid or a trivial single-level wrap
    with open(os.path.join('resources', 'params_test_2d.yml'), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)
    operation_params = params['operations'][0]
    operation_params['input']['extra_metadata'] = {
        'channels': [{'label': f'ch{index}'} for index in range(4)]
    }

    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_data()

    sims = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in reg.msims]
    fused_trivial, _ = reg.fuse(wrap_sims_as_msims(sims), transform_key=reg.source_transform_key)
    fused_pyramid, _ = reg.fuse(reg.msims, transform_key=reg.source_transform_key)

    sim_trivial = msi_utils.get_sim_from_msim(fused_trivial, scale='scale0')
    sim_pyramid = msi_utils.get_sim_from_msim(fused_pyramid, scale='scale0')
    assert sim_trivial.dims == sim_pyramid.dims
    assert sim_trivial.shape == sim_pyramid.shape
    np.testing.assert_array_equal(np.asarray(sim_trivial.data), np.asarray(sim_pyramid.data))


def test_preprocess_scale_selects_real_subpyramid_not_a_resize(tmp_path):
    # `scale` selects every native level at or coarser than it as a genuine sub-pyramid,
    # not a resize to one exact resolution
    reg, _ = registration_from_resource('params_test_2d.yml', tmp_path)
    full_scale_keys = msi_utils.get_sorted_scale_keys(reg.msims[0])

    reg.preprocess(reg.msims, scale=2)

    assert reg.register_msims is not None
    sub_scale_keys = msi_utils.get_sorted_scale_keys(reg.register_msims[0])
    assert 1 <= len(sub_scale_keys) < len(full_scale_keys)

    # the sub-pyramid's scale0 must be byte-identical to the matching native level (not a resize)
    sub_sim0 = msi_utils.get_sim_from_msim(reg.register_msims[0], scale='scale0')
    matching_native_level = full_scale_keys[len(full_scale_keys) - len(sub_scale_keys)]
    native_sim = msi_utils.get_sim_from_msim(reg.msims[0], scale=matching_native_level)
    assert sub_sim0.shape == native_sim.shape
    np.testing.assert_array_equal(np.asarray(sub_sim0.data), np.asarray(native_sim.data))


def test_init_progress_resumes_a_registration_of_the_same_files_only(tmp_path):
    """Reopening a registered project puts the saved transform on every msim (Interface's init_progress relies on it);
    a mappings.json left by another fileset in the same output directory is ignored, not a crash."""
    reg, operation_params, reg_params = registered_from_resource(tmp_path)
    reg.register(reg.register_msims, reg.register_indices, params=reg_params)
    output_filename = operation_to_past_participle(operation_params['operation'])

    resumed, _ = registration_from_resource('params_test_2d.yml', tmp_path)
    resumed.init_progress(output_filename, 'ome.zarr')

    assert resumed.is_global_registered()
    assert len(resumed.msims) == len(reg.msims)
    for msim, orig_msim in zip(resumed.msims, reg.msims):
        assert reg.reg_transform_key in get_msim_transform_keys(msim)
        resumed_transform = msi_utils.get_transform_from_msim(msim, reg.reg_transform_key)
        orig_transform = msi_utils.get_transform_from_msim(orig_msim, reg.reg_transform_key)
        assert (resumed_transform.values == orig_transform.values).all()

    other_fileset, _ = registration_from_resource('params_test_2d.yml', tmp_path)
    other_fileset.filenames[0] = 'data/S000/not_in_the_saved_mapping.ome.zarr'
    other_fileset.init_progress(output_filename, 'ome.zarr')

    assert other_fileset.state is RegState.INIT
