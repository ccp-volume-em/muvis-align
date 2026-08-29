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
def test_msims_match_sims(resource_file):
    from multiview_stitcher import spatial_image_utils as si_utils

    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    operation_params = params['operations'][0]
    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    msims = reg.init_data()  # msims is init_data()'s primary return value
    sims = reg.sims

    assert len(msims) == len(sims)
    for sim, msim in zip(sims, msims):
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
    reg.preprocess(reg.msims)
    reg_msims, reg_indices = reg.register_msims, reg.register_indices
    reg.register_pairs(reg_msims, params=reg_params)
    msims = [msi_utils.get_msim_from_sim(sim) for sim in reg.sims]
    reg.register_global(msims, params=reg_params)

    # register_global must also propagate the registered transform onto self.msims,
    # msim -> msim (every scale), not just onto the pair_msims (`msims`) it was called with -
    # check this before fuse(), which (via make_msims_3d, for datasets with multiple z positions)
    # mutates reg.msims' transforms in place to promote them to 3D for the output stack, a
    # pre-existing side effect unrelated to this
    from multiview_stitcher import spatial_image_utils as si_utils
    assert len(reg.msims) == len(msims)
    for pair_msim, msim in zip(msims, reg.msims):
        affine_pair = si_utils.get_affine_from_sim(
            msi_utils.get_sim_from_msim(pair_msim, scale='scale0'), reg.reg_transform_key)
        for scale_key in msi_utils.get_sorted_scale_keys(msim):
            level_sim = msi_utils.get_sim_from_msim(msim, scale=scale_key)
            affine_msim = si_utils.get_affine_from_sim(level_sim, reg.reg_transform_key)
            assert (affine_pair.values == affine_msim.values).all()

    reg.fuse(reg.msims, output_filename='output')


@pytest.mark.parametrize(
    "resource_file", test_filenames
)
def test_fuse_with_real_pyramid_matches_trivial_wrap(resource_file):
    # fuse() only ever takes msims - verify that fusing from each source's real, full multiscale
    # pyramid (self.msims) produces byte-identical scale0 output to fusing a trivial single-level
    # wrap of the same working-resolution sims (util.wrap_sims_as_msims, the escape hatch used by
    # callers with no real pyramid available, e.g. an ad-hoc preview resolution) - for a single
    # registration run (registration itself can be non-deterministic across separate runs, e.g.
    # RANSAC-based methods, so both fuse() calls here reuse the same already-registered reg.msims)
    import numpy as np
    from muvis_align.image.source_helper import create_image_source
    from muvis_align.image.util import wrap_sims_as_msims
    import shutil

    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    operation_params = params['operations'][0]
    reg_params = operation_params['registration']
    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_data()
    reg.preprocess(reg.msims, **operation_params.get('preprocess', {}))
    reg.register(reg.register_msims, reg.register_indices, params=operation_params)

    # reg.sims is never updated with the registered transform (only reg.msims is, by
    # register_global) - extract the registered scale0 sims from reg.msims instead
    from multiview_stitcher import msi_utils
    registered_sims = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in reg.msims]
    trivial_msims = wrap_sims_as_msims(registered_sims)
    filename_trivial, _ = 'test_fuse_trivial', reg.fuse(trivial_msims, output_filename='test_fuse_trivial')[1]
    filename_pyramid, _ = 'test_fuse_pyramid', reg.fuse(reg.msims, output_filename='test_fuse_pyramid')[1]

    path_trivial = reg.output + filename_trivial + '.ome.zarr'
    path_pyramid = reg.output + filename_pyramid + '.ome.zarr'
    try:
        a = create_image_source(path_trivial).get_level_data(0).compute()
        b = create_image_source(path_pyramid).get_level_data(0).compute()
        assert a.shape == b.shape
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    finally:
        shutil.rmtree(path_trivial, ignore_errors=True)
        shutil.rmtree(path_pyramid, ignore_errors=True)


def test_fuse_channel_overlay_real_pyramid_matches_trivial_wrap():
    # the is_channel_overlay path (fuse()'s per-source-as-channel branch, triggered by
    # extra_metadata['channels'] having more than one entry) has its own separate combine-as-
    # channels code path - verify it too produces byte-identical output whether fusing from the
    # real pyramid or a trivial single-level wrap
    import numpy as np
    from multiview_stitcher import msi_utils
    from muvis_align.image.util import wrap_sims_as_msims

    with open(os.path.join('resources', 'params_test_2d.yml'), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)
    operation_params = params['operations'][0]
    operation_params['input']['extra_metadata'] = {
        'channels': [{'label': f'ch{i}'} for i in range(4)]
    }

    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_data()

    fused_trivial, _ = reg.fuse(wrap_sims_as_msims(reg.sims), transform_key=reg.source_transform_key)
    fused_pyramid, _ = reg.fuse(reg.msims, transform_key=reg.source_transform_key)

    sim_trivial = msi_utils.get_sim_from_msim(fused_trivial, scale='scale0')
    sim_pyramid = msi_utils.get_sim_from_msim(fused_pyramid, scale='scale0')
    assert sim_trivial.dims == sim_pyramid.dims
    assert sim_trivial.shape == sim_pyramid.shape
    a = np.asarray(sim_trivial.data.compute() if hasattr(sim_trivial.data, 'compute') else sim_trivial.data)
    b = np.asarray(sim_pyramid.data.compute() if hasattr(sim_pyramid.data, 'compute') else sim_pyramid.data)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "resource_file", test_filenames
)
@pytest.mark.parametrize(
    "preprocess_kwargs", [
        {},
        {'gaussian_sigma': 3, 'normalisation': 'global'},
        {'normalisation': 'individual'},
    ],
    ids=['none', 'gaussian+global-norm', 'individual-norm'],
)
def test_register_msims_multilevel_matches_register_sims(resource_file, preprocess_kwargs):
    # preprocess() tracks a real multiscale counterpart (self.register_msims) alongside the
    # single-resolution self.register_sims, for the preprocessing steps that generalise cleanly
    # per pyramid level (plain passthrough, gaussian, normalisation) - verify its scale0 matches
    # register_sims exactly, and that every native pyramid level survived
    import numpy as np

    with open(os.path.join('resources', resource_file), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)

    operation_params = params['operations'][0]
    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_data()
    reg.preprocess(reg.msims, **preprocess_kwargs)
    reg_sims = reg.register_sims

    assert reg.register_msims is not None
    assert len(reg.register_msims) == len(reg_sims)
    for sim, msim in zip(reg_sims, reg.register_msims):
        scale_keys = msi_utils.get_sorted_scale_keys(msim)
        assert len(scale_keys) == len(reg.msims[0].children)
        sim0 = msi_utils.get_sim_from_msim(msim, scale=scale_keys[0])
        a = np.asarray(sim.data.compute() if hasattr(sim.data, 'compute') else sim.data)
        b = np.asarray(sim0.data.compute() if hasattr(sim0.data, 'compute') else sim0.data)
        np.testing.assert_array_equal(a, b)


def test_preprocess_scale_selects_real_subpyramid_not_a_resize():
    # preprocess()'s `scale` override should select every native level at or coarser than the
    # requested scale as a genuine (smaller) sub-pyramid - not resize to one exact resolution
    # (unnecessary for registration) and not disable msims/auto-resolution-selection entirely
    import numpy as np

    with open(os.path.join('resources', 'params_test_2d.yml'), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)
    operation_params = params['operations'][0]

    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_data()
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
    a = np.asarray(sub_sim0.data.compute() if hasattr(sub_sim0.data, 'compute') else sub_sim0.data)
    b = np.asarray(native_sim.data.compute() if hasattr(native_sim.data, 'compute') else native_sim.data)
    np.testing.assert_array_equal(a, b)

    # register_sims (the single working-resolution sim) must match the sub-pyramid's own scale0
    a2 = np.asarray(reg.register_sims[0].data.compute() if hasattr(reg.register_sims[0].data, 'compute')
                    else reg.register_sims[0].data)
    np.testing.assert_array_equal(a2, a)


def test_init_progress_resume_writes_registered_transform_onto_msims():
    # a fresh MVSRegistration resuming from a previously globally-registered run's saved
    # mappings.json must end up with reg_transform_key set on self.msims (every scale), not just
    # self.sims - Interface.py's init_progress() (copy_transforms(self.reg.msims, ...)) depends on
    # this when reopening a project that was already registered
    from muvis_align.image.util import get_msim_transform_keys
    from muvis_align.util import operation_to_past_participle
    from muvis_align.constants import zarr_extension

    with open(os.path.join('resources', 'params_test_2d.yml'), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)
    operation_params = params['operations'][0]
    reg_params = operation_params['registration']

    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_data()
    reg.preprocess(reg.msims)
    reg.register(reg.register_msims, reg.register_indices, params=reg_params)
    # register() saves pair_mappings.json/mappings.json/metrics.json to disk as a side effect

    resumed = MVSRegistration()
    resumed.init_params(params['general'], operation_params)
    resumed.init_data()
    output_filename = operation_to_past_participle(operation_params['operation'])
    output_format = params['general'].get('output', {}).get('format', zarr_extension)
    resumed.init_progress(output_filename, output_format)

    assert resumed.is_global_registered()
    assert len(resumed.msims) == len(reg.msims)
    for msim, orig_msim in zip(resumed.msims, reg.msims):
        assert reg.reg_transform_key in get_msim_transform_keys(msim)
        resumed_transform = msi_utils.get_transform_from_msim(msim, reg.reg_transform_key)
        orig_transform = msi_utils.get_transform_from_msim(orig_msim, reg.reg_transform_key)
        assert (resumed_transform.values == orig_transform.values).all()


if __name__ == "__main__":
    for filename in test_filenames:
        print()
        print()
        print('TESTING:', filename)
        print()
        test(filename)
