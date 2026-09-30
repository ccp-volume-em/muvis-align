import glob
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from multiview_stitcher import msi_utils, param_utils, registration
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.split_registration import register_groups, split_groups, within_group_pairs


def test_groups_are_z_planes_or_for_channel_registration_channels():
    positions = [{'z': 0.1, 'y': 0, 'x': 0}, {'z': 0.0, 'y': 0, 'x': 5}, {'z': 0.1, 'y': 5, 'x': 0}]
    sources = [SimpleNamespace(get_channels=lambda label=label: [{'label': label}]) for label in ('b', 'a', 'a')]

    assert split_groups(positions, sources) == [1, 0, 1]
    assert split_groups(positions, sources, 'c') == [1, 0, 0]
    assert within_group_pairs([(0, 1), (0, 2), (1, 2)], [1, 0, 1]) == [(0, 2)]


def textured_msim(image, origin_x):
    sim = si_utils.get_sim_from_array(image, dims=['y', 'x'], scale={'y': 1.0, 'x': 1.0},
                                      translation={'y': 0.0, 'x': origin_x}, transform_key='source')
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    return msi_utils.multiscale_sel_coords(msim, {'c': sim.coords['c'].values[0]})


@pytest.mark.parametrize('shift', [0.0, 7.0])
def test_a_misplaced_plane_is_registered_back_onto_the_one_before(shift):
    """Two planes of the same content, the second placed `shift` too far in x: stage 2 must correct it."""
    rng = np.random.default_rng(0)
    # sharp-edged blobs coarser than the plane's smoothing (tile / 25), as cells: smooth noise smoothed again
    # leaves a correlation peak so broad the shift came out half a pixel short; small, so registered unbinned
    blobs = gaussian_filter(rng.random((200, 200)), 6)
    image = ((blobs > np.median(blobs)) * 500 + 100).astype(np.float32)
    msims = [textured_msim(image, 0.0), textured_msim(image, shift)]
    identity = param_utils.affine_to_xaffine(np.eye(3), t_coords=[0])

    transforms, _ = register_groups(msims, [identity, identity], [0, 1], 'source',
                                 registration.phase_correlation_registration,
                                 resolution_kwargs={'transform': 'translation'}, binning=1)

    translations = [np.asarray(transform).squeeze()[:2, 2] for transform in transforms]
    assert np.allclose(translations[0], 0, atol=0.2)
    assert np.allclose(translations[1], [0, -shift], atol=0.2)
    # the stage-1 transforms are only borrowed to fuse the planes
    assert 'split_stage1' not in msims[0]['scale0'].ds.data_vars


def test_planes_sharing_a_tile_pattern_and_outline_are_registered_by_their_content():
    """As serial sections: both planes show the same fixed grid and outline at the same place, their content shifted
    20 px inside them - left in, the pattern and outline would match at zero shift."""
    rng = np.random.default_rng(2)
    blobs = gaussian_filter(rng.random((200, 260)), 6)
    content = (blobs > np.median(blobs)) * 500.0 + 300
    pattern = np.ones((160, 200))
    pattern[::12, :] = pattern[:, ::12] = 3.0
    pattern[:, :30] = 0     # a part of the outline without tiles
    planes = [content[20:180, 30:230] * pattern, content[20:180, 50:250] * pattern]
    msims = [textured_msim(plane.astype(np.float32), 0.0) for plane in planes]
    identity = param_utils.affine_to_xaffine(np.eye(3), t_coords=[0])

    transforms, _ = register_groups(msims, [identity, identity], [0, 1], 'source',
                                 registration.phase_correlation_registration,
                                 resolution_kwargs={'transform': 'translation'}, binning=1)

    translations = [np.asarray(transform).squeeze()[:2, 2] for transform in transforms]
    assert np.allclose(translations[1] - translations[0], [0, 20.0], atol=1.0)


def test_split_pairing_registers_no_pairs_across_planes():
    from muvis_align.MVSRegistration import MVSRegistration

    reg = MVSRegistration()
    reg.init(operation='register', input_path=sorted(glob.glob('data/S*/*.ome.zarr')),
             output_path='../../output/test_split_pairing/')
    reg.init_data()
    reg.preprocess(reg.msims)
    params = {'method': 'phase_correlation', 'pairing': 'split', 'transform_type': 'translation', 'metrics': [],
              'n_parallel_pairwise_regs': 1}

    reg.register_pairs(reg.register_msims, params=params)
    groups = reg.split_groups()
    assert reg.pairs_graph.number_of_edges() > 0
    assert all(groups[first] == groups[second] for first, second in reg.pairs_graph.edges)

    results = reg.register_global(reg.pair_msims, params=params)
    assert len(results['mappings']) == len(reg.pair_msims)


def test_group_pairs_are_registered_binned_by_multiview_stitcher(monkeypatch):
    import muvis_align.split_registration as split_registration
    seen = {}

    def compute(msims, graph, **kwargs):
        seen.update(kwargs)
        return graph
    monkeypatch.setattr(split_registration, 'compute_pairwise_registrations', compute)
    image = np.ones((64, 64), dtype=np.float32)
    msims = [textured_msim(image, 0.0), textured_msim(image, 0.0)]
    for msim in msims:
        msi_utils.set_affine_transform(msim, param_utils.affine_to_xaffine(np.eye(3), t_coords=[0]),
                                       transform_key=split_registration.GROUP_KEY, base_transform_key='source')

    split_registration.register_group_pair(msims[0], msims[1], 0, None, binning=4)

    assert seen['registration_binning'] == {'y': 4, 'x': 4}


def test_a_group_is_labelled_by_its_sources_common_prefix_up_to_a_separator():
    from muvis_align.split_registration import group_label

    assert group_label(['S000_000_000', 'S000_007_007', 'S000_003_001']) == 'S000'
    assert group_label(['S000_000_000']) == 'S000_000_000'
    assert group_label(['tile1', 'tile2']) == ''


def test_split_group_pairs_are_saved_with_the_tile_pairs_and_restored_apart_from_them(tmp_path):
    import json
    from muvis_align.MVSRegistration import MVSRegistration

    def open_registration():
        reg = MVSRegistration()
        reg.init(operation='register', input_path=sorted(glob.glob('data/S*/*.ome.zarr')),
                 output_path=tmp_path.as_posix() + '/')
        reg.init_data()
        return reg

    reg = open_registration()
    reg.preprocess(reg.msims)
    params = {'method': 'phase_correlation', 'pairing': 'split', 'transform_type': 'translation', 'metrics': [],
              'n_parallel_pairwise_regs': 1}
    reg.register(reg.register_msims, params=params)

    saved = json.load(open(reg.output + 'pair_mappings.json'))
    group_entries = {key: value for key, value in saved.items() if value.get('kind') == 'split_group'}
    assert list(group_entries) == [json.dumps(['S000', 'S001'])]
    assert {'mapping', 'quality'} <= set(group_entries[json.dumps(['S000', 'S001'])])
    assert len(saved) - len(group_entries) == reg.pairs_graph.number_of_edges()

    resumed = open_registration()
    resumed.init_progress('registered', 'ome.zarr')
    # the labels match files under S000/ and S001/ - they must not come back as tile pairs
    assert resumed.pairs_graph.number_of_edges() == reg.pairs_graph.number_of_edges()
    assert list(resumed.group_pairs) == [('S000', 'S001')]
    assert ('S000', 'S001') in resumed.metrics['group_pairs']
