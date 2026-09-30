import glob
from types import SimpleNamespace

import numpy as np
import xarray as xr
from multiview_stitcher import msi_utils, param_utils
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


def translation_xparam(shift_yx):
    matrix = np.eye(3)
    matrix[:2, 2] = shift_yx
    return param_utils.affine_to_xaffine(matrix, t_coords=[0])


def test_each_group_is_corrected_by_its_group_pair_registration_on_top_of_stage1(monkeypatch):
    """The group pair registration stubbed with a known result - the second plane's content 7 further in x - so what
    is checked is the plumbing: the correction resolved per group and composed after each tile's stage-1 transform."""
    import muvis_align.split_registration as split_registration

    def registered(msims, graph, **kwargs):
        for edge in graph.edges:
            graph.edges[edge]['transform'] = translation_xparam([0, 7.0])
            graph.edges[edge]['quality'] = translation_xparam([0, 0]).isel(x_in=0, x_out=0).copy(data=[1.0])
            graph.edges[edge]['bbox'] = xr.DataArray([[[0.0, 0.0], [40.0, 40.0]]], dims=['t', 'point_index', 'dim'],
                                                     coords={'t': [0]})
        return graph
    monkeypatch.setattr(split_registration, 'compute_pairwise_registrations', registered)
    image = np.ones((40, 40), dtype=np.float32)
    msims = [textured_msim(image, 0.0), textured_msim(image, 0.0), textured_msim(image, 0.0)]
    stage1 = [translation_xparam(shift) for shift in ([0, 0], [0, 0], [3.0, 2.0])]

    transforms, graph = register_groups(msims, stage1, [0, 1, 1], 'source', None,
                                        resolution_kwargs={'transform': 'translation'}, binning=1)

    translations = np.array([np.asarray(transform).squeeze()[:2, 2] for transform in transforms])
    assert np.allclose(translations, [[0, 0], [0, -7.0], [3.0, -5.0]], atol=1e-9)
    assert list(graph.edges) == [(0, 1)]
    # the stage-1 transforms are only borrowed to fuse the planes
    assert 'split_stage1' not in msims[0]['scale0'].ds.data_vars


def test_fused_planes_share_one_grid_the_union_of_all_tiles_with_a_margin():
    from muvis_align.constants import split_grid_margin
    from muvis_align.split_registration import group_grid

    msims = [textured_msim(np.ones((100, 200), dtype=np.float32), origin_x) for origin_x in (0.0, 7.0)]

    grid, tile_size = group_grid(msims, 'source')

    margins = {dim: int(np.ceil(size * split_grid_margin)) for dim, size in (('y', 100), ('x', 207))}
    assert grid['spacing'] == {'y': 1.0, 'x': 1.0}
    assert grid['shape'] == {'y': 100 + 2 * margins['y'], 'x': 207 + 2 * margins['x']}
    assert grid['origin'] == {'y': -margins['y'], 'x': -margins['x']}
    assert tile_size == 200


def test_a_fused_plane_is_smoothed_with_its_background_filled_by_its_mean():
    from muvis_align.split_registration import smooth_group

    data = np.zeros((60, 200), dtype=np.float32)
    data[:, :50] = 10.0
    data[:, 50:100] = 30.0

    smoothed = smooth_group(data, spacing=1.0, tile_size=25.0)

    # far from its edge, the empty background holds the plane's mean instead of zero
    assert np.allclose(smoothed[:, 150:], 20.0, atol=1e-4)
    assert np.all(smoothed > 0)


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
