import glob
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from multiview_stitcher import msi_utils, param_utils, registration
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.param_resolution import groupwise_resolution

from muvis_align.image.util import build_view_adjacency_graph
from muvis_align.split_registration import (cross_group_edges, fit_transform, register_groups, split_groups,
                                            within_group_graph)


def test_groups_are_z_planes_or_for_channel_registration_channels():
    positions = [{'z': 0.1, 'y': 0, 'x': 0}, {'z': 0.0, 'y': 0, 'x': 5}, {'z': 0.1, 'y': 5, 'x': 0}]
    sources = [SimpleNamespace(get_channels=lambda label=label: [{'label': label}]) for label in ('b', 'a', 'a')]

    assert split_groups(positions, sources) == [1, 0, 1]
    assert split_groups(positions, sources, 'c') == [1, 0, 0]
    graph = nx.Graph([(0, 1), (0, 2), (1, 2)])
    assert cross_group_edges(graph, [1, 0, 1]) == [(0, 1), (1, 2)]
    within = within_group_graph(graph, [1, 0, 1])
    assert list(within.edges) == [(0, 2)] and sorted(within.nodes) == [0, 1, 2]


@pytest.mark.parametrize('transform_type', ['translation', 'rigid', 'similarity', 'affine'])
def test_fit_transform_recovers_a_transform_of_its_type(transform_type):
    rng = np.random.default_rng(1)
    angle, factor = 0.1, (1.2 if transform_type in ('similarity', 'affine') else 1.0)
    linear = np.eye(2) if transform_type == 'translation' else factor * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if transform_type == 'affine':
        linear = linear @ np.array([[1.0, 0.2], [0.0, 0.9]])
    expected = np.eye(3)
    expected[:2, :2], expected[:2, 2] = linear, [3.0, -5.0]
    points = rng.random((20, 2)) * 100

    fitted = fit_transform(points, points @ linear.T + expected[:2, 2], rng.random(20) + 0.1, transform_type)

    assert np.allclose(fitted, expected, atol=1e-9)


def translation_xparam(shift_yx):
    matrix = np.eye(3)
    matrix[:2, 2] = shift_yx
    return param_utils.affine_to_xaffine(matrix, t_coords=[0])


def pair_graph(edges):
    """A pair graph with 50x50 tiles at spacing 1: {edge: (shift of the pair's transform, quality)}."""
    graph = nx.Graph()
    for node in {node for edge in edges for node in edge}:
        graph.add_node(node, stack_props={'spacing': {'y': 1.0, 'x': 1.0}})
    template = translation_xparam([0, 0])
    for edge, (shift, quality) in edges.items():
        bbox = template.isel(x_out=[0, 1]).isel(x_in=[0, 1]).rename({'x_in': 'point_index', 'x_out': 'dim'})
        graph.add_edge(*edge, transform=translation_xparam(shift),
                       quality=template.isel(x_in=0, x_out=0).copy(data=[quality]),
                       bbox=bbox.copy(data=[[[0.0, 0.0], [50.0, 50.0]]]), overlap=2500.0)
    return graph


def translations(transforms):
    return np.array([np.asarray(transform).squeeze()[:2, 2] for transform in transforms])


def test_a_misplaced_plane_is_moved_back_by_its_tile_pairs_and_an_outlier_pair_does_not_pull_it():
    """Planes 0 (tiles 0, 1) and 1 (tiles 2, 3); the pairs across say plane 1 sits 7 too far in x, one says 30."""
    shift = np.array([0.0, 7.0])
    graph = pair_graph({(0, 1): ([0, 0], 1.0), (2, 3): ([0, 0], 1.0),
                        (0, 2): (shift, 0.9), (1, 3): (shift, 0.8), (0, 3): ([0, 30.0], 0.9)})
    identity = translation_xparam([0, 0])

    transforms = register_groups(graph, [identity] * 4, [0, 0, 1, 1],
                                 resolution_kwargs={'transform': 'translation'})

    result = translations(transforms)
    assert np.allclose(result[:2], 0, atol=0.05)
    assert np.allclose(result[2:], -shift, atol=0.05)


def test_the_tile_pairs_across_planes_are_measured_from_the_stage1_placement():
    """Stage 1 moved tile 3 by (0, 2) within plane 1; a pair measured before that moves with it."""
    stage1 = [translation_xparam(shift) for shift in ([0, 0], [0, 0], [0, 0], [0, 2.0])]
    # at the metadata positions: tile 2 is 7 off, tile 3 (before its stage-1 correction) 5
    graph = pair_graph({(0, 2): ([0, 7.0], 1.0), (1, 3): ([0, 5.0], 1.0)})

    transforms = register_groups(graph, stage1, [0, 0, 1, 1], resolution_kwargs={'transform': 'translation'})

    result = translations(transforms)
    assert np.allclose(result[2], [0, -7.0], atol=0.05)
    assert np.allclose(result[3], [0, -5.0], atol=0.05)


def test_a_group_without_tile_pairs_to_the_others_keeps_its_stage1_placement():
    graph = pair_graph({(0, 1): ([0, 7.0], 1.0), (2, 3): ([0, 0], 1.0)})
    graph.add_node(4, stack_props={'spacing': {'y': 1.0, 'x': 1.0}})

    transforms = register_groups(graph, [translation_xparam([0, 0])] * 5, [0, 1, 2, 2, 3],
                                 resolution_kwargs={'transform': 'translation'})

    assert np.allclose(translations(transforms), [[0, 0], [0, -7.0], [0, 0], [0, 0], [0, 0]], atol=0.05)


def textured_msim(image, origin_yx):
    sim = si_utils.get_sim_from_array(image, dims=['y', 'x'], scale={'y': 1.0, 'x': 1.0},
                                      translation={'y': origin_yx[0], 'x': origin_yx[1]}, transform_key='source')
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    return msi_utils.multiscale_sel_coords(msim, {'c': sim.coords['c'].values[0]})


def test_two_stages_of_real_pair_registrations_place_every_tile():
    """Two tiles a plane, both planes the same content; in the metadata plane 1 sits (3, -6) off and its second tile
    another (0, 4) off: phase correlation within and across planes, as the plugin's split pairing runs it."""
    rng = np.random.default_rng(0)
    image = gaussian_blurred(rng.random((160, 260)))
    true_origins = [(0, 0), (0, 100)] * 2
    errors = [(0, 0), (0, 0), (3, -6), (3, -2)]
    msims = [textured_msim(image[:, origin[1]:origin[1] + 160], np.add(origin, error))
             for origin, error in zip(true_origins, errors)]
    groups = [0, 0, 1, 1]
    graph = build_view_adjacency_graph(msims, 'source', [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)],
                                       overlap_tolerance=0)
    graph = registration.compute_pairwise_registrations(msims, graph, transform_key='source',
                                                        pairwise_reg_func=registration.phase_correlation_registration)
    stage1, _ = groupwise_resolution(within_group_graph(graph, groups), method='robust_linear',
                                     transform='translation')

    transforms = register_groups(graph, [stage1[node] for node in range(4)], groups,
                                 resolution_kwargs={'transform': 'translation'})

    placed = translations(transforms) + np.array(errors)
    assert np.allclose(placed - placed[0], 0, atol=0.5)


def gaussian_blurred(image):
    from scipy.ndimage import gaussian_filter
    return (gaussian_filter(image, 2) * 1000).astype(np.float32)


def test_split_pairing_registers_the_pairs_across_planes_and_resolves_them_after():
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
    assert cross_group_edges(reg.pairs_graph, groups)
    assert within_group_graph(reg.pairs_graph, groups).number_of_edges() > 0

    results = reg.register_global(reg.pair_msims, params=params)
    assert len(results['mappings']) == len(reg.pair_msims)
