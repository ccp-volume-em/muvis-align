import glob
import os

import numpy as np
import pytest
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.fusion_slabs import (fuse_native_levels_to_ome_zarr, native_level_spacings,
                                      native_level_stack_properties, block_sources)
from tests.data_builders import make_msim, make_sim


@pytest.mark.parametrize('source_spacings, shape, expected', [
    # tiles and overviews: doubling from the tiles, then a level at the overviews' own size
    ([0.01] * 5 + [0.249] * 2, [3000, 4000], [0.01, 0.02, 0.04, 0.08, 0.16, 0.249]),
    # a coarser source gets its level even below the size the doubling stops at
    ([0.5, 2.0], [256, 256], [0.5, 1.0, 2.0]),
    # sizes within the tolerance are one
    ([0.01, 0.0102], [300, 400], [0.01, 0.02]),
    # a step of under sqrt(2) to a source size replaces the level before it
    ([0.01, 0.17], [3000, 4000], [0.01, 0.02, 0.04, 0.08, 0.17, 0.34]),
    # ...when that level is a doubled one, never another source's size (the HPC's overviews at 0.249 and 0.3322)
    ([0.01, 0.249, 0.3322], [3000, 4000], [0.01, 0.02, 0.04, 0.08, 0.16, 0.249, 0.3322]),
    ([0.01, 0.2, 0.249], [3000, 4000], [0.01, 0.02, 0.04, 0.08, 0.2, 0.249]),
    # a single size: plain doubling, down to about 100 pixels
    ([1.0], [1000, 800], [1.0, 2.0, 4.0, 8.0]),
])
def test_native_levels_never_upsample_and_meet_every_source_size(source_spacings, shape, expected):
    assert native_level_spacings(source_spacings, shape) == pytest.approx(expected)


def test_native_levels_share_one_extent_and_keep_an_unscaled_dim():
    level0 = {'origin': {'z': 0.0, 'y': 0.0, 'x': 0.0}, 'spacing': {'z': 1.0, 'y': 0.5, 'x': 0.5},
              'shape': {'z': 3, 'y': 256, 'x': 200}}

    levels = native_level_stack_properties(level0, [0.5, 2.0], scaled_dims=['y', 'x'])

    assert levels[1]['spacing'] == {'z': 1.0, 'y': 2.0, 'x': 2.0}
    assert levels[1]['shape'] == {'z': 3, 'y': 64, 'x': 50}
    # pixel centres: a 4x coarser pixel's centre sits 1.5 fine pixels in
    assert levels[1]['origin'] == {'z': 0.0, 'y': 0.75, 'x': 0.75}


def test_each_block_is_fused_from_only_the_sources_reaching_it():
    properties = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 1.0, 'x': 1.0}, 'shape': {'y': 40, 'x': 40}}
    # one source inside block (1, 2), one across blocks (0, 0) and (0, 1)
    bounds = np.array([[[12.0, 18.0], [25.0, 28.0]], [[2.0, 5.0], [5.0, 15.0]]])

    groups = block_sources(bounds, properties, {'y': 10, 'x': 10}, ['y', 'x'])

    assert groups == {(1,): [(0, 0), (0, 1)], (0,): [(1, 2)]}


# a section step other than 1 is not grid-aligned with a plane's placeholder z spacing of 1
@pytest.mark.parametrize('z_step', [1.0, 0.05])
def test_a_block_is_fused_from_only_the_planes_that_reach_it(z_step):
    from muvis_align.fusion_slabs import source_bounds

    sims = [make_sim(np.ones((1, 8, 8), dtype=np.uint16), 'zyx', translation={'z': plane * z_step, 'y': 0.0, 'x': 0.0},
                     transform_key='source') for plane in (0, 0, 1, 2)]
    properties = {'origin': {'z': 0.0, 'y': 0.0, 'x': 0.0}, 'spacing': {'z': z_step, 'y': 1.0, 'x': 1.0},
                  'shape': {'z': 3, 'y': 8, 'x': 8}}
    bounds = source_bounds(sims, 'source', properties)

    by_plane = block_sources(bounds, properties, {'z': 1, 'y': 8, 'x': 8}, ['z', 'y', 'x'])
    by_two_planes = block_sources(bounds, properties, {'z': 2, 'y': 8, 'x': 8}, ['z', 'y', 'x'])

    assert by_plane == {(0, 1): [(0, 0, 0)], (2,): [(1, 0, 0)], (3,): [(2, 0, 0)]}
    assert by_two_planes == {(0, 1, 2): [(0, 0, 0)], (3,): [(1, 0, 0)]}


def flat_msim(value, size, spacing, origin):
    return make_msim(np.full((size, size), value, dtype=np.uint8), scale={'y': spacing, 'x': spacing},
                     translation={'y': origin, 'x': origin}, transform_key='source')


@pytest.mark.parametrize('ngff_version', ['0.5', '0.6'])
def test_native_fusion_keeps_the_overview_out_of_the_tiles_levels(tmp_path, ngff_version):
    """A tile (200, at 0.5) inside an overview (10, at 2.0): the finer levels hold the tile alone, written only
    where it is; the overview's level holds both. 0.6 stores its arrays as 0.5 does, its metadata naming a
    coordinate system each level's transforms lead into."""
    import json
    from multiview_stitcher import ngff_utils

    overview, tile = flat_msim(10, 64, 2.0, 0.0), flat_msim(200, 64, 0.5, 40.0)
    level0 = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 0.5, 'x': 0.5}, 'shape': {'y': 256, 'x': 256}}
    url = (tmp_path / 'fused.ome.zarr').as_posix()

    fuse_native_levels_to_ome_zarr([overview, tile], [2.0, 0.5], url, 'source', level0, {'y': 64, 'x': 64},
                                   ['y', 'x'], zarr_options={'ngff_version': ngff_version})

    group = json.load(open(os.path.join(url, 'zarr.json')))
    assert group['zarr_format'] == 3
    assert group['attributes']['ome']['version'] == ngff_version
    assert ('coordinateSystems' in group['attributes']['ome']['multiscales'][0]) == (ngff_version == '0.6')
    read = ngff_utils.read_msim_from_ome_zarr(url, transform_key='read', array_backend='dask')
    levels = [msi_utils.get_sim_from_msim(read, scale=key) for key in msi_utils.get_sorted_scale_keys(read)]
    assert [si_utils.get_spacing_from_sim(level)['x'] for level in levels] == pytest.approx([0.5, 1.0, 2.0])
    finest = np.asarray(levels[0].data).squeeze()
    assert finest[100, 100] == 200 and finest[10, 10] == 0
    # 4 of the 16 blocks reach the tile; the rest are never written
    chunks = [path for path in glob.glob(os.path.join(url, '0', 'c', '**', '*'), recursive=True) if os.path.isfile(path)]
    assert len(chunks) == 4
    coarsest = np.asarray(levels[2].data).squeeze()
    assert coarsest[5, 5] == 10
    assert coarsest[28, 28] == 200


@pytest.mark.parametrize('bounds, side, chunk_side, budget, expected', [
    # ten sources stacked in one corner: a block holding them all must shrink, wherever else is empty
    ([[[0.0, 300.0], [0.0, 300.0]]] * 10, 4096, 4096, 10 * 512 * 512 * 12, 512),
    # one source alone keeps the full size
    ([[[0.0, 300.0], [0.0, 300.0]]], 4096, 4096, 4096 * 4096 * 12, 4096),
    # 12 sources over every block, a little over budget: 1472 fits, where halving went to 800
    ([[[0.0, 6400.0], [0.0, 6400.0]]] * 12, 6400, 1600, 352 * 1000 ** 2, 1472),
])
def test_a_block_meeting_many_sources_shrinks_just_enough(bounds, side, chunk_side, budget, expected):
    from muvis_align.fusion_slabs import budget_chunksize

    properties = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 1.0, 'x': 1.0}, 'shape': {'y': side, 'x': side}}

    chunk = budget_chunksize(np.array(bounds), properties, {'y': chunk_side, 'x': chunk_side}, ['y', 'x'], budget, 12)

    assert chunk == {'y': expected, 'x': expected}


def test_where_a_finer_view_covers_a_pixel_the_coarser_ones_are_left_out():
    from dask.utils import has_keyword
    from multiview_stitcher.fusion import simple_average_fusion, weighted_average_fusion
    from muvis_align.fusion_slabs import prioritise_finer_views

    nan = np.nan
    # views: two tiles (rank 1) meeting in the middle, an overview (rank 2) under all of it
    views = np.array([[10.0, 20.0, nan, nan], [nan, 40.0, 60.0, nan], [5.0, 5.0, 5.0, 5.0]])
    params = [np.eye(3) * scale for scale in (1, 2, 3)]
    fused = prioritise_finer_views(simple_average_fusion, params, [1, 1, 2])(transformed_views=views, params=params)
    np.testing.assert_array_equal(fused, [10, 30, 60, 5])

    weighted = prioritise_finer_views(weighted_average_fusion, params, [1, 1, 2])
    assert has_keyword(weighted, 'blending_weights') and has_keyword(weighted, 'params')
    # only the tiles' weights count where they cover: a 3:1 split of 20 and 40, renormalised without the overview
    weights = np.array([[0.5, 0.3, 0.0, 0.0], [0.0, 0.1, 0.5, 0.0], [0.5, 0.6, 0.5, 1.0]])
    fused = weighted(transformed_views=views, blending_weights=weights, params=params)
    np.testing.assert_allclose(fused, [10, 25, 60, 5])


@pytest.mark.parametrize('params, group_params, group_ranks, expected', [
    # only some of a block's sources reach it
    ([np.eye(3), np.eye(3) * 3], [np.eye(3) * scale for scale in (1, 2, 3)], [1, 1, 2], [1, 2]),
    # a view passed without its singleton z
    ([np.diag([2.0, 2.0, 1.0])], [np.diag([1.0, 1.0, 1.0, 1.0]), np.diag([1.0, 2.0, 2.0, 1.0])], [1, 2], [2]),
])
def test_views_are_ranked_by_the_group_source_they_come_from(params, group_params, group_ranks, expected):
    from muvis_align.fusion_slabs import _view_ranks

    assert _view_ranks(params, group_params, group_ranks).tolist() == expected


def test_level_names_sort_as_text_in_level_order():
    from muvis_align.fusion_slabs import level_paths

    assert level_paths(10) == [str(index) for index in range(10)]
    eleven = level_paths(11)
    assert eleven[0] == '00' and eleven[-1] == '10'
    assert sorted(eleven) == eleven
