import glob
import os

import numpy as np
import pytest
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.fusion_slabs import (fuse_native_levels_to_ome_zarr, native_level_spacings,
                                      native_level_stack_properties, block_sources)


@pytest.mark.parametrize('source_spacings, shape, expected', [
    # tiles and overviews: doubling from the tiles, then a level at the overviews' own size
    ([0.01] * 5 + [0.249] * 2, [3000, 4000], [0.01, 0.02, 0.04, 0.08, 0.16, 0.249]),
    # a coarser source gets its level even below the size the doubling stops at
    ([0.5, 2.0], [256, 256], [0.5, 1.0, 2.0]),
    # sizes within the tolerance are one
    ([0.01, 0.0102], [300, 400], [0.01, 0.02]),
    # a step of under sqrt(2) to a source size replaces the level before it
    ([0.01, 0.17], [3000, 4000], [0.01, 0.02, 0.04, 0.08, 0.17, 0.34]),
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


def flat_msim(value, size, spacing, origin):
    sim = si_utils.get_sim_from_array(np.full((size, size), value, dtype=np.uint8), dims=['y', 'x'],
                                      scale={'y': spacing, 'x': spacing},
                                      translation={'y': origin, 'x': origin}, transform_key='source')
    return msi_utils.get_msim_from_sim(sim, scale_factors=[])


def test_native_fusion_keeps_the_overview_out_of_the_tiles_levels(tmp_path):
    """A tile (200, at 0.5) inside an overview (10, at 2.0): the finer levels hold the tile alone, written only
    where it is; the overview's level holds both, the tile over the overview where it covers it."""
    overview, tile = flat_msim(10, 64, 2.0, 0.0), flat_msim(200, 64, 0.5, 40.0)
    level0 = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 0.5, 'x': 0.5}, 'shape': {'y': 256, 'x': 256}}
    url = (tmp_path / 'fused.ome.zarr').as_posix()

    fused = fuse_native_levels_to_ome_zarr([overview, tile], [2.0, 0.5], url, 'source', level0,
                                           {'y': 64, 'x': 64}, ['y', 'x'], zarr_options={'ngff_version': '0.5'})

    levels = [msi_utils.get_sim_from_msim(fused, scale=key) for key in msi_utils.get_sorted_scale_keys(fused)]
    assert [si_utils.get_spacing_from_sim(level)['x'] for level in levels] == pytest.approx([0.5, 1.0, 2.0])
    finest = np.asarray(levels[0].data).squeeze()
    assert finest[100, 100] == 200 and finest[10, 10] == 0
    # 4 of the 16 blocks reach the tile; the rest are never written
    chunks = [path for path in glob.glob(os.path.join(url, '0', 'c', '**', '*'), recursive=True) if os.path.isfile(path)]
    assert len(chunks) == 4
    coarsest = np.asarray(levels[2].data).squeeze()
    assert coarsest[5, 5] == 10
    assert coarsest[28, 28] == 200


def test_a_block_meeting_many_sources_is_made_smaller():
    """Ten sources stacked in one corner: a block holding them all must shrink, wherever else is empty."""
    from muvis_align.fusion_slabs import budget_chunksize

    properties = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 1.0, 'x': 1.0}, 'shape': {'y': 4096, 'x': 4096}}
    bounds = np.array([[[0.0, 300.0], [0.0, 300.0]]] * 10)
    budget = 10 * 512 * 512 * 12

    chunk = budget_chunksize(bounds, properties, {'y': 4096, 'x': 4096}, ['y', 'x'], budget, 12)

    assert chunk == {'y': 512, 'x': 512}
    # one source alone keeps the full size
    assert budget_chunksize(bounds[:1], properties, {'y': 4096, 'x': 4096}, ['y', 'x'], 4096 * 4096 * 12, 12) == \
        {'y': 4096, 'x': 4096}


def test_a_block_just_over_budget_shrinks_a_little_not_by_half():
    """12 sources over every block, at 1600px a little over budget: 1472 fits, where halving went to 800."""
    from muvis_align.fusion_slabs import budget_chunksize

    properties = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 1.0, 'x': 1.0}, 'shape': {'y': 6400, 'x': 6400}}
    bounds = np.array([[[0.0, 6400.0], [0.0, 6400.0]]] * 12)
    budget = 352 * 1000 ** 2

    chunk = budget_chunksize(bounds, properties, {'y': 1600, 'x': 1600}, ['y', 'x'], budget, 12)

    assert chunk == {'y': 1472, 'x': 1472}
    assert 12 * 1472 * 1472 * 12 <= budget < 12 * 1600 * 1600 * 12


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


def test_views_are_ranked_when_only_some_of_a_blocks_sources_reach_it():
    from muvis_align.fusion_slabs import _view_ranks

    group_params = [np.eye(3) * scale for scale in (1, 2, 3)]

    assert _view_ranks([group_params[0], group_params[2]], group_params, [1, 1, 2]).tolist() == [1, 2]


def test_a_view_passed_without_its_singleton_z_is_still_ranked():
    from muvis_align.fusion_slabs import _view_ranks

    group_params = [np.diag([1.0, 1.0, 1.0, 1.0]), np.diag([1.0, 2.0, 2.0, 1.0])]

    assert _view_ranks([np.diag([2.0, 2.0, 1.0])], group_params, [1, 2]).tolist() == [2]
