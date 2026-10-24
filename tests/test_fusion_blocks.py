import glob
import os

import numpy as np
import pytest
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.fusion_blocks import (export_levels, fuse_levels_to_ome_zarr, native_level_spacings,
                                      native_level_stack_properties, block_sources)
from tests.data_builders import make_msim, make_sim


@pytest.mark.parametrize('source_spacings, shape, expected', [
    # tiles and overviews: doubling from the tiles, then a level at the overviews' own size
    ([0.01] * 5 + [0.249] * 2, [3000, 4000], [0.01, 0.02, 0.04, 0.08, 0.16, 0.249]),
    # a coarser source gets its level even below the size the doubling stops at
    ([0.5, 2.0], [256, 256], [0.5, 1.0, 2.0]),
    # sizes under sqrt(2) apart are one level, at the finer
    ([0.01, 0.0102], [300, 400], [0.01, 0.02]),
    # a step of under sqrt(2) to a source size replaces the level before it
    ([0.01, 0.17], [3000, 4000], [0.01, 0.02, 0.04, 0.08, 0.17, 0.34]),
    # a few sections imaged coarser join their overviews' level, doubling on from it (the HPC's 0.3322 among 0.249)
    ([0.01] + [0.2492] * 1048 + [0.2469] * 30 + [0.3322] * 3, [12000, 16000],
     [0.01, 0.02, 0.04, 0.08, 0.16, 0.2469, 0.4938, 0.9876]),
    # a size sqrt(2) or more above the group before it still gets its own level
    ([0.01, 0.249, 0.36], [3000, 4000], [0.01, 0.02, 0.04, 0.08, 0.16, 0.249, 0.36]),
    # a single size: plain doubling, down to about 100 pixels
    ([1.0], [1000, 800], [1.0, 2.0, 4.0, 8.0]),
])
def test_native_levels_never_upsample_and_meet_every_source_size(source_spacings, shape, expected):
    assert native_level_spacings(source_spacings, shape) == pytest.approx(expected)


def test_a_source_joining_a_coarser_level_is_fused_from_that_level_on():
    """An overview section imaged at 0.3322 among overviews at 0.2469 is in the 0.2469 level, not only from the next."""
    level0 = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 0.01, 'x': 0.01}, 'shape': {'y': 12000, 'x': 16000}}

    levels = export_levels([0.01, 0.2469, 0.3322], level0, ['y', 'x'])

    assert [(round(spacing, 4), selected) for spacing, _, selected in levels] == [
        (0.01, [0]), (0.02, [0]), (0.04, [0]), (0.08, [0]), (0.16, [0]), (0.2469, [0, 1, 2]), (0.4938, [0, 1, 2]),
        (0.9876, [0, 1, 2])]


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
    from muvis_align.fusion_blocks import source_bounds

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
@pytest.mark.parametrize('peak_only', [False, True])
def test_native_fusion_keeps_the_overview_out_of_the_tiles_levels(tmp_path, ngff_version, peak_only, caplog,
                                                                  monkeypatch):
    """A tile (200, at 0.5) inside an overview (10, at 2.0): the finer levels hold the tile alone, written only
    where it is; the overview's level holds both, its one block logged with what was budgeted for it. 0.6 stores its
    arrays as 0.5 does, its metadata naming a coordinate system each level's transforms lead into."""
    import json
    import logging
    from multiview_stitcher import ngff_utils

    overview, tile = flat_msim(10, 64, 2.0, 0.0), flat_msim(200, 64, 0.5, 40.0)
    level0 = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 0.5, 'x': 0.5}, 'shape': {'y': 256, 'x': 256}}
    url = (tmp_path / 'fused.ome.zarr').as_posix()

    if peak_only:
        # as on macOS, where only the peak is known
        import muvis_align.fusion_blocks as fusion_blocks
        monkeypatch.setattr(fusion_blocks, 'get_process_memory', lambda: (None, 1024 ** 3))
    with caplog.at_level(logging.INFO):
        fuse_levels_to_ome_zarr([overview, tile], export_levels([2.0, 0.5], level0, ['y', 'x']), url, 'source',
                                level0, {'y': 64, 'x': 64}, zarr_options={'ngff_version': ngff_version},
                                ranks=[2.0, 0.5])

    # the overview's level is one 64x64 block of both sources, 3 float32 arrays each; its only batch is its first high
    joined = [record.getMessage() for record in caplog.records if '(max 2 a block)' in record.getMessage()]
    assert len(joined) == 1
    assert joined[0].startswith('Fusion batch 1/1: 1 blocks, 2 sources (max 2 a block), budgeted 96.0 KB,'
                                ' source levels 8.0 KB (max 8.0 KB a block)')

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


def test_a_native_level_with_the_sources_of_the_level_above_is_averaged_from_it(tmp_path):
    """Fused from a source level finer than itself, a coarser level samples rather than averages it (aliasing);
    where its sources are the level above's, it is that level's 2x2 mean. A level a new source joins is fused. The
    tile is off the output grid, as real ones are: on it, sampling the middle of 2x2 pixels is their mean."""
    import zarr
    import muvis_align.fusion_blocks as fusion_blocks

    rng = np.random.default_rng(0)
    tile = make_msim(rng.integers(50, 250, (64, 64), dtype=np.uint8), scale={'y': 0.5, 'x': 0.5},
                     translation={'y': 40.3, 'x': 40.1}, transform_key='source')
    overview = flat_msim(10, 64, 2.0, 0.0)
    level0 = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 0.5, 'x': 0.5}, 'shape': {'y': 256, 'x': 256}}
    levels = export_levels([2.0, 0.5], level0, ['y', 'x'])
    url = (tmp_path / 'fused.ome.zarr').as_posix()

    fuse_levels_to_ome_zarr([overview, tile], levels, url, 'source', level0, {'y': 64, 'x': 64}, ranks=[2.0, 0.5])

    finest, middle, coarsest = (np.asarray(zarr.open_array(os.path.join(url, path), mode='r')[:]).squeeze()
                                for path in fusion_blocks.level_paths(len(levels)))
    assert [selected for _, _, selected in levels] == [[1], [1], [0, 1]]
    np.testing.assert_array_equal(middle, finest.reshape(128, 2, 128, 2).mean(axis=(1, 3)).astype(np.uint8))
    # the overview's level is fused: the overview where the tile is not
    assert coarsest[5, 5] == 10


@pytest.mark.parametrize('affine', [
    # translated by a fraction of a pixel: axis-aligned, not grid-aligned
    np.array([[1.0, 0.0, 0.3], [0.0, 1.0, -0.6], [0.0, 0.0, 1.0]]),
    # rotated: the general overlap path
    np.array([[np.cos(0.05), -np.sin(0.05), 2.0], [np.sin(0.05), np.cos(0.05), 1.0], [0.0, 0.0, 1.0]]),
])
def test_the_fusion_plan_checks_on_numpy_fuse_what_multiview_stitcher_does(tmp_path, monkeypatch, affine):
    """The per-block plan's affine checks read each transform once instead of an xarray sel per entry."""
    import zarr
    from contextlib import nullcontext
    import muvis_align.fusion_blocks as fusion_blocks

    rng = np.random.default_rng(0)
    tiles = [make_msim(rng.integers(0, 255, (64, 64), dtype=np.uint8), scale={'y': 0.5, 'x': 0.5},
                       translation={'y': 0.0, 'x': offset}, transform_key='source', affine=affine)
             for offset in (0.0, 28.0)]
    # an overview below them gives the coarser levels
    msims = [flat_msim(10, 64, 2.0, 0.0)] + tiles
    level0 = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 0.5, 'x': 0.5}, 'shape': {'y': 256, 'x': 256}}
    levels = export_levels([2.0, 0.5, 0.5], level0, ['y', 'x'])

    def fused(name):
        url = (tmp_path / name).as_posix()
        fuse_levels_to_ome_zarr(msims, levels, url, 'source', level0, {'y': 32, 'x': 32}, ranks=[2.0, 0.5, 0.5])
        return [np.asarray(zarr.open_array(os.path.join(url, path), mode='r')[:])
                for path in fusion_blocks.level_paths(len(levels))]

    with_numpy = fused('numpy.ome.zarr')
    monkeypatch.setattr(fusion_blocks, '_numpy_fusion_plan', nullcontext)
    with_xarray = fused('xarray.ome.zarr')

    assert len(with_numpy) == len(levels) > 1
    for numpy_level, xarray_level in zip(with_numpy, with_xarray):
        assert numpy_level.any()
        np.testing.assert_array_equal(numpy_level, xarray_level)


@pytest.mark.parametrize('matrices', [
    [np.eye(3), np.array([[1.0, 0.0, 0.25], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])],
    [np.eye(3), np.array([[np.cos(0.05), -np.sin(0.05), 2.0], [np.sin(0.05), np.cos(0.05), 1.0], [0, 0, 1.0]])],
    # x only translated: y is not axis-aligned, x is
    [np.array([[1.0, 0.0, 0.0], [0.2, 1.0, 0.5], [0.0, 0.0, 1.0]])],
])
def test_the_numpy_plan_checks_match_multiview_stitchers(matrices):
    import multiview_stitcher.fusion._core as fusion_core
    from multiview_stitcher import param_utils
    from muvis_align.fusion_blocks import (_axis_aligned_translation_dims, _axis_aligned_translation_overlap,
                                          _grid_aligned_translation_dims)

    sdims = ['y', 'x']
    sparams = [param_utils.affine_to_xaffine(matrix) for matrix in matrices]
    views_bb = [{'origin': {'y': 0.0, 'x': 10.0 * index}, 'spacing': {'y': 0.5, 'x': 0.5}, 'shape': {'y': 64, 'x': 64}}
                for index in range(len(sparams))]
    output = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 0.5, 'x': 0.5}, 'shape': {'y': 32, 'x': 32}}

    assert (_axis_aligned_translation_dims(sparams, sdims)
            == fusion_core._get_axis_aligned_translation_dims(sparams=sparams, sdims=sdims))
    assert (_grid_aligned_translation_dims(sparams, views_bb, output, sdims)
            == fusion_core._get_grid_aligned_translation_dims(sparams=sparams, views_bb=views_bb,
                                                             output_stack_properties=output, sdims=sdims))
    for param, view_bb in zip(sparams, views_bb):
        for extent in (None, {'y': 1, 'x': 1}):
            assert (_axis_aligned_translation_overlap(output, view_bb, param, sdims, extent)
                    == fusion_core._get_axis_aligned_translation_overlap(output, view_bb, param, sdims, extent))


@pytest.mark.parametrize('bounds, side, chunk_side, budget, expected', [
    # ten sources stacked in one corner: a block holding them all must shrink, wherever else is empty
    ([[[0.0, 300.0], [0.0, 300.0]]] * 10, 4096, 4096, 10 * 512 * 512 * 12, 512),
    # one source alone keeps the full size
    ([[[0.0, 300.0], [0.0, 300.0]]], 4096, 4096, 4096 * 4096 * 12, 4096),
    # 12 sources over every block, a little over budget: 1472 fits, where halving went to 800
    ([[[0.0, 6400.0], [0.0, 6400.0]]] * 12, 6400, 1600, 352 * 1000 ** 2, 1472),
])
def test_a_block_meeting_many_sources_shrinks_just_enough(bounds, side, chunk_side, budget, expected):
    from muvis_align.fusion_blocks import budget_chunksize

    properties = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 1.0, 'x': 1.0}, 'shape': {'y': side, 'x': side}}

    chunk = budget_chunksize(np.array(bounds), properties, {'y': chunk_side, 'x': chunk_side}, ['y', 'x'], budget, 12)

    assert chunk == {'y': expected, 'x': expected}


def test_where_a_finer_view_covers_a_pixel_the_coarser_ones_are_left_out():
    from dask.utils import has_keyword
    from multiview_stitcher.fusion import simple_average_fusion, weighted_average_fusion
    from muvis_align.fusion_blocks import prioritise_finer_views

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
    from muvis_align.fusion_blocks import _view_ranks

    assert _view_ranks(params, group_params, group_ranks).tolist() == expected


def test_level_names_sort_as_text_in_level_order():
    from muvis_align.fusion_blocks import level_paths

    assert level_paths(10) == [str(index) for index in range(10)]
    eleven = level_paths(11)
    assert eleven[0] == '00' and eleven[-1] == '10'
    assert sorted(eleven) == eleven


# the most that may run at once: as many as fit side by side, and one larger than the whole pool alone, not never
@pytest.mark.parametrize('need, capacity, most_allowed', [(2, 10, 4), (6, 10, 1), (20, 10, 1)])
def test_blocks_run_together_only_as_far_as_the_memory_pool_holds(need, capacity, most_allowed):
    import threading
    import time
    from muvis_align.fusion_blocks import _run_blocks

    running, most, fused, lock = [0], [0], [], threading.Lock()

    def fuse_chunk(block_id):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.02)
        with lock:
            running[0] -= 1
            fused.append(block_id)

    _run_blocks(fuse_chunk, list(range(8)), {'workers': 4, 'memory_pool': capacity}, block_bytes=need)

    assert sorted(fused) == list(range(8))
    assert most[0] <= most_allowed


def test_the_memory_pool_leaves_the_fused_output_unchanged(tmp_path):
    """One block at a time (a pool smaller than any block) against the plain sequential run."""
    overview, tile = flat_msim(10, 64, 2.0, 0.0), flat_msim(200, 64, 0.5, 40.0)
    level0 = {'origin': {'y': 0.0, 'x': 0.0}, 'spacing': {'y': 0.5, 'x': 0.5}, 'shape': {'y': 256, 'x': 256}}
    levels = export_levels([2.0, 0.5], level0, ['y', 'x'])
    fused = {}
    for name, options in (('sequential', None), ('pooled', {'workers': 4, 'memory_pool': 1})):
        url = (tmp_path / f'{name}.ome.zarr').as_posix()
        read = fuse_levels_to_ome_zarr([overview, tile], levels, url, 'source', level0, {'y': 32, 'x': 32},
                                       batch_options=options, ranks=[2.0, 0.5])
        fused[name] = [np.asarray(msi_utils.get_sim_from_msim(read, scale=key).data)
                       for key in msi_utils.get_sorted_scale_keys(read)]
    for sequential, pooled in zip(fused['sequential'], fused['pooled']):
        np.testing.assert_array_equal(sequential, pooled)
