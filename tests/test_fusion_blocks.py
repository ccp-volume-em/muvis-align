"""How a zarr export's fusion is blocked, and that its blocks run concurrently.

A 6.8GB export ran for hours on one core: multiview_stitcher walks blocks sequentially unless
given a batch_func, and the block size was the configured tile_size (1024), whose fixed per-block
cost was paid 3600 times. Measured on 54 real sources over identical output pixels: 64.0s
sequential at 1024, 15.9s parallel at the 3328 the sizer picks.
"""
import threading

import numpy as np
import pytest

from muvis_align.constants import (default_export_chunk_size, default_export_fusion_chunk_bytes,
                                   default_fusion_workers, fusion_stack_arrays)
from muvis_align.image.util import get_chunk_sizes, get_export_chunk_sizes
from muvis_align.MVSRegistration import MVSRegistration


def make_sources(count, extent):
    """`count` sources, each `extent` pixels square.

    Only shape and spacing are ever read (get_sim_physical_size), so this builds one and repeats
    the reference: a real 20000px array would be 763MB for dask to then tokenize.
    """
    import dask.array as da
    from multiview_stitcher import spatial_image_utils as si_utils
    sim = si_utils.get_sim_from_array(da.zeros((extent, extent), dtype=np.uint16,
                                               chunks=(1024, 1024)),
                                      dims=['y', 'x'], scale={'y': 1.0, 'x': 1.0})
    return [sim] * count


def output_properties(side, z=None):
    shape, spacing, origin = {'y': side, 'x': side}, {'y': 1.0, 'x': 1.0}, {'y': 0.0, 'x': 0.0}
    if z is not None:
        shape, spacing, origin = {'z': z, **shape}, {'z': 1.0, **spacing}, {'z': 0.0, **origin}
    return {'shape': shape, 'spacing': spacing, 'origin': origin}


@pytest.mark.parametrize('saving_zarr, workers, expected', [
    (False, None, None),    # the in-memory path builds a lazy graph dask already parallelises
    (True, 1, None),
    (True, 4, 4),
    (True, None, default_fusion_workers if default_fusion_workers > 1 else None),
])
def test_only_a_zarr_export_batches_its_blocks(saving_zarr, workers, expected):
    options = MVSRegistration._fusion_batch_options(saving_zarr, max_workers=workers)
    assert (options['n_batch'] if options else None) == expected


def test_a_batch_fuses_every_block_once_and_concurrently():
    """Sequentially these would deadlock on the barrier, so reaching the assert is the test."""
    workers = 4
    barrier = threading.Barrier(workers, timeout=10)
    fused, lock = [], threading.Lock()

    def fuse_chunk(block_id):
        barrier.wait()
        with lock:
            fused.append(block_id)

    blocks = [(0, i) for i in range(workers)]
    MVSRegistration._fusion_batch_options(True, max_workers=workers)['batch_func'](
        fuse_chunk, blocks)

    assert sorted(fused) == sorted(blocks)


def test_a_failing_block_is_not_swallowed():
    """A block that raises must fail the export, not leave a hole in the output."""
    def fuse_chunk(block_id):
        if block_id == 3:
            raise ValueError('block 3')

    options = MVSRegistration._fusion_batch_options(True, max_workers=4)
    with pytest.raises(ValueError, match='block 3'):
        options['batch_func'](fuse_chunk, list(range(8)))


def test_a_full_resolution_export_is_not_sized_as_if_every_source_met_every_block():
    """get_chunk_sizes() takes every source in a plane as landing in one chunk - true of a coarse
    preview level, not of a full-resolution export, where a block spans a few microns. Sized that
    way a source-dense export lands on 128-pixel blocks, hundreds of thousands of them."""
    props = output_properties(40000)
    by_count = get_chunk_sizes(np.dtype('uint16'), list(props['shape']), num_sources=400,
                               num_z_positions=1, xy_chunk_size=default_export_chunk_size)
    by_geometry = get_export_chunk_sizes(np.dtype('uint16'), props, make_sources(400, extent=2000))

    assert by_geometry['y'] > by_count['y'] * 4, (
        f'geometry {by_geometry} barely improved on source count {by_count}')


@pytest.mark.parametrize('count, extent, side, expected', [
    (4, 100, 100000, 'capped'),        # too sparse to bind: the cap is what keeps chunks sane
    (400, 2000, 40000, 'budgeted'),    # a mosaic
    (400, 20000, 40000, 'budgeted'),   # the same sources piled on the same ground
    (4000, 1000, 60000, 'budgeted'),
])
def test_a_block_fits_the_budget_and_the_cap(count, extent, side, expected):
    sizes = get_export_chunk_sizes(np.dtype('uint16'), output_properties(side),
                                   make_sources(count, extent))

    assert sizes['y'] <= default_export_chunk_size
    if expected == 'capped':
        assert sizes['y'] == default_export_chunk_size
    # the sizer's own estimate of the sources reaching one block of this size
    density = count / side ** 2
    reaching = min(count, max(1.0, density * (sizes['y'] + extent) * (sizes['x'] + extent)))
    block_bytes = reaching * sizes['y'] * sizes['x'] * 4 * fusion_stack_arrays
    assert block_bytes <= default_export_fusion_chunk_bytes * 1.01, f'{sizes} exceeds the budget'


def test_overlapping_sources_shrink_the_block():
    """The budget exists because a block holds every source that reaches it, and sources piled on
    the same ground reach the same blocks."""
    spread = get_export_chunk_sizes(np.dtype('uint16'), output_properties(40000),
                                    make_sources(400, extent=2000))
    piled = get_export_chunk_sizes(np.dtype('uint16'), output_properties(40000),
                                   make_sources(400, extent=20000))

    assert piled['y'] < spread['y']


def test_sources_at_distinct_z_keep_one_plane_per_block():
    sizes = get_export_chunk_sizes(np.dtype('uint16'), output_properties(18399, z=6),
                                   make_sources(54, extent=6400), num_z_positions=6)

    assert sizes['z'] == 1
    assert sizes['y'] > 1024, 'the real subset export should block far larger than its tile_size'
