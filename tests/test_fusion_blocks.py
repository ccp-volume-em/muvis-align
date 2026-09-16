"""How a zarr export's fusion is blocked, and that its blocks run concurrently.

A 6.8GB export ran for hours on one core: multiview_stitcher walks its blocks sequentially unless
given a batch_func, and the block size was the configured on-disk tile_size (1024), which carries
a fixed per-block cost 3600 times over. Measured on 54 real sources, over identical output pixels:
61.4s sequential at 1024, 15.8s parallel at 4096.
"""
import threading

import numpy as np
import pytest

from muvis_align.constants import default_export_chunk_size, default_fusion_workers
from muvis_align.image.util import get_chunk_sizes
from muvis_align.MVSRegistration import MVSRegistration


def test_no_batching_for_a_non_zarr_fusion():
    """Only the zarr export walks blocks itself; the in-memory path builds a lazy graph dask
    already parallelises."""
    assert MVSRegistration._fusion_batch_options(False) is None


def test_every_block_is_fused_exactly_once():
    fused = []
    lock = threading.Lock()

    def fuse_chunk(block_id):
        with lock:
            fused.append(block_id)

    options = MVSRegistration._fusion_batch_options(True, max_workers=4)
    blocks = [(0, i) for i in range(10)]
    options['batch_func'](fuse_chunk, blocks)

    assert sorted(fused) == sorted(blocks)


def test_blocks_of_a_batch_run_concurrently():
    """The point of the batch_func: sequentially these would deadlock on the barrier."""
    workers = 4
    barrier = threading.Barrier(workers, timeout=10)
    options = MVSRegistration._fusion_batch_options(True, max_workers=workers)

    def fuse_chunk(_block_id):
        barrier.wait()

    options['batch_func'](fuse_chunk, list(range(workers)))
    assert options['n_batch'] == workers


def test_a_failing_block_is_not_swallowed():
    """A block that raises must fail the export, not leave a hole in the output."""
    def fuse_chunk(block_id):
        if block_id == 3:
            raise ValueError('block 3')

    options = MVSRegistration._fusion_batch_options(True, max_workers=4)
    with pytest.raises(ValueError, match='block 3'):
        options['batch_func'](fuse_chunk, list(range(8)))


def test_one_worker_stays_sequential():
    assert MVSRegistration._fusion_batch_options(True, max_workers=1) is None


def test_default_workers_are_used():
    options = MVSRegistration._fusion_batch_options(True)
    if default_fusion_workers > 1:
        assert options['n_batch'] == default_fusion_workers
    else:
        assert options is None


def test_an_export_may_use_larger_blocks_than_a_preview():
    """The block size is also the export's on-disk chunk size, and 1024 is what a preview wants:
    x/y chunks larger than the screen buy a preview nothing, while an export pays the fixed
    per-block cost for every one of them. Both stay bounded by the memory budget."""
    shape = ['z', 'y', 'x']
    preview = get_chunk_sizes(np.dtype('uint16'), shape, num_sources=54, num_z_positions=6)
    export = get_chunk_sizes(np.dtype('uint16'), shape, num_sources=54, num_z_positions=6,
                             xy_chunk_size=default_export_chunk_size)

    assert export['y'] >= preview['y'] and export['x'] >= preview['x']
    assert export['y'] <= default_export_chunk_size


def test_the_memory_budget_still_bounds_an_export_block():
    """Lifting the preview's cap must not overrule the budget: many sources in one block is what
    the budget exists to bound, and an export is no less exposed to it than a preview."""
    few = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4, num_z_positions=1,
                          xy_chunk_size=default_export_chunk_size)
    many = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4000,
                           num_z_positions=1, xy_chunk_size=default_export_chunk_size)

    assert many['y'] < few['y'], 'the block size ignored how many sources land in one block'


def test_sources_spread_over_z_keep_one_plane_per_block():
    """A block spanning Nz planes pulls in every source from all of them - quadratic, so z stays
    at one plane. The export path takes this from the budget now rather than hard-coding it."""
    sizes = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=54, num_z_positions=6,
                            xy_chunk_size=default_export_chunk_size)
    assert sizes['z'] == 1

    stack = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=54, num_z_positions=1,
                            xy_chunk_size=default_export_chunk_size)
    assert stack['z'] >= 1
