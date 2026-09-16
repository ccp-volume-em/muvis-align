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
from muvis_align.image.util import get_chunk_sizes, get_export_chunk_sizes
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


def make_sources(count, extent, spacing=1.0):
    """`count` sources, each `extent` pixels square at `spacing`.

    Only their shape and spacing are ever read (get_sim_physical_size), so this builds one and
    repeats the reference: the pixels are never touched, and a real 20000px array would be 763MB
    that dask would then tokenize.
    """
    import dask.array as da
    from multiview_stitcher import spatial_image_utils as si_utils
    sim = si_utils.get_sim_from_array(
        da.zeros((extent, extent), dtype=np.uint16, chunks=(1024, 1024)),
        dims=['y', 'x'], scale={'y': spacing, 'x': spacing})
    return [sim] * count


def output_properties(side, spacing=1.0, z=None):
    shape = {'y': side, 'x': side}
    spacings = {'y': spacing, 'x': spacing}
    origin = {'y': 0.0, 'x': 0.0}
    if z is not None:
        shape = {'z': z, **shape}
        spacings = {'z': 1.0, **spacings}
        origin = {'z': 0.0, **origin}
    return {'shape': shape, 'spacing': spacings, 'origin': origin}


def implied_block_bytes(sizes, sources_in_block):
    from muvis_align.constants import fusion_stack_arrays
    voxels = np.prod([size for size in sizes.values()])
    return sources_in_block * voxels * 4 * fusion_stack_arrays


def test_a_full_resolution_export_is_not_sized_as_if_every_source_met_every_block():
    """get_chunk_sizes() takes every source in a plane as landing in one chunk - true of a coarse
    preview level, not of a full-resolution export, where a block spans a few microns. Sized that
    way a source-dense export lands on 128-pixel blocks, hundreds of thousands of them."""
    sources = make_sources(400, extent=2000)
    props = output_properties(40000)

    by_count = get_chunk_sizes(np.dtype('uint16'), list(props['shape']), num_sources=400,
                               num_z_positions=1, xy_chunk_size=default_export_chunk_size)
    by_geometry = get_export_chunk_sizes(np.dtype('uint16'), props, sources)

    assert by_geometry['y'] > by_count['y'] * 4, (
        f'geometry {by_geometry} barely improved on source count {by_count}')


def test_overlapping_sources_still_shrink_the_block():
    """The budget exists because a block holds every source that reaches it. Sources piled on the
    same ground reach the same blocks, and must still pull the size down."""
    spread = get_export_chunk_sizes(np.dtype('uint16'), output_properties(40000),
                                    make_sources(400, extent=2000))
    piled = get_export_chunk_sizes(np.dtype('uint16'), output_properties(40000),
                                   make_sources(400, extent=20000))

    assert piled['y'] < spread['y'], 'heavy overlap did not shrink the block'


def test_the_block_stays_inside_the_memory_budget():
    """The sizer's own arithmetic, checked against the budget it is given: one block's fusion
    stack must fit, for a mosaic and for a heavily overlapped set alike."""
    from muvis_align.constants import default_export_fusion_chunk_bytes

    for count, extent, side in ((400, 2000, 40000), (400, 20000, 40000), (4000, 1000, 60000)):
        sources = make_sources(count, extent=extent)
        props = output_properties(side)
        sizes = get_export_chunk_sizes(np.dtype('uint16'), props, sources)
        # the same estimate the sizer makes: sources reaching one block of this size
        density = count / side ** 2
        reaching = min(count, max(1.0, density * (sizes['y'] + extent) * (sizes['x'] + extent)))
        assert implied_block_bytes(sizes, reaching) <= default_export_fusion_chunk_bytes * 1.01, (
            f'{count} sources of {extent}px over {side}px: {sizes} exceeds the budget')


def test_the_export_cap_still_applies():
    """A handful of small sources over a large output could take any block size; the cap is what
    keeps the on-disk chunk sane, since for a zarr export the two are the same value."""
    sizes = get_export_chunk_sizes(np.dtype('uint16'), output_properties(100000),
                                   make_sources(4, extent=100))
    assert sizes['y'] == default_export_chunk_size


def test_sources_at_distinct_z_keep_one_plane_per_block():
    sources = make_sources(54, extent=6400)
    sizes = get_export_chunk_sizes(np.dtype('uint16'), output_properties(18399, z=6), sources,
                                   num_z_positions=6)
    assert sizes['z'] == 1
    assert sizes['y'] > 1024, 'the real subset export should block far larger than its tile_size'
