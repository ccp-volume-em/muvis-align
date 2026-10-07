"""get_chunk_sizes() must bound what fusing one output chunk actually costs.

multiview_stitcher's fusion transforms every source overlapping an output chunk into a
full-chunk-sized float32 array and stacks them (fusion._core: field_ims_t, plus a same-shaped
blending-weight stack and their product), so one chunk's peak memory is
~views_in_chunk * chunk_voxels * 4 * fusion_stack_arrays - independent of the output dtype.
fuse() also reuses one output_chunksize for every pyramid level it builds, so the worst case is
a *coarse* level whose whole extent fits in one chunk covering every source.

These tests model that cost over the full pyramid, the way the real fusion graph pays it.
"""
import numpy as np
import pytest
from multiview_stitcher import msi_utils

from muvis_align.constants import default_export_chunk_size, default_export_fusion_chunk_bytes, fusion_stack_arrays
from muvis_align.image.util import get_chunk_sizes, get_export_chunk_sizes


def worst_chunk_fusion_bytes(chunk_sizes, output_shape, num_sources, num_z_positions):
    """Peak bytes for the most expensive single chunk over every pyramid level fuse() builds."""
    dims = list(output_shape)
    sources_per_plane = max(1, round(num_sources / max(1, num_z_positions)))
    level_shapes, _, _ = msi_utils.calc_resolution_levels(output_shape)
    worst = 0
    for shape in level_shapes:
        # a chunk is clipped to the level's own extent - that clipping is exactly what makes
        # coarse levels span every source
        chunk = {dim: min(chunk_sizes[dim], shape[dim]) for dim in dims}
        xy_fraction = np.prod([chunk[dim] / shape[dim] for dim in dims if dim in ('x', 'y')])
        # sources reaching one chunk: those in the z planes it spans, scaled by its share of
        # the field of view (sources tile the output, so area share ~ source share)
        z_span = chunk.get('z', 1) if num_z_positions > 1 else 1
        views = max(1, min(num_sources, sources_per_plane * z_span) * xy_fraction)
        chunk_voxels = np.prod([chunk[dim] for dim in dims])
        worst = max(worst, views * chunk_voxels * 4 * fusion_stack_arrays)
    return worst


# the production default comes from this machine's allocation; a fixed budget keeps the assertions portable
BUDGET = 256 * 1024 ** 2

# (label, num_sources, num_z_positions, output shape)
CASES = [
    # byte-budget-only sizing gave this a deep z chunk whose coarse levels fuse every source in 32 sections
    ('sectioned stack', 4733, 72, {'z': 72, 'y': 6800, 'x': 6800}),
    ('small sectioned stack', 54, 6, {'z': 6, 'y': 2000, 'x': 2000}),
    ('single-plane mosaic', 200, 1, {'y': 20000, 'x': 20000}),
    ('single-plane mosaic 3d', 200, 1, {'z': 8, 'y': 20000, 'x': 20000}),
    ('native z-stacks', 12, 1, {'z': 200, 'y': 4000, 'x': 4000}),
    ('one source', 1, 1, {'z': 100, 'y': 1000, 'x': 1000}),
]


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'float32'])
@pytest.mark.parametrize('label, num_sources, num_z_positions, output_shape', CASES)
def test_chunk_sizes_bound_peak_fusion_memory(label, num_sources, num_z_positions, output_shape,
                                              dtype):
    chunk_sizes = get_chunk_sizes(np.dtype(dtype), list(output_shape),
                                  num_sources=num_sources, num_z_positions=num_z_positions,
                                  fusion_target_bytes=BUDGET)

    assert set(chunk_sizes) == set(output_shape)
    assert all(size >= 1 for size in chunk_sizes.values())

    worst = worst_chunk_fusion_bytes(chunk_sizes, output_shape, num_sources, num_z_positions)
    assert worst <= BUDGET, (f'{label} ({dtype}): chunks {chunk_sizes} need '
                             f'{worst / 1024 ** 2:.0f} MB for one chunk')


def test_sources_spread_over_z_get_single_plane_chunks():
    # a chunk spanning Nz sections costs Nz ** 2 (Nz times the sources, Nz times the voxels),
    # so z must not be widened at all once sources sit at distinct z positions
    chunk_sizes = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'],
                                  num_sources=4733, num_z_positions=72,
                                  fusion_target_bytes=BUDGET)
    assert chunk_sizes['z'] == 1


def test_native_z_stack_still_gets_a_deeper_z_chunk():
    # num_z_positions == 1: every source spans the whole z range, so a deeper z chunk adds
    # voxels but no extra views - the output-byte budget stays the binding constraint there
    chunk_sizes = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'],
                                  num_sources=1, num_z_positions=1,
                                  fusion_target_bytes=BUDGET)
    assert chunk_sizes['z'] > 1
    assert chunk_sizes['y'] == chunk_sizes['x'] == 1024


def test_xy_chunks_shrink_as_sources_multiply():
    def xy(num_sources):
        return get_chunk_sizes(np.dtype('uint16'), ['y', 'x'], num_sources=num_sources,
                               fusion_target_bytes=BUDGET)['x']

    # up to a couple of dozen sources the generous default already fits the budget
    assert xy(1) == xy(10) == 1024
    assert xy(100) < 1024
    assert xy(1000) < xy(100)
    # never below one whole block, however many sources there are
    assert xy(10 ** 6) == 64


def test_a_bigger_budget_buys_bigger_chunks_not_deeper_z():
    """An HPC allocation should spend its headroom on the generous default chunk size (fewer
    chunks, so less graph to build), never on a deeper z chunk - depth is what pulls whole
    extra sections of sources into one chunk."""
    laptop = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4733,
                             num_z_positions=72, fusion_target_bytes=64 * 1024 ** 2)
    hpc = get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4733,
                          num_z_positions=72, fusion_target_bytes=4 * 1024 ** 3)
    assert laptop['z'] == hpc['z'] == 1
    assert laptop['x'] < hpc['x']
    # and never past the generous default, however much headroom there is
    assert hpc['x'] == 1024


def test_default_budget_comes_from_this_machines_allocation():
    from muvis_align.constants import default_fusion_chunk_bytes

    # a real, plausible per-chunk budget - not zero, and not the whole machine
    assert 64 * 1024 ** 2 <= default_fusion_chunk_bytes <= 4 * 1024 ** 3
    # the default path and an explicit equal budget must agree
    assert (get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4733,
                            num_z_positions=72)
            == get_chunk_sizes(np.dtype('uint16'), ['z', 'y', 'x'], num_sources=4733,
                               num_z_positions=72,
                               fusion_target_bytes=default_fusion_chunk_bytes))


# get_export_chunk_sizes() sizes a full-resolution export by the sources each block meets
def make_sources(count, extent):
    """`count` sources, each `extent` pixels square: one lazy sim repeated, as only its shape and spacing are read."""
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
