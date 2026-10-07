import logging
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.source_helper import create_image_source
from muvis_align.image.util import (build_source_msim, build_source_redimensioned_msim, get_contrast_limits,
                                    get_level_from_scale, make_msims_3d, map_msim_levels, msim_is_already_3d,
                                    promote_sim_to_3d, rechunk_if_monolithic, select_msim_subpyramid_at_scale,
                                    unify_msim_channels)
from tests.data_builders import DATA_DIR, TIFF_FILES, make_msim, make_sim


@pytest.mark.parametrize('chunks, chunk_size, expected', [
    ((2000, 2000), 1024, (1024, 976)),
    ((500, 500), 1024, (500, 500, 500, 500)),
    ((2000, 2000), None, (2000,)),
])
def test_rechunk_if_monolithic_splits_only_single_chunk_data(chunks, chunk_size, expected):
    image = xr.DataArray(da.zeros((2000, 2000), dtype=np.uint8, chunks=chunks), dims=['y', 'x'])

    rechunked = rechunk_if_monolithic(image, chunk_size)

    assert rechunked.chunksizes['y'] == rechunked.chunksizes['x'] == expected


def test_build_source_redimensioned_msim_rechunks_monolithic_levels():
    """A single-chunk source is split once, at msim creation, not every time a sim is taken from it."""
    msim = make_msim(da.zeros((2000, 2000), dtype=np.uint8, chunks=(2000, 2000)))
    source = SimpleNamespace(msim=msim, dimension_order='yx', get_channels=lambda: [])

    redimensioned = build_source_redimensioned_msim(source, 'yx', chunk_size=1024)

    image0 = msi_utils.get_sim_from_msim(redimensioned, scale='scale0')
    assert image0.chunksizes['y'] == (1024, 976)
    assert image0.chunksizes['x'] == (1024, 976)


def make_pyramid_source(nlevels=4, size=1024, pixel_size=0.5):
    """A source with a real pyramid - enough of one for the msim build to walk its levels."""
    shapes, pixel_sizes, data = [], [], []
    for level in range(nlevels):
        side = size // (2 ** level)
        shapes.append((side, side))
        pixel_sizes.append({'y': pixel_size * 2 ** level, 'x': pixel_size * 2 ** level})
        data.append(da.zeros((side, side), chunks=(256, 256), dtype=np.uint16))
    # c_coords as ImageSource._build_msim passes them, so this msim is the one a real source would have
    sims = [make_sim(array, 'yx', scale=pixel_sizes[level], translation={'y': 11.0, 'x': 23.0},
                     c_coords=['channel 0'])
            for level, array in enumerate(data)]
    source = SimpleNamespace(
        msim=msi_utils.get_msim_from_sims(sims), dimension_order='yx', shapes=shapes,
        shape=shapes[0], pixel_sizes=pixel_sizes, data=data,
        get_channels=lambda: [{'label': 'channel 0'}],
        get_pixel_size=lambda: pixel_sizes[0],
        position={'y': 11.0, 'x': 23.0},
        scale_factors=[{'y': shapes[0][0] / shape[0], 'x': shapes[0][1] / shape[1]} for shape in shapes],
        _redimensioned_msims={})
    source.get_msim = lambda output_order, from_level=0, ends_only=False: build_source_redimensioned_msim(
        source, output_order, from_level=from_level, ends_only=ends_only)
    return source


def describe_levels(msim):
    """Each level's sizes, spacing and origin - the geometry, independent of level naming."""
    described = []
    for scale_key in msi_utils.get_sorted_scale_keys(msim):
        sim = msim[scale_key].ds['image']
        described.append((dict(sim.sizes),
                          {dim: round(float(value), 6)
                           for dim, value in si_utils.get_spacing_from_sim(sim).items()},
                          {dim: round(float(value), 6)
                           for dim, value in si_utils.get_origin_from_sim(sim).items()}))
    return described


@pytest.mark.parametrize('from_level', [0, 1, 2, 3])
def test_build_source_msim_from_level_equals_the_tail_of_the_full_pyramid(from_level):
    """Starting coarser may only skip levels: level n keeps level n's pixel size, not scale0's."""
    full = build_source_msim(make_pyramid_source(), 'yx', {'y': 11.0, 'x': 23.0}, None,
                             'source_metadata')
    truncated = build_source_msim(make_pyramid_source(), 'yx', {'y': 11.0, 'x': 23.0}, None,
                                  'source_metadata', from_level=from_level)

    assert describe_levels(truncated) == describe_levels(full)[from_level:]
    assert msi_utils.get_sorted_scale_keys(truncated)[0] == 'scale0'


@pytest.mark.parametrize('scale', [1, 2, 4, 8])
def test_building_at_a_scale_then_selecting_it_matches_building_everything_first(scale):
    """Pre-processing builds from the level a scale needs: the result must be what a full build sliced down gives."""
    geometry = ('yx', {'y': 11.0, 'x': 23.0}, None, 'source_metadata')

    full_source = make_pyramid_source()
    full = select_msim_subpyramid_at_scale(
        [build_source_msim(full_source, *geometry)], [full_source], scale)[0]

    scaled_source = make_pyramid_source()
    from_level = get_level_from_scale(scaled_source, scale)[0]
    scaled = select_msim_subpyramid_at_scale(
        [build_source_msim(scaled_source, *geometry, from_level=from_level)], [scaled_source],
        scale)[0]

    assert describe_levels(scaled) == describe_levels(full)


@pytest.mark.parametrize('output_order, z_scale', [('yx', None), ('zyx', 2.5), ('cyx', None)])
def test_level_images_from_raw_arrays_match_the_ones_from_the_source_msim(output_order, z_scale):
    """A source without raw arrays (a natively-read OME-Zarr) comes through its msim: both paths must agree."""
    from_arrays = make_pyramid_source()
    from_msim = make_pyramid_source()
    from_msim.data = []

    built_from_arrays = build_source_msim(from_arrays, output_order, {'y': 11.0, 'x': 23.0}, None,
                                          'source_metadata', z_scale=z_scale)
    built_from_msim = build_source_msim(from_msim, output_order, {'y': 11.0, 'x': 23.0}, None,
                                        'source_metadata', z_scale=z_scale)

    assert describe_levels(built_from_arrays) == describe_levels(built_from_msim)
    for scale_key in msi_utils.get_sorted_scale_keys(built_from_arrays):
        from_array_sim = msi_utils.get_sim_from_msim(built_from_arrays, scale=scale_key)
        from_msim_sim = msi_utils.get_sim_from_msim(built_from_msim, scale=scale_key)
        assert from_array_sim.dims == from_msim_sim.dims
        assert from_array_sim.chunksizes == from_msim_sim.chunksizes
        assert list(from_array_sim.coords['c'].values) == list(from_msim_sim.coords['c'].values)


def test_building_levels_from_raw_arrays_never_builds_the_source_msim():
    source = make_pyramid_source()
    # reaching for source.msim would raise AttributeError
    del source.msim

    build_source_msim(source, 'yx', {'y': 11.0, 'x': 23.0}, None, 'source_metadata')


@pytest.mark.parametrize('from_level, nlevels', [(0, 4), (1, 4), (2, 4), (0, 2), (0, 1)])
def test_ends_only_keeps_the_first_and_coarsest_levels_unchanged(from_level, nlevels):
    """Pre-processing builds only the finest (registration) and coarsest (preview cap) levels, as the full pyramid has them."""
    geometry = ('yx', {'y': 11.0, 'x': 23.0}, None, 'source_metadata')
    full = describe_levels(build_source_msim(make_pyramid_source(nlevels), *geometry, from_level=from_level))
    ends = build_source_msim(make_pyramid_source(nlevels), *geometry, from_level=from_level, ends_only=True)

    expected = full[:1] + full[-1:] if len(full) > 2 else full
    assert describe_levels(ends) == expected
    assert msi_utils.get_sorted_scale_keys(ends) == [f'scale{index}' for index in range(len(expected))]


def channel_msim(labels, offset=0.0):
    return make_msim(np.ones((len(labels), 16, 16), dtype=np.uint8), 'cyx', scale_factors=[2],
                     translation={'y': 0.0, 'x': offset}, c_coords=list(labels))


def msim_channel_labels(msim):
    return [list(msim[scale_key].ds['image'].coords['c'].values) for scale_key in msi_utils.get_sorted_scale_keys(msim)]


def test_a_single_channel_named_otherwise_takes_the_common_label_on_every_level():
    msims = [channel_msim(['#0']), channel_msim(['channel 0'], offset=16.0)]

    unified = unify_msim_channels(msims)

    assert unified[0] is msims[0]
    assert msim_channel_labels(unified[1]) == [['#0'], ['#0']]
    xr.testing.assert_identical(unified[1]['scale1'].ds['image'].drop_vars('c'),
                                msims[1]['scale1'].ds['image'].drop_vars('c'))


def test_matching_channels_in_another_order_are_left_alone():
    msims = [channel_msim(['a', 'b']), channel_msim(['b', 'a'])]

    unified = unify_msim_channels(msims)

    assert all(new is old for new, old in zip(unified, msims))


def test_differing_multichannel_labels_are_an_error():
    with pytest.raises(ValueError, match='different channels'):
        unify_msim_channels([channel_msim(['a', 'b']), channel_msim(['a', 'c'])])


def test_sources_with_differing_channel_names_fuse_once_unified():
    from multiview_stitcher import fusion

    msims = unify_msim_channels([channel_msim(['#0']), channel_msim(['channel 0'], offset=16.0)])

    fused = msi_utils.get_sim_from_msim(fusion.fuse(msims, transform_key='source_metadata'), scale='scale0')

    assert list(fused.coords['c'].values) == ['#0']
    assert int(fused.max().compute()) == 1


# --- which pyramid level a target scale selects ---

def scaled_source(factors, reduce_z=False):
    """A source whose pyramid reduces x/y by `factors` - and z too, for a real z-stack."""
    return SimpleNamespace(
        scale_factors=[{'z': factor if reduce_z else 1.0, 'y': factor, 'x': factor} for factor in factors],
        get_pixel_size=lambda: {'z': 1.0, 'y': 0.1, 'x': 0.1},
    )


@pytest.mark.parametrize('factors, reduce_z, target, level', [
    ([1, 2, 4, 8, 16], False, 1, 0),
    ([1, 2, 4, 8, 16], False, 4, 2),
    ([1, 2, 4, 8, 16], False, 16, 4),
    # never coarser than asked; a size-1 z keeps factor 1 at every level and must not decide
    ([1, 2, 4, 8, 16], False, 6, 2),
    ([1, 2, 4, 8, 16], False, 12, 3),
    # a text field hands a factor over as text
    ([1, 2, 4, 8, 16], False, '6', 2),
    ([1, 2, 4, 8, 16], False, '16', 4),
    # a z-stack reduces z too, so z counts: 3x cannot take the 4x level
    ([1, 2, 4], True, 3, 1),
    ([1, 2, 4], True, 4, 2),
    ([1], False, 16, 0),
])
def test_level_from_scale_selects_the_coarsest_level_still_fine_enough(factors, reduce_z, target, level):
    assert get_level_from_scale(scaled_source(factors, reduce_z), target)[0] == level


def test_a_target_beyond_the_pyramid_selects_its_coarsest_level_and_reports_the_shortfall():
    level, residual, _ = get_level_from_scale(scaled_source([1, 2, 4]), 32)

    assert level == 2
    assert residual['x'] == 8


def test_a_pixel_size_selects_the_level_nearest_it_without_going_coarser():
    """0.1um pixels: 0.4um is the 4x level exactly, 0.6um the 4x (never the 8x), 1600nm the 16x."""
    source = scaled_source([1, 2, 4, 8, 16])

    assert get_level_from_scale(source, '0.6 um')[0] == 2
    assert get_level_from_scale(source, '1600nm')[0] == 4
    level, _, pixel_size = get_level_from_scale(source, '0.4um')
    assert level == 2
    assert pixel_size['x'] == pixel_size['y'] == 0.4


@pytest.mark.parametrize('dims, factors, logged', [
    ('yx', [1, 2], True),
    ('yx', [1, 2, 4, 8, 16], False),
    # one level short is not worth a warning
    ('yx', [1, 2, 4, 8], False),
    # a size-1 z is the same at every level, so its full residual is no shortfall
    ('zyx', [1, 2, 4, 8], False),
])
def test_a_preview_scale_shortfall_is_logged(caplog, dims, factors, logged):
    shape = (1, 64, 64) if 'z' in dims else (64, 64)
    msim = make_msim(np.zeros(shape, dtype=np.uint16), dims)
    source = SimpleNamespace(
        scale_factors=[{**({'z': 1.0} if 'z' in dims else {}), 'y': float(factor), 'x': float(factor)}
                       for factor in factors],
        dimension_order=dims,
        get_shape=lambda level=0: shape,
        get_pixel_size=lambda: {dim: 1.0 for dim in dims})

    with caplog.at_level(logging.WARNING):
        select_msim_subpyramid_at_scale([msim], [source], 16)

    assert ('Preview scale 16 not reachable' in caplog.text) == logged
    assert ('8x finer' in caplog.text) == logged


# --- display contrast limits, which must never be the step that blocks first paint ---

def test_contrast_limits_widen_a_flat_range_so_napari_gets_a_usable_span():
    low, high = get_contrast_limits(make_msim(np.zeros((4, 4), dtype=np.uint16)))
    assert low < high


@pytest.mark.parametrize('dtype, expected', [(np.uint16, [0, np.iinfo(np.uint16).max]), (np.float32, [0.0, 1.0])])
def test_cheap_contrast_limits_are_the_dtype_range_without_computing(dtype, expected):
    assert get_contrast_limits(make_msim(np.zeros((8, 8), dtype=dtype)), cheap=True) == expected


def test_contrast_limits_fall_back_to_the_dtype_range_when_the_coarsest_level_is_expensive():
    # many small chunks stand in for a level fused from thousands of sources: its graph is large
    sim = make_sim(np.full((64, 64), 700, dtype=np.uint16)).chunk({'y': 1, 'x': 1})
    msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
    assert len(msim['scale0'].ds['image'].data.dask) > 64

    assert get_contrast_limits(msim, max_tasks=64) == [0, np.iinfo(np.uint16).max]
    # and the real range under the threshold
    assert get_contrast_limits(msim, max_tasks=10 ** 6) == [700.0, 701.0]


# --- promoting 2D slices to a size-1 z: what lets napari step through a stack, done once ---

KEY = 'source_metadata'


def flat_msims_and_positions(count=3):
    sources = [create_image_source(str(DATA_DIR / name)) for name in TIFF_FILES[:count]]
    positions = [{'z': float(index), 'y': 10.0 * index, 'x': 20.0 * index} for index in range(count)]
    flat = [build_source_msim(source, 'tcyx', position, None, KEY) for source, position in zip(sources, positions)]
    return flat, positions


def test_promotion_adds_a_z_dim_at_each_slices_height_as_promoting_each_level_would():
    flat, positions = flat_msims_and_positions()
    # one with a second level, as pre-processing's scaled pyramids have
    flat[0] = msi_utils.get_msim_from_sim(msi_utils.get_sim_from_msim(flat[0]), scale_factors=[{'y': 2, 'x': 2}])
    assert all('z' not in msim['scale0'].ds['image'].dims for msim in flat)

    promoted = make_msims_3d(flat, z_scale=1.0, positions=positions)

    for msim, original, position in zip(promoted, flat, positions):
        expected = map_msim_levels(original, lambda sim, scale_key: promote_sim_to_3d(sim, position['z']))
        assert msim.identical(expected)
        for scale_key in msi_utils.get_sorted_scale_keys(msim):
            image = msim[scale_key].ds['image']
            assert image.sizes['z'] == 1
            assert float(image.coords['z'].values[0]) == pytest.approx(position['z'])


def test_a_promoted_msim_is_recognised_and_left_alone():
    flat, positions = flat_msims_and_positions()
    assert not any(msim_is_already_3d(msim) for msim in flat)

    promoted = make_msims_3d(flat, z_scale=1.0, positions=positions)
    assert all(msim_is_already_3d(msim) for msim in promoted)

    again = make_msims_3d(promoted, z_scale=1.0, positions=positions)

    # returned untouched, not rebuilt into an equal-but-new tree
    assert all(after is before for before, after in zip(promoted, again))


def test_a_natively_3d_source_is_left_alone_too():
    msim = make_msim(np.zeros((4, 32, 32), dtype=np.uint16), 'zyx', scale={'z': 2.0, 'y': 1.0, 'x': 1.0})
    assert msim_is_already_3d(msim)

    result = make_msims_3d([msim], z_scale=1.0, positions=[{'z': 0.0}])

    assert result[0] is msim
    assert result[0]['scale0'].ds['image'].sizes['z'] == 4
