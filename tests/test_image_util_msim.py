from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from multiview_stitcher import msi_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.util import (build_source_msim, build_source_redimensioned_msim,
                                    get_level_from_scale, rechunk_if_monolithic,
                                    select_msim_subpyramid_at_scale)


def test_rechunk_if_monolithic_splits_single_chunk_data():
    data = da.from_array(np.zeros((2000, 2000), dtype=np.uint8), chunks=(2000, 2000))
    image = xr.DataArray(data, dims=['y', 'x'])

    rechunked = rechunk_if_monolithic(image, 1024)

    assert rechunked.chunksizes['y'] == (1024, 976)
    assert rechunked.chunksizes['x'] == (1024, 976)


def test_rechunk_if_monolithic_leaves_already_chunked_data_alone():
    data = da.from_array(np.zeros((2000, 2000), dtype=np.uint8), chunks=(500, 500))
    image = xr.DataArray(data, dims=['y', 'x'])

    rechunked = rechunk_if_monolithic(image, 1024)

    assert rechunked.chunksizes == image.chunksizes


def test_rechunk_if_monolithic_noop_when_chunk_size_falsy():
    data = da.from_array(np.zeros((2000, 2000), dtype=np.uint8), chunks=(2000, 2000))
    image = xr.DataArray(data, dims=['y', 'x'])

    rechunked = rechunk_if_monolithic(image, None)

    assert rechunked.chunksizes == image.chunksizes


def test_build_source_redimensioned_msim_rechunks_monolithic_levels():
    """A badly-chunked source (e.g. a single-chunk TIFF) must still get split into smaller dask
    chunks - here, once, at msim-creation time, rather than downstream every time a sim is
    extracted from it."""
    data = da.from_array(np.zeros((2000, 2000), dtype=np.uint8), chunks=(2000, 2000))
    sim = si_utils.get_sim_from_array(data, dims=['y', 'x'], transform_key='source_metadata')
    msim = msi_utils.get_msim_from_sims([sim])
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
    # c_coords as ImageSource._build_msim passes them, so the msim this fake offers is the one
    # a real source would have - an unlabelled 'c' would differ from the raw-array path for a
    # reason no real source has
    sims = [si_utils.get_sim_from_array(array, dims=['y', 'x'], scale=pixel_sizes[level],
                                        translation={'y': 11.0, 'x': 23.0},
                                        transform_key='source_metadata',
                                        c_coords=['channel 0'])
            for level, array in enumerate(data)]
    source = SimpleNamespace(
        msim=msi_utils.get_msim_from_sims(sims), dimension_order='yx', shapes=shapes,
        shape=shapes[0], pixel_sizes=pixel_sizes, data=data,
        get_channels=lambda: [{'label': 'channel 0'}],
        get_pixel_size=lambda: pixel_sizes[0],
        position={'y': 11.0, 'x': 23.0},
        scale_factors=[{'y': shapes[0][0] / s[0], 'x': shapes[0][1] / s[1]} for s in shapes],
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
    """Starting coarser may only skip levels, never change the ones it keeps: the pixel size
    of level n has to stay level n's, not the truncated pyramid's own scale0.
    """
    full = build_source_msim(make_pyramid_source(), 'yx', {'y': 11.0, 'x': 23.0}, None,
                             'source_metadata')
    truncated = build_source_msim(make_pyramid_source(), 'yx', {'y': 11.0, 'x': 23.0}, None,
                                  'source_metadata', from_level=from_level)

    assert describe_levels(truncated) == describe_levels(full)[from_level:]
    assert msi_utils.get_sorted_scale_keys(truncated)[0] == 'scale0'


@pytest.mark.parametrize('scale', [1, 2, 4, 8])
def test_building_at_a_scale_then_selecting_it_matches_building_everything_first(scale):
    """The optimisation pre-processing relies on: a pyramid built from the level a scale needs,
    and then selected at that scale, is the one a full build would have been sliced down to -
    sliced once, not twice.
    """
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
    """The two ways a level's image can be obtained must be indistinguishable.

    Building from the source's own arrays skips a get_sim_from_array and a DataTree per source,
    but a source with no raw arrays (a natively-read OME-Zarr) still comes through its msim, so
    both paths stay live and have to describe the same image.
    """
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
    """The point of the raw-array path: a source whose msim would cost a get_sim_from_array per
    level must not have it built behind the optimisation's back.
    """
    source = make_pyramid_source()
    del source.msim

    build_source_msim(source, 'yx', {'y': 11.0, 'x': 23.0}, None, 'source_metadata')


@pytest.mark.parametrize('from_level, nlevels', [(0, 4), (1, 4), (2, 4), (0, 2), (0, 1)])
def test_ends_only_keeps_the_first_and_coarsest_levels_unchanged(from_level, nlevels):
    """Pre-processing builds only what registration (the finest) and the preview cap (the
    coarsest) read: those two levels exactly as the full pyramid has them, in order."""
    geometry = ('yx', {'y': 11.0, 'x': 23.0}, None, 'source_metadata')
    full = describe_levels(build_source_msim(make_pyramid_source(nlevels), *geometry, from_level=from_level))
    ends = build_source_msim(make_pyramid_source(nlevels), *geometry, from_level=from_level, ends_only=True)

    expected = full[:1] + full[-1:] if len(full) > 2 else full
    assert describe_levels(ends) == expected
    assert msi_utils.get_sorted_scale_keys(ends) == [f'scale{index}' for index in range(len(expected))]
