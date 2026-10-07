"""Coarse pyramid levels: one rule (util.calc_pyramid_level_factors) decides them, both for the levels a
reader synthesizes - described at init, built only when the data is asked for - and for the ones the
OME-Zarr export writes, so a written store carries what a reader would otherwise have to invent."""
import dask.array as da
import numpy as np
import pytest
import tifffile
import zarr
from multiview_stitcher import msi_utils

from muvis_align.constants import default_chunk_size
from muvis_align.image.ome_zarr_helper import get_padding_scale_factors
from muvis_align.image.source_helper import create_image_source
from muvis_align.image.util import build_missing_pyramid_levels, calc_pyramid_level_factors, get_level_from_scale
from tests.data_builders import write_ome_zarr, write_tiff_pyramid


@pytest.mark.parametrize('trigger', ['data', 'msim'])
def test_synthesized_levels_are_described_at_init_and_built_only_on_access(tmp_path, trigger):
    path = str(tmp_path / 'flat.tiff')
    tifffile.imwrite(path, np.zeros((4096, 3072), dtype=np.uint16))
    source = create_image_source(path)

    assert source._data_loaded is False
    assert source.shapes == [(4096, 3072), (2048, 1536), (1024, 768)]
    assert [size['x'] for size in source.pixel_sizes] == [1.0, 2.0, 4.0]
    assert [factor['x'] for factor in source.scale_factors] == [1.0, 2.0, 4.0]

    getattr(source, trigger)

    assert source._data_loaded is True
    assert len(msi_utils.get_sorted_scale_keys(source.msim)) == len(source.shapes)
    # repeated access does not keep adding levels
    for _ in range(3):
        assert [tuple(level.shape) for level in source.data] == source.shapes


def test_an_already_pyramidal_source_synthesizes_nothing(tmp_path):
    source = create_image_source(write_tiff_pyramid(tmp_path / 'pyramid.tiff', levels=2))

    assert source._synthesized_level_factors() == []
    assert len(source.shapes) == len(source.data) == 2


def test_a_written_single_resolution_store_carries_its_own_coarse_levels(tmp_path):
    source = create_image_source(write_ome_zarr(tmp_path / 'single.ome.zarr', 'yx', (4096, 4096)))

    assert source._synthesized_level_factors() == []
    assert len(source.data) == len(source.shapes) > 1
    assert max(source.shapes[-1]) <= default_chunk_size
    # the coarse level is reachable: a 4x request gets one rather than a much finer level
    level, residual, _ = get_level_from_scale(source, 4)
    assert level > 0
    assert max(residual.values()) == 1


def test_single_resolution_zarr_synthesizes_levels(tmp_path):
    """A store without a pyramid of its own gets one synthesized, or a coarse preview fuses at full resolution."""
    path = str(tmp_path / 'flat.ome.zarr')
    root = zarr.open_group(path, mode='w', zarr_format=3)
    root.create_array('scale0/image', shape=(4096, 3072), chunks=(512, 512), dtype='uint16')
    root.attrs['ome'] = {
        'version': '0.5',
        'multiscales': [{
            'axes': [{'name': 'y', 'type': 'space', 'unit': 'micrometer'},
                     {'name': 'x', 'type': 'space', 'unit': 'micrometer'}],
            'datasets': [{'path': 'scale0/image', 'coordinateTransformations': [
                {'type': 'scale', 'scale': [1.0, 1.0]},
                {'type': 'translation', 'translation': [0.0, 0.0]}]}],
        }],
    }
    zarr.consolidate_metadata(root.store)

    source = create_image_source(path)

    # the store has exactly one dataset - every further level here is synthesized
    assert len(source.shapes) > 1
    assert [tuple(data.shape) for data in source.data] == [tuple(shape) for shape in source.shapes]
    assert max(source.get_shape(len(source.shapes) - 1)) <= max(source.get_shape(0)) // 2


@pytest.mark.parametrize('sizes, expected', [
    ({'y': 4096, 'x': 4096}, [{'y': 2, 'x': 2}, {'y': 4, 'x': 4}]),
    # cumulative, not per step
    ({'y': 8192, 'x': 8192}, [{'y': 2, 'x': 2}, {'y': 4, 'x': 4}, {'y': 8, 'x': 8}]),
    ({'y': 512, 'x': 512}, []),
    ({}, []),
    # y/x drive the stopping rule; z bottoms out at size 1 rather than going fractional
    ({'z': 3, 'y': 4096, 'x': 4096}, [{'z': 2, 'y': 2, 'x': 2}, {'z': 4, 'y': 4, 'x': 4}]),
])
def test_factors_halve_until_small_enough(sizes, expected):
    assert calc_pyramid_level_factors(sizes) == expected


def test_odd_extents_round_up_like_strided_slicing():
    # data[::2] on 4097 rows yields 2049: the committed shapes must match the arrays built later
    factors = calc_pyramid_level_factors({'y': 4097, 'x': 4097})
    for level_factors in factors:
        strided = np.zeros((4097, 4097))[::level_factors['y'], ::level_factors['x']]
        assert strided.shape == (int(np.ceil(4097 / level_factors['y'])),
                                 int(np.ceil(4097 / level_factors['x'])))


@pytest.mark.parametrize('shape, dim_order', [
    ((2048, 2048), 'yx'), ((4096, 4096), 'yx'), ((6400, 6400), 'yx'), ((1, 3, 4096, 4096), 'tcyx'), ((5,), 't'),
])
def test_export_padding_reaches_the_readers_threshold_over_spatial_dims_only(shape, dim_order):
    spatial_sizes = {dim: size for dim, size in zip(dim_order, shape) if dim in 'xyz'}

    factors = get_padding_scale_factors(shape, dim_order)

    assert factors == calc_pyramid_level_factors(spatial_sizes)
    coarsest = [-(-size // (factors[-1][dim] if factors else 1)) for dim, size in spatial_sizes.items()]
    assert max(coarsest, default=0) <= default_chunk_size


@pytest.mark.parametrize('shape, dim_order', [
    ((4096, 3072), 'yx'),
    ((4097, 4097), 'yx'),
    ((3, 4096, 4096), 'zyx'),
    ((2, 3, 2048, 2048), 'czyx'),
    ((1000, 1000), 'yx'),
])
def test_committed_metadata_matches_what_the_arrays_actually_become(shape, dim_order):
    """Init commits shapes from the factors alone; build_missing_pyramid_levels() later slices the arrays: they must agree."""
    data = da.zeros(shape, dtype=np.uint16)
    pixel_size = {dim: 0.5 for dim in dim_order if dim in 'zyx'}
    datas, pixel_sizes = build_missing_pyramid_levels(data, dim_order, pixel_size)

    sizes = {dim: size for dim, size in zip(dim_order, shape) if dim in 'xyz'}
    factors = calc_pyramid_level_factors(sizes)
    assert len(factors) == len(datas) - 1

    axes = {dim: axis for axis, dim in enumerate(dim_order)}
    for level_factors, expected_data, expected_pixel_size in zip(factors, datas[1:], pixel_sizes[1:]):
        committed_shape = tuple(-(-size // level_factors.get(dim, 1))
                                for dim, size in zip(dim_order, shape))
        assert committed_shape == tuple(expected_data.shape)
        committed_pixel_size = {dim: value * shape[axes[dim]] / committed_shape[axes[dim]]
                                for dim, value in pixel_size.items()}
        assert committed_pixel_size == pytest.approx(expected_pixel_size)
