import numpy as np
import pytest
from pathlib import Path

from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.TiffImageSource import TiffImageSource
from muvis_align.image.ZarrImageSource import ZarrImageSource
from muvis_align.image.source_helper import create_image_source
from muvis_align.image.util import combine_transforms
from muvis_align.util import create_transform, find_all_numbers, split_numeric_dict

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'S000'

TIFF_FILES = [
    '000_000_0.tiff',
    '000_001_0.tiff',
    '001_000_0.tiff',
    '001_001_0.tiff',
]

ZARR_FILES = [
    'S000_000_000.ome.zarr',
    'S000_000_001.ome.zarr',
    'S000_001_000.ome.zarr',
    'S000_001_001.ome.zarr',
]


def _expected_position(filename, formula):
    filename_numeric = find_all_numbers(str(filename))
    context = {'fn': filename_numeric}
    return {dim: eval(expr, context) for dim, expr in formula.items()}


def _sim_at(source, level=0):
    # ImageSource has no get_sim()/get_msim() of its own (removed - unused outside tests);
    # source.msim is the public attribute, this just extracts one scale's sim from it
    return msi_utils.get_sim_from_msim(source.msim, scale=f'scale{level}')


@pytest.mark.parametrize('filename', TIFF_FILES)
def test_tiff_image_source_basic(filename):
    source = TiffImageSource(str(DATA_DIR / filename))

    assert source.dimension_order == 'yx'
    assert source.shape == source.shapes[0]
    assert source.dtype is not None

    assert msi_utils.is_msim(source.msim)

    sim0 = _sim_at(source, 0)
    assert (sim0.sizes['y'], sim0.sizes['x']) == tuple(source.shape)


@pytest.mark.parametrize('filename', TIFF_FILES)
def test_tiff_image_source_metadata_overrides_reach_msim(filename):
    source_metadata = {
        'scale': {'x': 0.004, 'y': 0.004},
        'position': {'z': 'fn[-4]', 'y': 'fn[-3]*24', 'x': 'fn[-2]*24'},
    }
    source = TiffImageSource(str(DATA_DIR / filename), source_metadata=source_metadata)

    expected_position = _expected_position(
        DATA_DIR / filename,
        {'z': 'fn[-4]', 'y': 'fn[-3]*24', 'x': 'fn[-2]*24'},
    )

    pixel_size = source.get_pixel_size()
    assert pixel_size['x'] == pytest.approx(0.004)
    assert pixel_size['y'] == pytest.approx(0.004)

    position = source.get_position()
    assert position['x'] == pytest.approx(expected_position['x'])
    assert position['y'] == pytest.approx(expected_position['y'])
    assert position['z'] == pytest.approx(expected_position['z'])

    # the override must reach the msim's own geometry, not just the plain getters
    sim0 = _sim_at(source, 0)
    spacing = si_utils.get_spacing_from_sim(sim0)
    origin = si_utils.get_origin_from_sim(sim0)
    assert spacing['x'] == pytest.approx(0.004)
    assert spacing['y'] == pytest.approx(0.004)
    assert origin['x'] == pytest.approx(expected_position['x'])
    assert origin['y'] == pytest.approx(expected_position['y'])


@pytest.mark.parametrize('filename', ZARR_FILES)
def test_zarr_image_source_basic(filename):
    source = ZarrImageSource(str(DATA_DIR / filename))

    scale_keys = msi_utils.get_sorted_scale_keys(source.msim)
    assert len(scale_keys) == 3  # real 0/1/2 resolution levels on disk

    sim0 = _sim_at(source, 0)
    sim1 = _sim_at(source, 1)
    assert sim1.sizes['x'] == sim0.sizes['x'] // 2
    assert sim1.sizes['y'] == sim0.sizes['y'] // 2


@pytest.mark.parametrize('filename', ZARR_FILES)
def test_zarr_image_source_never_extracts_sims_or_populates_data(filename):
    """ZarrImageSource works entirely off self.msim - it never calls msi_utils.get_sim_from_msim
    (metadata comes straight from each scale's own 'image' DataArray) and never populates
    self.data (get_level_data() reads the raw array directly off the msim instead)."""
    source = ZarrImageSource(str(DATA_DIR / filename))

    assert source.data == []
    for level in range(len(source.pixel_sizes)):
        level_data = source.get_level_data(level)
        sim_data = _sim_at(source, level).data
        assert level_data.shape == sim_data.shape
        np.testing.assert_array_equal(np.asarray(level_data.compute()), np.asarray(sim_data.compute()))


def test_zarr_image_source_scale_override_reaches_getters_not_native_msim_coords():
    """A source_metadata scale/position override always reaches get_pixel_size()/get_position()
    (what the real registration pipeline actually reads - MVSRegistration never reads geometry
    off a source's own msim coords). For ZarrImageSource specifically, the msim itself keeps
    read_msim_from_ome_zarr's native coordinates untouched - trusted as correct since they come
    from the file's own calibration - rather than being restamped to match the override."""
    native_pixel_size = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0])).get_pixel_size()

    source_metadata = {'scale': {'x': 0.01, 'y': 0.01}, 'position': {'x': 5, 'y': 7}}
    source = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0]), source_metadata=source_metadata)

    # the override is reflected in the plain getters
    assert source.get_pixel_size()['x'] == pytest.approx(0.01)
    assert source.get_position()['x'] == pytest.approx(5)

    # ...but not in the msim's own native coordinates
    sim0 = _sim_at(source, 0)
    sim1 = _sim_at(source, 1)
    spacing0 = si_utils.get_spacing_from_sim(sim0)
    spacing1 = si_utils.get_spacing_from_sim(sim1)
    assert spacing0['x'] == pytest.approx(native_pixel_size['x'])
    assert spacing1['x'] == pytest.approx(native_pixel_size['x'] * 2)  # level 1 is half the resolution
    origin0 = si_utils.get_origin_from_sim(sim0)
    assert origin0['x'] == pytest.approx(0)


def test_zarr_image_source_restamped_affine_matches_native_shape():
    """ZarrImageSource keeps the msim read_msim_from_ome_zarr() builds natively (2D data still
    gets a 4x4 identity transform there, since z is a real if trivial spatial dim in NGFF) and
    replaces just that transform with source.transform, in place. The replacement must end up
    the same shape/convention si_utils.get_sim_from_array uses (2D x_in/x_out, no 't' dim) - not
    left as a stale 4x4 with NaN from a shape mismatch against the dropped native transform."""
    source = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0]), source_metadata={'rotation': 30})

    own = param_utils.invert_coordinate_order(
        create_transform(source.position, source.rotation, matrix_size=3))

    scale_keys = msi_utils.get_sorted_scale_keys(source.msim)
    assert len(scale_keys) == 3
    for scale_key in scale_keys:
        sim = msi_utils.get_sim_from_msim(source.msim, scale=scale_key)
        affine = si_utils.get_affine_from_sim(sim, source.transform_key)
        assert affine.shape == (3, 3)
        assert not np.isnan(affine.values).any()
        np.testing.assert_allclose(affine.values, own)


def test_create_image_source_dispatches_on_extension():
    tiff_source = create_image_source(str(DATA_DIR / TIFF_FILES[0]))
    zarr_source = create_image_source(str(DATA_DIR / ZARR_FILES[0]))
    assert isinstance(tiff_source, TiffImageSource)
    assert isinstance(zarr_source, ZarrImageSource)


def test_extra_metadata_composes_with_own_rotation_transform():
    extra_transform = np.eye(3)
    extra_transform[0, 2] = 100  # translate x by 100

    source = TiffImageSource(
        str(DATA_DIR / TIFF_FILES[0]),
        source_metadata={'rotation': 15},
        extra_metadata={'t1': extra_transform.tolist()},
        file_label='t1',
    )

    own = param_utils.invert_coordinate_order(
        create_transform(source.position, source.rotation, matrix_size=3))
    expected = np.array(combine_transforms([own, extra_transform]))

    assert source.transform is not None
    np.testing.assert_allclose(source.transform, expected)

    # reads the transform straight off the msim - no need to extract a sim just for this
    affine = msi_utils.get_transform_from_msim(source.msim, source.transform_key)
    np.testing.assert_allclose(np.array(affine), expected)
