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


@pytest.mark.parametrize('filename', TIFF_FILES)
def test_tiff_image_source_basic(filename):
    source = TiffImageSource(str(DATA_DIR / filename))

    assert source.dimension_order == 'yx'
    assert source.shape == source.shapes[0]
    assert source.dtype is not None

    msim = source.get_msim()
    assert msi_utils.is_msim(msim)

    sim0 = source.get_sim(0)
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
    sim0 = source.get_sim(0)
    spacing = si_utils.get_spacing_from_sim(sim0)
    origin = si_utils.get_origin_from_sim(sim0)
    assert spacing['x'] == pytest.approx(0.004)
    assert spacing['y'] == pytest.approx(0.004)
    assert origin['x'] == pytest.approx(expected_position['x'])
    assert origin['y'] == pytest.approx(expected_position['y'])


@pytest.mark.parametrize('filename', ZARR_FILES)
def test_zarr_image_source_basic(filename):
    source = ZarrImageSource(str(DATA_DIR / filename))

    scale_keys = msi_utils.get_sorted_scale_keys(source.get_msim())
    assert len(scale_keys) == 3  # real 0/1/2 resolution levels on disk

    sim0 = source.get_sim(0)
    sim1 = source.get_sim(1)
    assert sim1.sizes['x'] == sim0.sizes['x'] // 2
    assert sim1.sizes['y'] == sim0.sizes['y'] // 2


def test_zarr_image_source_scale_override_propagates_across_levels():
    source_metadata = {'scale': {'x': 0.01, 'y': 0.01}, 'position': {'x': 5, 'y': 7}}
    source = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0]), source_metadata=source_metadata)

    sim0 = source.get_sim(0)
    sim1 = source.get_sim(1)

    spacing0 = si_utils.get_spacing_from_sim(sim0)
    spacing1 = si_utils.get_spacing_from_sim(sim1)
    assert spacing0['x'] == pytest.approx(0.01)
    assert spacing1['x'] == pytest.approx(0.02)  # level 1 is half the resolution of level 0

    origin0 = si_utils.get_origin_from_sim(sim0)
    origin1 = si_utils.get_origin_from_sim(sim1)
    assert origin0['x'] == pytest.approx(5)
    assert origin1['x'] == pytest.approx(5)


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

    sim0 = source.get_sim(0)
    affine = si_utils.get_affine_from_sim(sim0, source.transform_key)
    np.testing.assert_allclose(np.array(affine), expected)
