import numpy as np
import pytest
from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.TiffImageSource import TiffImageSource
from muvis_align.image.ZarrImageSource import ZarrImageSource
from muvis_align.image.source_helper import create_image_source
from muvis_align.image.util import combine_transforms, get_data_mapping
from muvis_align.util import create_transform, find_all_numbers
from tests.data_builders import DATA_DIR, TIFF_FILES, ZARR_FILES


def _sim_at(source, level=0):
    # one scale's sim straight off source.msim, the native-dimension_order msim
    return msi_utils.get_sim_from_msim(source.msim, scale=f'scale{level}')


def test_tiff_image_source_labels_a_forced_c_dim_with_its_channel_name():
    """registration's 'channel' param selects by label (.sel(c=...)), so the size-1 c forced onto a
    single-channel source must carry its channel name, not a plain index."""
    source = TiffImageSource(str(DATA_DIR / TIFF_FILES[0]))
    assert source.dimension_order == 'yx'
    assert source.shape == source.shapes[0]

    channel_label = source.get_channels()[0]['label']
    sim0 = _sim_at(source, 0)
    assert list(sim0.coords['c'].values) == [channel_label]
    assert (sim0.sizes['y'], sim0.sizes['x']) == tuple(source.shape)

    # the exact call registration/preview_registration makes to select a channel by name
    selected = msi_utils.multiscale_sel_coords(source.get_msim('yx'), {'c': channel_label})
    assert 'c' not in msi_utils.get_sim_from_msim(selected, scale='scale0').dims


def test_tiff_image_source_metadata_overrides_reach_msim():
    # a file whose name numbers are not all zero, so the position formula is exercised
    filename = DATA_DIR / TIFF_FILES[3]
    formula = {'z': 'fn[-4]', 'y': 'fn[-3]*24', 'x': 'fn[-2]*24'}
    source = TiffImageSource(str(filename), source_metadata={'scale': {'x': 0.004, 'y': 0.004}, 'position': formula})
    filename_numbers = find_all_numbers(str(filename))
    expected_position = {dim: eval(expression, {'fn': filename_numbers}) for dim, expression in formula.items()}

    assert source.get_pixel_size() == pytest.approx({'y': 0.004, 'x': 0.004})
    assert source.get_position() == pytest.approx(expected_position)

    # the override must reach the msim's own geometry, not just the plain getters
    sim0 = _sim_at(source, 0)
    assert si_utils.get_spacing_from_sim(sim0) == pytest.approx({'y': 0.004, 'x': 0.004})
    assert si_utils.get_origin_from_sim(sim0) == pytest.approx({dim: expected_position[dim] for dim in 'yx'})


def test_zarr_image_source_reads_its_levels_straight_off_the_store():
    """self.data holds one raw dask array per level, opened off the store, which is what lets the base
    class synthesize coarse levels for a single-resolution store."""
    source = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0]))

    # real 0/1/2 resolution levels on disk
    assert len(source.data) == len(source.shapes) == 3
    assert [tuple(data.shape) for data in source.data] == [tuple(shape) for shape in source.shapes]
    assert _sim_at(source, 1).sizes['x'] == _sim_at(source, 0).sizes['x'] // 2
    for level in range(len(source.pixel_sizes)):
        level_data = source.get_level_data(level)
        sim_data = _sim_at(source, level).data
        assert level_data.shape == sim_data.shape
        np.testing.assert_array_equal(np.asarray(level_data.compute()), np.asarray(sim_data.compute()))


def test_zarr_image_source_scale_override_reaches_getters_and_msim_coords():
    native_pixel_size = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0])).get_pixel_size()
    assert native_pixel_size['x'] != pytest.approx(0.01)

    source_metadata = {'scale': {'x': 0.01, 'y': 0.01}, 'position': {'x': 5, 'y': 7}}
    source = ZarrImageSource(str(DATA_DIR / ZARR_FILES[0]), source_metadata=source_metadata)

    assert source.get_pixel_size()['x'] == pytest.approx(0.01)
    assert source.get_position()['x'] == pytest.approx(5)
    # ...and in the msim's own coordinates, level by level
    assert si_utils.get_spacing_from_sim(_sim_at(source, 0))['x'] == pytest.approx(0.01)
    assert si_utils.get_spacing_from_sim(_sim_at(source, 1))['x'] == pytest.approx(0.02)
    assert si_utils.get_origin_from_sim(_sim_at(source, 0))['x'] == pytest.approx(5)


TRANSLATE_X_100 = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize('filename, extra_transform', [
    (TIFF_FILES[0], TRANSLATE_X_100),
    (ZARR_FILES[0], None),
], ids=['tiff with an extra transform', 'zarr'])
def test_the_own_rotation_and_any_extra_transform_reach_every_msim_level(filename, extra_transform):
    extra = {'extra_metadata': {'t1': extra_transform.tolist()}, 'file_label': 't1'} if extra_transform is not None else {}
    source = create_image_source(str(DATA_DIR / filename), source_metadata={'rotation': 15}, **extra)

    own = param_utils.invert_coordinate_order(create_transform(source.position, source.rotation, matrix_size=3))
    expected = np.array(combine_transforms([own, extra_transform]) if extra_transform is not None else own)

    np.testing.assert_allclose(source.transform, expected)
    for scale_key in msi_utils.get_sorted_scale_keys(source.msim):
        affine = si_utils.get_affine_from_sim(msi_utils.get_sim_from_msim(source.msim, scale=scale_key),
                                              source.transform_key)
        # 2D x_in/x_out like get_sim_from_array makes, not a stale 4x4 from the store
        assert affine.shape == (3, 3)
        np.testing.assert_allclose(affine.values, expected)



@pytest.mark.parametrize('rotation, transform_rotation, expected', [
    (None, 2, 2),
    (-30, 2, -28),
    (-30, 0, -30),
    (None, 0, None),
], ids=['transform only', 'source and transform', 'source only', 'neither'])
def test_data_mapping_adds_the_transforms_rotation_to_the_sources(rotation, transform_rotation, expected):
    sim = si_utils.get_sim_from_array(np.zeros((4, 4)), dims=['y', 'x'])
    transform = param_utils.affine_to_xaffine(
        param_utils.invert_coordinate_order(create_transform(None, transform_rotation)))

    _, mapped_rotation = get_data_mapping(sim, transform=transform, rotation=rotation)

    assert mapped_rotation == (pytest.approx(expected) if expected is not None else None)

@pytest.mark.parametrize('rotation', ['invert', 'source invert'])
def test_invert_alone_inverts_the_sources_own_values(rotation):
    source = create_image_source(str(DATA_DIR / TIFF_FILES[0]))
    source.position = {'x': 5.0, 'y': 7.0}
    source.rotation = 12.0

    # z is not in the source, so there is nothing to invert
    source.fix_metadata({'position': {'x': rotation, 'z': rotation}, 'rotation': rotation})

    assert source.position == {'x': -5.0, 'y': 7.0}
    assert source.rotation == -12.0


def test_get_msim_caches_by_output_order_and_start_level():
    source = TiffImageSource(str(DATA_DIR / TIFF_FILES[0]))

    msim_1 = source.get_msim('yx')
    msim_2 = source.get_msim('yx')

    assert msim_1 is msim_2
    assert list(source._redimensioned_msims.keys()) == [('yx', 0, False)]

    # a coarser start is a different pyramid, so it gets its own entry
    source.get_msim('yx', from_level=1)
    assert sorted(source._redimensioned_msims) == [('yx', 0, False), ('yx', 1, False)]

    image0 = msi_utils.get_sim_from_msim(msim_1, scale='scale0')
    assert image0.dims == ('t', 'c', 'y', 'x')
    assert (image0.sizes['y'], image0.sizes['x']) == tuple(source.shape)


def test_create_image_source_dispatches_on_extension():
    assert isinstance(create_image_source(str(DATA_DIR / TIFF_FILES[0])), TiffImageSource)
    assert isinstance(create_image_source(str(DATA_DIR / ZARR_FILES[0])), ZarrImageSource)
