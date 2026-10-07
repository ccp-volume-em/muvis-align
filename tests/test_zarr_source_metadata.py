"""ZarrImageSource reads its metadata off the store's consolidated metadata (ngff_zarr's parse as the
fallback) without building a msim, and must get the same answers the msim would report."""
import pytest
from multiview_stitcher import msi_utils, ngff_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.ome_zarr_util import (_read_consolidated_ome_zarr_metadata,
                                             _read_ngff_ome_zarr_metadata,
                                             read_ome_zarr_source_metadata)
from muvis_align.image.source_helper import create_image_source
from tests.data_builders import assert_same_metadata, write_ome_zarr

METADATA_KEYS = ('dimension_order', 'shapes', 'dtype', 'nchannels', 'pixel_sizes', 'position')

# (dim_order, shape, pixel size, translation) - the layouts a source can actually arrive in
LAYOUTS = [
    pytest.param('yx', (256, 192), None, None, id='2d'),
    pytest.param('tcyx', (1, 1, 256, 192), None, None, id='2d forced t/c'),
    pytest.param('cyx', (3, 256, 192), None, None, id='multichannel'),
    pytest.param('zyx', (4, 256, 192), None, None, id='3d'),
    pytest.param('czyx', (2, 4, 256, 192), None, None, id='3d multichannel'),
    pytest.param('tczyx', (1, 2, 4, 256, 192), None, None, id='full'),
    pytest.param('zyx', (4, 256, 192), {'z': 7.0, 'y': 0.25, 'x': 0.125}, {'z': -2.0, 'y': 11.5, 'x': 4.25},
                 id='non-default spacing and origin'),
]


def metadata_via_msim(path):
    """What init_metadata used to derive, straight off the eagerly-built msim."""
    msim = ngff_utils.read_msim_from_ome_zarr(path, array_backend='dask',
                                              transform_key='affine_metadata')
    images = [msim[key].ds['image'] for key in msi_utils.get_sorted_scale_keys(msim)]
    image0 = images[0]
    return {'dimension_order': ''.join(image0.dims),
            'shapes': [tuple(image.shape) for image in images],
            'dtype': image0.dtype,
            'pixel_sizes': [si_utils.get_spacing_from_sim(image) for image in images],
            'position': si_utils.get_origin_from_sim(image0),
            'nchannels': image0.sizes.get('c', 1)}


@pytest.mark.parametrize('dim_order, shape, pixel_size, translation', LAYOUTS)
def test_both_read_paths_and_the_source_match_the_msim(tmp_path, dim_order, shape, pixel_size, translation):
    path = write_ome_zarr(tmp_path / 'store.ome.zarr', dim_order, shape, pixel_size, translation, levels=2)
    reference = metadata_via_msim(path)

    consolidated = _read_consolidated_ome_zarr_metadata(path)
    assert consolidated is not None, 'the consolidated fast path should apply to a freshly written v0.5 store'
    assert_same_metadata(consolidated, reference, METADATA_KEYS)
    assert_same_metadata(_read_ngff_ome_zarr_metadata(path), reference, METADATA_KEYS)
    if translation is not None:
        assert consolidated['position'] == pytest.approx(translation)

    source = create_image_source(path)
    source_metadata = {'dimension_order': source.dimension_order, 'shapes': source.shapes, 'dtype': source.dtype,
                       'nchannels': len(source.channels), 'pixel_sizes': source.pixel_sizes}
    assert_same_metadata(source_metadata, reference, METADATA_KEYS[:-1])
    # no msim during init, and it still builds correctly on first access
    assert source._msim is None
    keys = msi_utils.get_sorted_scale_keys(source.msim)
    assert [tuple(source.msim[key].ds['image'].shape) for key in keys] == reference['shapes']


def test_fast_path_declines_a_missing_or_v2_store(tmp_path):
    assert _read_consolidated_ome_zarr_metadata(str(tmp_path / 'nope.ome.zarr')) is None
    (tmp_path / 'v2.ome.zarr').mkdir()
    (tmp_path / 'v2.ome.zarr' / 'zarr.json').write_text('{"zarr_format": 2}')
    assert _read_consolidated_ome_zarr_metadata(str(tmp_path / 'v2.ome.zarr')) is None


def test_real_v04_store_declines_the_fast_path_but_still_matches(tmp_path):
    """v0.4 stores have no consolidated zarr-v3 root, so they take ngff_zarr's own parse - still without a msim."""
    path = write_ome_zarr(tmp_path / 'v04.ome.zarr', levels=2, ome_version='0.4')

    assert _read_consolidated_ome_zarr_metadata(path) is None
    assert_same_metadata(read_ome_zarr_source_metadata(path), metadata_via_msim(path), METADATA_KEYS)

    source = create_image_source(path)
    assert source._msim is None


def test_a_written_store_is_padded_down_to_min_length(tmp_path):
    # already below default_chunk_size (e.g. a scaled convert output), yet it still gets a pyramid
    path = write_ome_zarr(tmp_path / 'small.ome.zarr', 'yx', (512, 768), min_length=128)

    source = create_image_source(path)

    sizes = [max(size for dim, size in zip(source.dimension_order, shape) if dim in 'xyz') for shape in source.shapes]
    assert sizes == [768, 384, 192, 96]


def test_a_version_ome_zarr_py_cannot_write_is_refused_not_relabelled():
    from muvis_align.image.ome_zarr_helper import get_ome_zarr_format

    assert get_ome_zarr_format('0.5')[1].version == '0.5'
    with pytest.raises(ValueError, match='0.6'):
        get_ome_zarr_format('0.6')
