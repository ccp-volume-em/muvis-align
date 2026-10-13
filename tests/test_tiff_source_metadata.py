"""TiffImageSource reads its metadata off tifffile without building arrays, and must get the same
answers as the ngff_zarr path it replaced - asserted for every file shape a source can arrive in."""
import numpy as np
import pytest
import tifffile

from muvis_align.image.ome_tiff_helper import (extract_ome_image_metadata, extract_ome_translation_from_xml,
                                               read_tiff_source_metadata)
from muvis_align.image.source_helper import create_image_source
from muvis_align.image.TiffImageSource import TiffImageSource
from tests.data_builders import assert_same_metadata, write_tiff_pyramid

METADATA_KEYS = ('dimension_order', 'shapes', 'dtype', 'pixel_sizes', 'channels')


def reference_metadata(path):
    """What init_metadata derives via ngff_zarr, its fallback path."""
    source = TiffImageSource.__new__(TiffImageSource)
    source.filename = path
    source.channels = []
    source.pixel_sizes = []
    source._data = []
    source._data_loaded = False
    source.shapes = []
    source.is_rgb = False
    source.position = {}
    source.dimension_order = ''
    source._init_metadata_from_ngff_zarr()
    return {'dimension_order': source.dimension_order,
            'shapes': [tuple(shape) for shape in source.shapes],
            'dtype': source.dtype,
            'pixel_sizes': source.pixel_sizes,
            'channels': source.channels,
            'is_rgb': source.is_rgb}


def ome_metadata(axes, unit='µm', **sizes):
    metadata = {'axes': axes}
    for dim, size in sizes.items():
        metadata[f'PhysicalSize{dim.upper()}'] = size
        metadata[f'PhysicalSize{dim.upper()}Unit'] = unit
    return metadata


# (filename, shape, dtype, levels, write kwargs, level-0 pixel size or None)
CASES = [
    pytest.param('plain.tiff', (1024, 768), np.uint16, 1, {}, None, id='plain 2d'),
    pytest.param('pyr.tiff', (2048, 2048), np.uint16, 3, {}, None, id='pyramidal 2d'),
    pytest.param('uint8.tiff', (512, 512), np.uint8, 1, {}, None, id='uint8'),
    pytest.param('float.tiff', (512, 512), np.float32, 1, {}, None, id='float32'),
    pytest.param('multi.ome.tiff', (3, 256, 256), np.uint16, 1,
                 {'metadata': {**ome_metadata('CYX', x=0.25, y=0.25), 'Channel': {'Name': ['DAPI', 'GFP', 'RFP']}}},
                 {'y': 0.25, 'x': 0.25}, id='multichannel ome'),
    pytest.param('z.ome.tiff', (4, 256, 256), np.uint16, 1, {'metadata': ome_metadata('ZYX', x=0.5, y=0.5, z=2.0)},
                 {'z': 2.0, 'y': 0.5, 'x': 0.5}, id='3d ome'),
    pytest.param('mm.ome.tiff', (256, 256), np.uint16, 1, {'metadata': ome_metadata('YX', 'mm', x=0.001, y=0.001)},
                 {'y': 1.0, 'x': 1.0}, id='millimetre ome'),
    pytest.param('pyr.ome.tiff', (2048, 2048), np.uint16, 3, {'metadata': ome_metadata('YX', x=0.5, y=0.5)},
                 {'y': 0.5, 'x': 0.5}, id='pyramidal ome'),
]


@pytest.mark.parametrize('filename, shape, dtype, levels, kwargs, pixel_size', CASES)
def test_the_fast_read_matches_ngff_zarrs(tmp_path, filename, shape, dtype, levels, kwargs, pixel_size):
    path = str(tmp_path / filename)
    if levels > 1:
        write_tiff_pyramid(path, shape, levels, dtype, **kwargs)
    else:
        tifffile.imwrite(path, np.zeros(shape, dtype=dtype), **kwargs)

    fast = read_tiff_source_metadata(path)

    assert_same_metadata(fast, reference_metadata(path), METADATA_KEYS)
    assert len(fast['shapes']) == levels
    if pixel_size is not None:
        # each level's pixel size scales with its downsampling
        for level, level_pixel_size in enumerate(fast['pixel_sizes']):
            assert level_pixel_size == pytest.approx({dim: size * 2 ** level for dim, size in pixel_size.items()})
    if 'Channel' in kwargs.get('metadata', {}):
        assert [channel['label'] for channel in fast['channels']] == kwargs['metadata']['Channel']['Name']


def test_rgb_lands_on_a_channel_dim(tmp_path):
    # tifffile reports 'YXS' (samples); ngff_zarr maps S onto 'c', so is_rgb must still hold
    path = str(tmp_path / 'rgb.tiff')
    tifffile.imwrite(path, np.zeros((512, 512, 3), dtype=np.uint8), photometric='rgb')
    reference = reference_metadata(path)
    fast = read_tiff_source_metadata(path)

    assert_same_metadata(fast, reference, METADATA_KEYS)
    assert 'c' in fast['dimension_order']
    assert create_image_source(path).is_rgb is reference['is_rgb'] is True


def test_ome_plane_position_is_read(tmp_path):
    path = str(tmp_path / 'pos.ome.tiff')
    tifffile.imwrite(path, np.zeros((256, 256), dtype=np.uint16),
                     metadata={'axes': 'YX',
                               'PhysicalSizeX': 1.0, 'PhysicalSizeY': 1.0,
                               'Plane': {'PositionX': [12.0], 'PositionXUnit': ['µm'],
                                         'PositionY': [-4.0], 'PositionYUnit': ['µm']}})
    assert read_tiff_source_metadata(path)['position'] == pytest.approx({'x': 12.0, 'y': -4.0})
    assert create_image_source(path).position == pytest.approx({'x': 12.0, 'y': -4.0})


def test_a_source_reports_the_reference_metadata_and_builds_its_arrays_only_when_read(tmp_path):
    path = write_tiff_pyramid(tmp_path / 'pyr.ome.tiff', metadata=ome_metadata('YX', x=0.25, y=0.25))
    source = create_image_source(path)

    assert source._data_loaded is False
    assert_same_metadata({key: getattr(source, key) for key in METADATA_KEYS}, reference_metadata(path), METADATA_KEYS)

    data = source.data
    assert source._data_loaded is True
    assert [tuple(level.shape) for level in data] == [tuple(shape) for shape in source.shapes]
    assert np.asarray(data[-1]).sum() == 0


def test_multi_image_ome_xml_parsing_stops_once_settled():
    """A multi-file OME-TIFF carries the whole dataset's XML per file, so the parse must stop early:
    the document is malformed past the first fed chunk, so reaching the end would raise."""
    def xml(count, tail=''):
        images = ''.join(
            f'<Image ID="Image:{index}"><Pixels ID="Pixels:{index}" Type="uint16" SizeX="64" SizeY="64"'
            f' SizeC="1" SizeZ="1" SizeT="1" PhysicalSizeX="0.5" PhysicalSizeY="0.5">'
            f'<Channel ID="Channel:{index}:0" Name="ch{index}"/>'
            f'<Plane TheC="0" TheZ="0" TheT="0" PositionX="{index}.0" PositionY="0.0"/>'
            f'</Pixels></Image>' for index in range(count))
        return ('<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/'
                f'2016-06">{images}{tail}</OME>')

    small = xml(2)
    large = xml(4000, tail='<Unclosed>' * 5 + '<<<not xml&&&')
    assert len(large) > 512 * 1024

    # scale and channels come from the first Image either way; position is voided for multi-Image
    for document in (small, large):
        metadata = extract_ome_image_metadata(document)
        assert metadata['scale'] == {'x': 0.5, 'y': 0.5}
        assert metadata['channel_names'] == ['ch0']
        assert metadata['position'] == {}


def ome_plane_xml(images):
    """images: list of dicts of Plane attributes (or None for an Image with no Plane)."""
    body = []
    for index, plane in enumerate(images):
        plane_xml = ''
        if plane is not None:
            attrs = ' '.join(f'{key}="{value}"' for key, value in plane.items())
            plane_xml = f'<Plane TheC="0" TheZ="0" TheT="0" {attrs}/>'
        body.append(f'<Image ID="Image:{index}"><Pixels ID="Pixels:{index}" Type="uint16"'
                    f' SizeX="64" SizeY="64" SizeC="1" SizeZ="1" SizeT="1">'
                    f'{plane_xml}</Pixels></Image>')
    return ('<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            + ''.join(body) + '</OME>')


def xml2dict_translation(ome_metadata):
    """The xml2dict version the streaming parse replaced, kept as the oracle."""
    from muvis_align.util import convert_to_um

    metadata = tifffile.xml2dict(ome_metadata)
    if 'OME' in metadata:
        metadata = metadata['OME']
    if 'Image' in metadata and 'Pixels' in metadata['Image'] and 'Plane' in metadata['Image']['Pixels']:
        plane_metadata = metadata['Image']['Pixels']['Plane']
        if isinstance(plane_metadata, list):
            plane_metadata = plane_metadata[0]
        position = {}
        for dim in ['X', 'Y', 'Z']:
            key = f'Position{dim}'
            if key in plane_metadata:
                position[dim.lower()] = convert_to_um(float(plane_metadata[key]),
                                                      plane_metadata.get(f'{key}Unit', 'um'))
        return position
    return {}


@pytest.mark.parametrize('images', [
    [{'PositionX': 12.5, 'PositionY': -3.25, 'PositionXUnit': 'um', 'PositionYUnit': 'um'}],
    [{'PositionX': 1.0, 'PositionY': 2.0, 'PositionZ': 3.0,
      'PositionXUnit': 'um', 'PositionYUnit': 'um', 'PositionZUnit': 'um'}],
    [{'PositionX': 7.0, 'PositionY': 8.0}],
    [{'PositionX': 1.5, 'PositionY': 2.5, 'PositionXUnit': 'mm', 'PositionYUnit': 'mm'}],
    [{'PositionX': 4.0}],
    [{}],
    [None],
    # multi-image: the historical behaviour is no position at all
    [{'PositionX': 1.0, 'PositionY': 2.0}, {'PositionX': 3.0, 'PositionY': 4.0}],
    [{'PositionX': float(index), 'PositionY': 0.0} for index in range(20)],
], ids=['xy um', 'xyz', 'no unit attributes', 'millimetre units', 'x only', 'no positions', 'no plane',
        'two images', 'many images'])
def test_the_ome_translation_matches_the_xml2dict_implementation(images):
    xml = ome_plane_xml(images)
    assert extract_ome_translation_from_xml(xml) == xml2dict_translation(xml)


def test_ome_translation_units_are_converted_to_um():
    xml = ome_plane_xml([{'PositionX': 1.5, 'PositionY': 2.5, 'PositionXUnit': 'mm', 'PositionYUnit': 'mm'}])
    assert extract_ome_translation_from_xml(xml) == pytest.approx({'x': 1500.0, 'y': 2500.0})


def test_source_levels_pickle_and_read_the_same_pixels_after(tmp_path):
    """Worker processes are sent a source's levels pickled: each must reopen its file and read what it read here."""
    import pickle
    import dask
    data = np.random.default_rng(0).integers(0, 1000, (1024, 1024), dtype=np.uint16)
    path = write_tiff_pyramid(tmp_path / 'pyr.tiff', data=data)
    levels = create_image_source(path).data
    expected = [np.asarray(level) for level in levels]

    # repeated on many threads: unpickled levels opened concurrently on their first reads once hit a closed file
    for _ in range(30):
        unpickled = pickle.loads(pickle.dumps(levels))
        assert len(unpickled) == 3
        with dask.config.set(scheduler='threads', num_workers=16):
            for level, copy in zip(expected, unpickled):
                assert np.array_equal(np.asarray(copy), level)


def test_an_unpickled_level_is_not_read_before_its_opening_file_is_closed(tmp_path, monkeypatch):
    """A thread reading the level while another opened it once kept the opener's soon-closed file handle."""
    import threading
    from muvis_align.image.ome_tiff_helper import _unpickle_tiff_level
    path = str(tmp_path / 'tiles.tiff')
    data = np.random.default_rng(1).integers(0, 1000, (512, 512), dtype=np.uint16)
    tifffile.imwrite(path, data, tile=(256, 256))
    level = _unpickle_tiff_level(path, '0', data.shape, data.dtype, (256, 256))
    read_meanwhile = threading.Event()
    reader = threading.Thread(target=lambda: (level[0:256, 0:256], read_meanwhile.set()))
    original_exit = tifffile.TiffFile.__exit__

    def exit_after_a_read_elsewhere(tif, *args):
        # the opener about to close its file, another thread reading the level (it waits, once fixed)
        monkeypatch.setattr(tifffile.TiffFile, '__exit__', original_exit)
        reader.start()
        read_meanwhile.wait(0.5)
        return original_exit(tif, *args)

    monkeypatch.setattr(tifffile.TiffFile, '__exit__', exit_after_a_read_elsewhere)
    assert np.array_equal(level[256:512, 256:512], data[256:512, 256:512])
    reader.join(10)
    assert np.array_equal(level[:, :], data)


def _level_wrappers(levels):
    from muvis_align.image.ome_tiff_helper import PicklableTiffLevel
    return [value for level in levels for layer in level.dask.layers.values()
            for value in (layer.values() if hasattr(layer, 'values') else []) if isinstance(value, PicklableTiffLevel)]


def test_uncompressed_strip_levels_are_read_as_their_rows_and_match_tifffile(tmp_path):
    import pickle
    from muvis_align.image.ome_tiff_helper import read_tiff_level_arrays
    path = str(tmp_path / 'strips.tiff')
    data = np.random.default_rng(2).integers(0, 60000, (300, 200), dtype=np.uint16)
    with tifffile.TiffWriter(path) as writer:
        writer.write(data, subifds=1, rowsperstrip=300)
        writer.write(data[::2, ::2], subfiletype=1, rowsperstrip=150)

    levels = read_tiff_level_arrays(path)
    wrappers = _level_wrappers(levels)
    with tifffile.TiffFile(path) as tif:
        expected = [level.asarray() for level in tif.series[0].levels]

    assert [wrapper.layout is not None for wrapper in wrappers] == [True, True]
    for wrapper, reference in zip(wrappers, expected):
        copy = pickle.loads(pickle.dumps(wrapper))
        assert copy.layout == wrapper.layout
        for key in [slice(None), (slice(10, 90), slice(5, 50)), 7, (slice(None, None, 3), 4), (-1,)]:
            assert np.array_equal(copy[key], reference[key])
    assert all(np.array_equal(np.asarray(level), reference) for level, reference in zip(levels, expected))


@pytest.mark.parametrize('options', [{'compression': 'zlib'}, {'tile': (64, 64)}])
def test_compressed_or_tiled_levels_are_read_through_zarr(tmp_path, options):
    from muvis_align.image.ome_tiff_helper import read_tiff_level_arrays
    path = str(tmp_path / 'other.tiff')
    data = np.random.default_rng(3).integers(0, 1000, (128, 128), dtype=np.uint16)
    tifffile.imwrite(path, data, **options)

    levels = read_tiff_level_arrays(path)

    assert [wrapper.layout for wrapper in _level_wrappers(levels)] == [None]
    assert np.array_equal(np.asarray(levels[0]), data)


def test_a_vendor_tiff_gives_its_own_pixel_size_and_stage_position(tmp_path):
    from tests._dummy_tiff import write_vendor_tiff

    path = str(tmp_path / 'vendor.tif')
    write_vendor_tiff(path, '<Vendor><pixelsizex>2e-9</pixelsizex><pixelsizey>2e-9</pixelsizey>'
                            '<Stage><X><value>10.5</value><units>um</units></X>'
                            '<Y><value>-3.5</value><units>um</units></Y></Stage>'
                            '<Beam><ElectricRotate>-30</ElectricRotate></Beam></Vendor>')
    source = create_image_source(path)

    assert source.pixel_size == pytest.approx({'y': 2e-3, 'x': 2e-3})
    assert source.position == {'x': 10.5, 'y': -3.5}
    # the scan rotation turns the image the other way
    assert source.rotation == 30.0
    assert 'Vendor' in str(source.metadata)
    assert 'StripOffsets' not in source.metadata
