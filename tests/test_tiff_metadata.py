# Copied from napari-meta-tiff (tests/test_metadata.py, 426b84f): keep in step with it

"""Tests for reading what the metadata says about space.

Run this module to execute the tests by hand:

    python -m tests.test_tiff_metadata
"""

import pytest
from tifffile import TiffFile, imwrite

from muvis_align.image.tiff_metadata import (get_extra_metadata,
                                             get_pixel_size_um, get_position_um,
                                             get_rotation_deg,
                                             get_stage_rotation_deg, parse_quantity,
                                             resolution_pixel_size)

from tests._dummy_tiff import (dummy_image, write_ome_tiff,
                               write_resolution_tiff, write_vendor_tiff)


# what one pixel of the image below measures, in micrometres
SHAPE = {'x': 1024, 'y': 512}


def test_parse_quantity_forms():
    """A measurement is read however the vendor chose to write it."""
    # a value beside its unit, as FEI and Fibics write it
    assert parse_quantity({'value': 1.5, 'units': 'mm'}) == 1500
    # the unit stuck on the end of the number, as Incucyte writes it
    assert parse_quantity('1.244 µm') == 1.244
    # the same, through the broken encoding Helios writes it in
    assert parse_quantity('21.12\xa6\xccm') == 21.12
    # a bare number, whose unit is known from where it was found
    assert parse_quantity(1.6529e-9, 'm') == 1.6529e-3
    # anything that is not a measurement at all
    assert parse_quantity('unknown') is None
    assert parse_quantity(1.5, 'furlong') is None


def test_garbled_micro_sign_is_repaired(tmp_path):
    """A micro sign written in GBK reads as one, not as '¦Ì'."""
    path = str(tmp_path / 'garbled.tif')
    # Helios writes the Greek mu in GBK, which tifffile reads as latin-1
    write_vendor_tiff(path, '<Vendor><Beam><HFW>21.12μm</HFW></Beam>'
                            '</Vendor>'.encode('gbk'))
    with TiffFile(path) as tif:
        metadata = get_extra_metadata(tif)
    assert metadata['65000']['Vendor']['Beam']['HFW'] == '21.12µm'


def test_pixel_size_from_vendor_fields(tmp_path):
    """A vendor stating the pixel size is read, metres where unstated."""
    path = str(tmp_path / 'vendor.tif')
    write_vendor_tiff(path, '<Vendor><pixelsizex>1.6528e-9</pixelsizex>'
                            '<pixelsizey>1.6528e-9</pixelsizey></Vendor>')
    with TiffFile(path) as tif:
        pixel_size = get_pixel_size_um(tif, get_extra_metadata(tif), SHAPE)
    assert pixel_size == pytest.approx({'x': 1.6528e-3, 'y': 1.6528e-3})


def test_pixel_size_from_field_of_view(tmp_path):
    """The width of the image divided by its pixels is a pixel size."""
    path = str(tmp_path / 'fov.tif')
    write_vendor_tiff(path, '<Vendor><Scan><FOV_X><value>129.044885430288'
                            '</value><units>um</units></FOV_X></Scan>'
                            '</Vendor>')
    with TiffFile(path) as tif:
        pixel_size = get_pixel_size_um(tif, get_extra_metadata(tif), SHAPE)
    # a field of view stated for one axis describes square pixels
    expected = 129.044885430288 / SHAPE['x']
    assert pixel_size == {'x': expected, 'y': expected}


def test_pixel_size_ignores_pixel_counts(tmp_path):
    """Fields counting pixels are not fields measuring one."""
    path = str(tmp_path / 'counts.tif')
    write_vendor_tiff(path, '<Vendor><PixelXDimension>5120</PixelXDimension>'
                            '<SamplesPerPixel>1</SamplesPerPixel>'
                            '<Scan><ResolutionX>8448</ResolutionX></Scan>'
                            '</Vendor>')
    with TiffFile(path) as tif:
        assert get_pixel_size_um(tif, get_extra_metadata(tif), SHAPE) == {}


def keys_anywhere(value):
    """Yield every key in a nested metadata structure."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys_anywhere(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys_anywhere(item)


def test_baseline_tags_pass_through_but_the_pixel_layout(tmp_path):
    """Every baseline tag describing the image is metadata, its pixel
    layout is not: the strips, tiles and compression only decode it."""
    path = str(tmp_path / 'baseline.tif')
    imwrite(path, dummy_image(32), tile=(16, 16), compression='zlib',
            resolution=(4, 4), resolutionunit='CENTIMETER',
            datetime='2023:11:07 12:24:09', software='Acme 1.0',
            description='a plain description', metadata=None)
    with TiffFile(path) as tif:
        metadata = get_extra_metadata(tif)

    assert metadata['ImageWidth'] == metadata['ImageLength'] == 32
    assert metadata['XResolution'] == (4, 1)
    assert metadata['ResolutionUnit'] == 'CENTIMETER'
    assert metadata['DateTime'] == '2023:11:07 12:24:09'
    assert metadata['Software'] == 'Acme 1.0'
    assert metadata['ImageDescription'] == 'a plain description'
    layout = {'TileOffsets', 'TileByteCounts', 'TileWidth', 'TileLength',
              'Compression', 'Predictor', 'PlanarConfiguration'}
    assert not layout & set(metadata)


def test_ome_metadata_without_its_pixel_layout(tmp_path):
    """An OME document is the metadata, but where its pixels are written."""
    path = str(tmp_path / 'layout.ome.tif')
    write_ome_tiff(path, x=(0.325, 'µm'), y=(0.325, 'µm'))
    with TiffFile(path) as tif:
        metadata = get_extra_metadata(tif)

    pixels = metadata['OME']['Image']['Pixels']
    assert pixels['PhysicalSizeX'] == 0.325
    assert not {'TiffData', 'BinData', 'BigEndian',
                'Interleaved'} & set(keys_anywhere(metadata))


def test_pixel_size_from_ome(tmp_path):
    """OME states the physical size, in micrometres unless it says."""
    path = str(tmp_path / 'stated.ome.tif')
    write_ome_tiff(path, x=(0.325, 'µm'), y=(0.325, 'µm'))
    with TiffFile(path) as tif:
        assert get_pixel_size_um(tif, get_extra_metadata(tif),
                                 SHAPE) == {'x': 0.325, 'y': 0.325}

    path = str(tmp_path / 'bare.ome.tif')
    write_ome_tiff(path, x=(0.325, None), y=(0.325, None))
    with TiffFile(path) as tif:
        assert get_pixel_size_um(tif, get_extra_metadata(tif),
                                 SHAPE) == {'x': 0.325, 'y': 0.325}


def test_resolution_in_centimetres(tmp_path):
    """A resolution per centimetre means what it says."""
    path = str(tmp_path / 'cm.tif')
    write_resolution_tiff(path, (25000, 1))
    with TiffFile(path) as tif:
        assert resolution_pixel_size(tif) == {'x': 0.4, 'y': 0.4}


def test_degenerate_resolution_is_per_metre(tmp_path):
    """A resolution that cancels out is read as pixels per metre.

    Delmic writes 250000000/250000000 per centimetre, where only the
    numerator carries the 4 nm pixel the instrument images at.
    """
    path = str(tmp_path / 'degenerate.tif')
    write_resolution_tiff(path, (250000000, 250000000))
    with TiffFile(path) as tif:
        assert resolution_pixel_size(tif) == {'x': 0.004, 'y': 0.004}


def test_implausible_resolution_is_dropped(tmp_path):
    """A resolution no image could have is not a measurement."""
    path = str(tmp_path / 'implausible.tif')
    # one pixel per metre, which is the 1/1 an overview is written with
    write_resolution_tiff(path, (1, 1))
    with TiffFile(path) as tif:
        assert get_pixel_size_um(tif, get_extra_metadata(tif), SHAPE) == {}


def test_screen_resolution_is_dropped(tmp_path):
    """The resolution a writer fills in for display is ignored."""
    for dots_per_inch in (72, 96):
        path = str(tmp_path / f'{dots_per_inch}dpi.tif')
        write_resolution_tiff(path, (dots_per_inch, 1), unit='INCH')
        with TiffFile(path) as tif:
            assert resolution_pixel_size(tif) == {}


def test_position_from_stage(tmp_path):
    """A stage position is read, metres where no unit is given."""
    path = str(tmp_path / 'stage.tif')
    write_vendor_tiff(path, '<Vendor><Stage><X><value>10.5</value>'
                            '<units>um</units></X><Y><value>-3.5</value>'
                            '<units>um</units></Y></Stage></Vendor>')
    with TiffFile(path) as tif:
        assert get_position_um(get_extra_metadata(tif)) == {'x': 10.5,
                                                            'y': -3.5}

    path = str(tmp_path / 'bare_stage.tif')
    write_vendor_tiff(path, '<Vendor><samplePosition><x>0.002</x>'
                            '<y>-0.001</y></samplePosition></Vendor>')
    with TiffFile(path) as tif:
        assert get_position_um(get_extra_metadata(tif)) == {'x': 2000.0,
                                                            'y': -1000.0}


def test_position_in_millimetres_from_stage_travel(tmp_path):
    """A named stage travel says a bare position is in millimetres.

    Helios writes 18.9417 beside a stage model naming a 110x110x69 mm
    travel, so the coordinate is millimetres: metres would put the
    sample further away than the stage can reach.
    """
    path = str(tmp_path / 'travel.tif')
    write_vendor_tiff(path, '<Vendor><Stage><StageType>'
                            'TMCM-GYLZ-4000X-FIVE-110x110x69</StageType>'
                            '<StagePosX>18.9417</StagePosX>'
                            '<StagePosY>11.1248</StagePosY></Stage></Vendor>')
    with TiffFile(path) as tif:
        position = get_position_um(get_extra_metadata(tif))
    assert position == pytest.approx({'x': 18941.7, 'y': 11124.8})


def test_position_beyond_stage_travel_is_metres(tmp_path):
    """A hint that does not fit the coordinate is not applied.

    A position larger than the travel cannot be millimetres, so it is
    read as metres, the way one is read with no hint at all.
    """
    path = str(tmp_path / 'beyond.tif')
    write_vendor_tiff(path, '<Vendor><Stage><StageType>'
                            'TMCM-110x110x69</StageType>'
                            '<StagePosX>500</StagePosX></Stage></Vendor>')
    with TiffFile(path) as tif:
        position = get_position_um(get_extra_metadata(tif))
    assert position == pytest.approx({'x': 500e6})


def test_position_prefers_the_stage_section():
    """The stage's own section wins over the copy in a beam's section.

    An FEI navigation camera image keeps another position in its IRBeam
    section, written before the Stage section.
    """
    metadata = {'FEI_HELIOS': {
        'IRBeam': {'StageX': -0.00126661, 'StageY': 0.002136},
        'Stage': {'StageX': 0.0699197, 'StageY': 0.000811186}}}
    assert get_position_um(metadata) == pytest.approx({'x': 69919.7,
                                                       'y': 811.186})


def test_position_ignores_other_fields(tmp_path):
    """Fields that are not a position in space are left alone."""
    path = str(tmp_path / 'other.tif')
    write_vendor_tiff(path, '<Vendor><ScanInfo><TrayPosition>Rear'
                            '</TrayPosition></ScanInfo><Scan><sourceTilt>'
                            '<x>-0.012</x></sourceTilt></Scan></Vendor>')
    with TiffFile(path) as tif:
        assert get_position_um(get_extra_metadata(tif)) == {}


# the image is turned by the scan rotation, the other way, apart from the
# stage rotation, which turns the sample: (metadata, image, stage rotation)
ROTATION_CASES = {
    # Ciqtek, whose row of tiles at ElectricRotate -135 lines up at 135
    'ciqtek': ({'Beam': {'ElectricRotate': -135.0},
                'Stage': {'StagePosR': -30.0, 'StagePosX': 0.002}}, 135.0, -30.0),
    # Ciqtek writes to the FEI tag, but in degrees, under its own names
    'ciqtek in the fei tag': ({'FEI_HELIOS': {'Beam': {'ElectricRotate': 0.0},
                                              'Stage': {'StagePosR': 11.7}}},
                              0.0, 11.7),
    # FEI writes every angle in radians without saying so
    'fei': ({'FEI_HELIOS': {'Beam': {'ScanRotation': 0.5},
                            'Stage': {'StageR': 1.07874}}}, -28.6479, 61.8072),
    'phenom': ({'acquisition': {'scan': {'rotation': 90}}}, -90.0, None),
    'fibics': ({'Fibics': {'Scan': {'ScanRot': {'units': 'deg', 'value': 10}}}},
               -10.0, None),
    'stated radians': ({'Stage': {'R': {'value': 0.5, 'units': 'rad'}}},
                       None, 28.6479),
    'neither': ({'Stage': {'StagePosX': 0.002}}, None, None),
}


@pytest.mark.parametrize('metadata, rotation, stage_rotation',
                         ROTATION_CASES.values(), ids=ROTATION_CASES.keys())
def test_rotations(metadata, rotation, stage_rotation):
    """The image and the stage rotation are read apart, in degrees."""
    image_rotation = get_rotation_deg(metadata)
    if rotation is None:
        assert image_rotation is None
    else:
        assert image_rotation == pytest.approx(rotation, abs=1e-4)
        # no rotation is not a turn of -0.0
        assert str(image_rotation) != '-0.0'
    if stage_rotation is None:
        assert get_stage_rotation_deg(metadata) is None
    else:
        assert get_stage_rotation_deg(metadata) == pytest.approx(stage_rotation,
                                                                 abs=1e-4)
    # neither turns up as a position
    assert 'r' not in get_position_um(metadata)



if __name__ == '__main__':
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)

        test_parse_quantity_forms()
        test_garbled_micro_sign_is_repaired(tmp_path)
        test_pixel_size_from_vendor_fields(tmp_path)
        test_pixel_size_from_field_of_view(tmp_path)
        test_pixel_size_ignores_pixel_counts(tmp_path)
        test_pixel_size_from_ome(tmp_path)
        test_resolution_in_centimetres(tmp_path)
        test_degenerate_resolution_is_per_metre(tmp_path)
        test_implausible_resolution_is_dropped(tmp_path)
        test_screen_resolution_is_dropped(tmp_path)
        test_position_from_stage(tmp_path)
        test_position_in_millimetres_from_stage_travel(tmp_path)
        test_position_beyond_stage_travel_is_metres(tmp_path)
        test_position_prefers_the_stage_section()
        test_position_ignores_other_fields(tmp_path)
        for case in ROTATION_CASES.values():
            test_rotations(*case)
