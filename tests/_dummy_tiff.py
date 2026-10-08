# Copied from napari-meta-tiff (tests/_dummy_tiff.py, 1e5d28a): keep in step with it

"""Helpers creating dummy TIFF files for the tests.

These are also used by the manual entry points of the test modules, so that
the files explored by hand in napari are the same ones the tests assert on.
"""

import os
import struct

import numpy as np
from tifffile import TiffFile, TiffWriter, imwrite


TEST_METADATA = {'test_metadata': 'test metadata'}
# Make and Model tiff tags, which the reader promotes to layer metadata
TEST_MAKE = 'Test Make'
TEST_MODEL = 'Test Model'
MAKE_MODEL_EXTRATAGS = [
    (271, 's', 0, TEST_MAKE, True),
    (272, 's', 0, TEST_MODEL, True),
]

# an Exif IFD, whose fields the reader merges into the layer metadata
# rather than keeping them behind the ExifTag pointing at them. Only
# fields fitting in the four value bytes of an entry are used, so the
# IFD written below needs nothing out of line.
TEST_EXIF = {'ExifVersion': '0230', 'ISOSpeedRatings': 400,
             'PixelXDimension': 16}
EXIF_TAG_CODE = 34665


def dummy_image(size: int) -> np.ndarray:
    """Return a deterministic gradient image, recognisable when viewed."""
    x = np.linspace(0, 1, size, dtype=np.float32)
    pattern = np.outer(np.sin(8 * x) ** 2, x) + np.outer(x, np.cos(8 * x) ** 2)
    return (pattern * 127).astype(np.uint8)


def write_dummy_tiff(path: str, size: int = 16, extratags=()) -> np.ndarray:
    """Write a single level dummy tiff and return the image data."""
    data = dummy_image(size)
    imwrite(path, data, metadata=TEST_METADATA, extratags=extratags)
    return data


def write_series_tiff(path: str, images) -> list:
    """Write each image as its own series, in the order given.

    Pages only group into one series when they share a shape and a
    dtype, so images differing in either, which is what the series
    ranking has to choose between, land in separate series.
    """
    with TiffWriter(path) as tif:
        for image in images:
            tif.write(image, metadata=TEST_METADATA, contiguous=False)
    return list(images)


def rgb_image(size: int) -> np.ndarray:
    """Return an 8 bit RGB rendering of the dummy image."""
    return np.repeat(dummy_image(size)[:, :, np.newaxis], 3, axis=2)


def grey_image(size: int) -> np.ndarray:
    """Return a 16 bit greyscale version of the dummy image."""
    return (dummy_image(size).astype(np.uint16) << 8)


def write_pyramid_tiff(path: str, nlevels: int = 4, size: int = 512,
                       tile=(128, 128), extratags=()) -> list:
    """Write a pyramidal dummy tiff and return the data of each level.

    The full resolution image is written first, followed by nlevels - 1
    downsampled subifds, so the levels are ordered highest resolution first.
    """
    # decimate a single pattern, so the levels are a genuine pyramid
    # of the same image rather than unrelated arrays
    base = dummy_image(size)
    levels = [base[::2**level, ::2**level] for level in range(nlevels)]
    with TiffWriter(path) as tif:
        tif.write(levels[0], tile=tile, subifds=nlevels - 1,
                  metadata=TEST_METADATA, extratags=extratags)
        for level_data in levels[1:]:
            tif.write(level_data, tile=tile, subfiletype=1)
    return levels


def write_exif_tiff(path: str, size: int = 16) -> np.ndarray:
    """Write a dummy tiff holding a real Exif IFD and return the image.

    tifffile refuses to write the Exif pointer tag itself, so the file is
    written normally and then given an Exif IFD, along with a copy of its
    main IFD pointing at it. The original entries are copied over
    untouched and the header is pointed at the copy, which leaves every
    value they hold out of line where it already is.
    """
    data = write_dummy_tiff(path, size=size)
    entries = [
        (0x9000, 7, 4, b'0230'),                        # ExifVersion
        (0x8827, 3, 1, struct.pack('<HH', 400, 0)),     # ISOSpeedRatings
        (0xA002, 4, 1, struct.pack('<I', size)),        # PixelXDimension
    ]
    with open(path, 'r+b') as fh:
        byteorder = '<' if fh.read(2) == b'II' else '>'
        fh.seek(4)
        ifd_offset, = struct.unpack(byteorder + 'I', fh.read(4))
        fh.seek(ifd_offset)
        count, = struct.unpack(byteorder + 'H', fh.read(2))
        old_entries = [fh.read(12) for _ in range(count)]
        next_ifd = fh.read(4)

        fh.seek(0, os.SEEK_END)
        if fh.tell() % 2:
            fh.write(b'\0')       # ifds start on a word boundary
        exif_offset = fh.tell()
        fh.write(struct.pack(byteorder + 'H', len(entries)))
        for code, dtype, value_count, value in sorted(entries):
            fh.write(struct.pack(byteorder + 'HHI', code, dtype, value_count)
                     + value.ljust(4, b'\0'))
        fh.write(struct.pack(byteorder + 'I', 0))       # no next ifd

        pointer = (struct.pack(byteorder + 'HHI', EXIF_TAG_CODE, 4, 1)
                   + struct.pack(byteorder + 'I', exif_offset))
        new_entries = sorted(old_entries + [pointer],
                             key=lambda entry: struct.unpack(
                                 byteorder + 'H', entry[:2]))
        main_offset = fh.tell()
        fh.write(struct.pack(byteorder + 'H', len(new_entries)))
        fh.write(b''.join(new_entries))
        fh.write(next_ifd)
        fh.seek(4)
        fh.write(struct.pack(byteorder + 'I', main_offset))
    return data


# a private tag code tifffile knows nothing about, so its value is
# collected the way a vendor's own tag is
VENDOR_TAG_CODE = 65000


def write_vendor_tiff(path: str, xml: str, size: int = 16) -> np.ndarray:
    """Write a tiff carrying a vendor xml document in a private tag."""
    return write_dummy_tiff(path, size=size,
                            extratags=[(VENDOR_TAG_CODE, 's', 0, xml, True)])


def write_resolution_tiff(path: str, resolution, unit: str = 'CENTIMETER',
                          size: int = 16) -> np.ndarray:
    """Write a tiff whose resolution is exactly the rational given.

    tifffile reduces a resolution to its lowest terms, which is the
    whole point of the degenerate rationals this has to produce, so the
    written values are patched in place afterwards.
    """
    data = dummy_image(size)
    imwrite(path, data, metadata=TEST_METADATA, resolution=(1, 1),
            resolutionunit=unit)
    numerator, denominator = resolution
    with open(path, 'r+b') as fh:
        byteorder = '<' if fh.read(2) == b'II' else '>'
        with TiffFile(path) as tif:
            offsets = [tif.pages.first.tags[name].valueoffset
                       for name in ('XResolution', 'YResolution')]
        for offset in offsets:
            fh.seek(offset)
            fh.write(struct.pack(byteorder + 'II', numerator, denominator))
    return data


def write_ome_tiff(path: str, size: int = 16, **pixel_size) -> np.ndarray:
    """Write an ome-tiff stating the physical size of a pixel."""
    data = dummy_image(size)
    metadata = {}
    for axis, (value, unit) in pixel_size.items():
        metadata[f'PhysicalSize{axis.upper()}'] = value
        if unit is not None:
            metadata[f'PhysicalSize{axis.upper()}Unit'] = unit
    imwrite(path, data, ome=True, metadata=metadata)
    return data
