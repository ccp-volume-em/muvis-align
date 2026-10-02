# Copied from napari-meta-tiff (src/napari_meta_tiff/_metadata.py, 75897dc): keep in step with it

"""Read the metadata of a TIFF file, and what it says about space.

Vendors write the pixel size and the stage position in their own tags,
under their own names and in their own units, so rather than reading the
tags of particular vendors, the whole metadata tree is searched for the
names these values are known by, and every value is normalised to
micrometres. That way a vendor which is not known here is picked up as
well, as long as it names its fields the way the rest do.
"""

from enum import Enum
import logging
import re
from tifffile import TiffFile, xml2dict
from typing import Any, Dict, Iterator, Optional, Tuple
from xml.etree.ElementTree import ParseError


logger = logging.getLogger(__name__)

# TIFF reserves tag codes at or above this for a vendor's private use:
# https://www.awaresystems.be/imaging/tiff/tifftags/private.html
PRIVATE_TAG_CODE = 32768

# the Exif tag points at an IFD of standard acquisition fields, such as
# the exposure time, rather than at a vendor's own structure, so those
# fields are collected beside the other metadata instead of below it
EXIF_TAG_NAME = 'ExifTag'

# baseline tags that only lay out or encode the pixel data, meaningless
# beside pixels already read; every other baseline tag, such as the
# instrument's Make and Model, the resolution or the date, describes the
# image and is metadata like the rest
LAYOUT_TAG_NAMES = {
    'NewSubfileType', 'SubfileType', 'Compression', 'Predictor',
    'PlanarConfiguration', 'FillOrder', 'StripOffsets', 'StripByteCounts',
    'RowsPerStrip', 'TileWidth', 'TileLength', 'TileDepth', 'TileOffsets',
    'TileByteCounts', 'SubIFDs', 'JPEGTables', 'JPEGProc',
    'JPEGInterchangeFormat', 'JPEGInterchangeFormatLength',
    'JPEGRestartInterval', 'JPEGLosslessPredictors', 'JPEGPointTransforms',
    'JPEGQTables', 'JPEGDCTables', 'JPEGACTables', 'YCbCrCoefficients',
    'YCbCrSubSampling', 'YCbCrPositioning', 'ReferenceBlackWhite',
    'ExtraSamples', 'ColorMap'}

# the elements and attributes of an OME document that say where and how
# its pixel data is written, rather than anything about the image
OME_LAYOUT_NAMES = ('TiffData', 'BinData', 'BigEndian', 'Interleaved')

# ElementTree expands a namespaced xml attribute into a {namespace}name
# key. Attributes in this namespace, such as xsi:type and
# xsi:noNamespaceSchemaLocation, describe the document rather than the
# image, and vendors sprinkle them at every level of their schema.
XSI_NAMESPACE = '{http://www.w3.org/2001/XMLSchema-instance}'


def get_extra_metadata(tif: TiffFile) -> Dict[str, Any]:
    """Return all the metadata in a TIFF file, but what lays out its pixels.

    Rather than reading the tags of particular vendors, every private tag
    is collected and normalised the same way, so that vendors which are
    not known here are picked up as well. TIFF reserves tag codes at or
    above 32768 for a vendor's own use, which is where instrument
    metadata ends up. The baseline tags below that are collected beside
    them, but those that only lay out or encode the pixel data, such as
    the strips, tiles and JPEG tables.

    The Exif IFD is the exception: its fields are standard rather than a
    vendor's own, so they are merged in beside the rest instead of being
    nested behind the name of the tag that points at them.

    An OME-TIFF's metadata is its OME document, but where and how its
    pixel data is written.
    """
    if tif.is_ome and tif.ome_metadata:
        return repair_text(drop_names(decode_metadata(tif.ome_metadata),
                                      OME_LAYOUT_NAMES))

    extra_metadata = {}
    # setdefault lets the first page win, as later pages tend to be
    # thumbnails or reduced resolutions; they only fill in missing fields
    for page in tif.pages:
        for tag in page.tags.values():
            is_private = tag.code >= PRIVATE_TAG_CODE
            value = (decode_metadata(tag.value) if is_private
                     else baseline_value(tag.value))
            if (is_private and tag.name == EXIF_TAG_NAME
                    and isinstance(value, dict)):
                for name, field in value.items():
                    extra_metadata.setdefault(name, field)
            elif is_private and value not in (None, '', {}):
                # key by tag name, so that different vendor tags in one
                # file are kept side by side
                extra_metadata.setdefault(tag.name, value)
            elif not is_private and tag.name not in LAYOUT_TAG_NAMES:
                extra_metadata.setdefault(tag.name, value)
    return repair_text(extra_metadata)


def baseline_value(value: Any) -> Any:
    """Return a baseline tag's value as plain data.

    An enumeration, such as the photometric interpretation, becomes its
    name, and bytes, such as an XMP packet, the text they hold.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


def drop_names(value: Any, names: Tuple[str, ...]) -> Any:
    """Recursively drop the entries with any of `names` as their key."""
    if isinstance(value, dict):
        return {key: drop_names(item, names) for key, item in value.items()
                if key not in names}
    if isinstance(value, list):
        return [drop_names(item, names) for item in value]
    return value


def decode_metadata(value: Any) -> Any:
    """Decode a metadata value into the structure it holds.

    Vendors store their metadata as an xml document or as a nested
    mapping, so parse the xml into a mapping, keeping its root element,
    such as OME, FeiImage or Fibics, which names the vendor's schema.
    """
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, str):
        parsed = parse_xml(value)
        if parsed is None:
            return value
        value = parsed
    return drop_document_details(value)


def parse_xml(value: str) -> Optional[Dict]:
    """Return value parsed as an xml document, or None if it is not one.

    The xml declaration is optional, and vendors do leave it out, so
    rather than looking for one, hand anything that opens like a
    document to the parser and let it decide.
    """
    if not value.lstrip().startswith('<'):
        return None
    try:
        return xml2dict(value)
    except ParseError:
        return None


def drop_document_details(value: Any) -> Any:
    """Recursively drop the entries describing the xml document itself.

    The plumbing appears at every level of a vendor's schema, not just
    at the top, so this has to walk the whole tree.
    """
    if isinstance(value, dict):
        return {key: drop_document_details(item)
                for key, item in value.items()
                if not is_document_detail(key, item)}
    if isinstance(value, list):
        return [drop_document_details(item) for item in value]
    return value


def is_document_detail(key: Any, value: Any) -> bool:
    """Return whether an entry describes the document, not the image."""
    return ((isinstance(key, str) and key.startswith(XSI_NAMESPACE))
            or (isinstance(value, str) and '.xsd' in value.lower()))


def repair_text(value: Any) -> Any:
    """Recursively put back the characters a broken encoding garbled.

    Only the sequences known to be garbled are replaced, rather than
    guessing at the encoding of every string, which would mangle text
    that was read correctly.
    """
    if isinstance(value, str):
        for garbled, meant in MOJIBAKE.items():
            value = value.replace(garbled, meant)
        return value
    if isinstance(value, dict):
        return {key: repair_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [repair_text(item) for item in value]
    return value


# how many micrometres a unit is worth, keyed by the spellings vendors
# use. Angstrom is included because electron microscopes report in it.
UM_CONVERSIONS = {
    'a': 1e-4, 'angstrom': 1e-4,
    'nm': 1e-3, 'nanometer': 1e-3, 'nanometre': 1e-3,
    'um': 1, 'µm': 1, 'micrometer': 1, 'micrometre': 1, 'micron': 1,
    'mm': 1e3, 'millimeter': 1e3, 'millimetre': 1e3,
    'cm': 1e4, 'centimeter': 1e4, 'centimetre': 1e4,
    'm': 1e6, 'meter': 1e6, 'metre': 1e6,
    'inch': 25400, 'in': 25400,
}

# a pixel below this is smaller than an atom, and one above it is wider
# than a hand, so a value outside the range is a misread rather than a
# measurement, whatever the metadata claims
PLAUSIBLE_PIXEL_SIZE_UM = (1e-5, 1e4)

# resolutions every second consumer writer fills in, in pixels per inch,
# which say what a viewer should print at rather than what was imaged
SCREEN_RESOLUTIONS = (72, 96)

# the value and unit of a quantity, where a vendor spells them out as a
# pair of fields rather than as one string
VALUE_KEYS = ('value', 'val')
UNIT_KEYS = ('unit', 'units')

# a unit written onto the end of a number, as in '21.12um'
QUANTITY_PATTERN = re.compile(
    r'^\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*([^\d\s]*)\s*$')

# text a vendor wrote in one encoding and tifffile read back as latin-1,
# keyed by how it reads, with the character that was meant. Helios writes
# the micro sign as the Greek mu in GBK, whose two bytes read as '¦Ì'.
MOJIBAKE = {
    '¦Ì': 'µ',
}

# names a pixel size goes by, once the axis is taken off the end
PIXEL_SIZE_NAMES = ('pixelsize', 'pixelspacing', 'physicalsize')

# 'pixel' on its own only means a size when the axis was named by a
# dimension rather than by a letter, as in pixelWidth. Left looser it
# would read SamplesPerPixel, and anything else counted per pixel, as
# a measurement of one.
DIMENSION_SUFFIXES = ('width', 'height', 'depth')

# names the width of the whole image goes by, which divided by the
# number of pixels across gives the size of one of them
FIELD_OF_VIEW_NAMES = ('fov', 'fieldofview', 'fieldwidth', 'hfw',
                       'horizontalfieldwidth', 'vfw')

# names of the structures a stage position sits in, and of the fields
# themselves where a vendor writes them out flat instead
POSITION_CONTAINER_NAMES = ('stage', 'position')
POSITION_NAMES = ('stage', 'stagepos', 'position', 'pos')

# the travel of a stage, as vendors write it into the name of the
# model: 110x110x69 for the Helios stage, which is millimetres
STAGE_TRAVEL_PATTERN = re.compile(r'(\d{1,4})x(\d{1,4})(?:x(\d{1,4}))?',
                                  re.IGNORECASE)

# an axis written onto the end of a field name
AXIS_SUFFIXES = (('x', 'x'), ('y', 'y'), ('z', 'z'),
                 ('width', 'x'), ('height', 'y'), ('depth', 'z'))


def normalise_name(key: Any) -> str:
    """Return a key as bare lowercase letters, for comparing names."""
    if not isinstance(key, str):
        return ''
    return ''.join(character for character in key.lower()
                   if character.isalnum())


def split_axis(name: str) -> Tuple[str, Optional[str], str]:
    """Split the axis off the end of a normalised field name.

    The suffix itself comes back as well, because a name means
    different things depending on how the axis was written.
    """
    for suffix, axis in AXIS_SUFFIXES:
        if name.endswith(suffix) and name != suffix:
            return name[:-len(suffix)], axis, suffix
    return name, None, ''


def convert_to_um(value: Any, unit: Any) -> Optional[float]:
    """Return a value in micrometres, or None if it cannot be read."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    factor = UM_CONVERSIONS.get(normalise_name(unit))
    if factor is None:
        return None
    return value * factor


def parse_quantity(value: Any, unit: Any = None) -> Optional[float]:
    """Return a metadata value in micrometres, however it is written.

    A vendor writes a measurement as a number beside a unit field, as a
    number with the unit stuck on the end of the string, or as a bare
    number whose unit is named by a neighbouring field.
    """
    if isinstance(value, dict):
        keyed = {normalise_name(key): item for key, item in value.items()}
        for value_key in VALUE_KEYS:
            if value_key in keyed:
                return parse_quantity(keyed[value_key],
                                      find_unit(keyed) or unit)
        return None
    if isinstance(value, str):
        match = QUANTITY_PATTERN.match(repair_text(value))
        if match is None:
            return None
        number, written_unit = match.groups()
        return convert_to_um(number, written_unit or unit)
    return convert_to_um(value, unit)


def find_unit(keyed: Dict[str, Any]) -> Optional[str]:
    """Return the unit a mapping names, if it names one."""
    for unit_key in UNIT_KEYS:
        if isinstance(keyed.get(unit_key), str):
            return keyed[unit_key]
    return None


def walk_fields(metadata: Any) -> Iterator[Tuple[str, Any, Dict]]:
    """Yield every field in the metadata, with the mapping holding it.

    The mapping comes along so that a field can be read together with
    its neighbours, which is where the unit of a bare number is named.
    """
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            yield key, value, metadata
            yield from walk_fields(value)
    elif isinstance(metadata, list):
        for item in metadata:
            yield from walk_fields(item)


def sibling_unit(key: str, siblings: Dict) -> Optional[str]:
    """Return the unit a neighbouring field names for this one.

    OME writes PhysicalSizeX beside PhysicalSizeXUnit, which is the same
    name with the unit on the end.
    """
    for unit_key in UNIT_KEYS:
        for sibling, value in siblings.items():
            if (isinstance(value, str)
                    and normalise_name(sibling) == normalise_name(key)
                    + unit_key):
                return value
    return None


def get_pixel_size_um(tif: TiffFile, metadata: Dict,
                      shape: Dict[str, int]) -> Dict[str, float]:
    """Return the size of a pixel in micrometres, per axis.

    The sources are tried in the order they can be trusted: what a
    vendor states outright, then the width of the image divided by the
    pixels across it, and last the baseline TIFF resolution, which is
    the one every writer fills in whether it knows the answer or not.
    """
    pixel_size = {}
    if tif.is_imagej and tif.imagej_metadata:
        pixel_size.update(imagej_pixel_size(tif))
    for source in (stated_pixel_size, field_of_view_pixel_size):
        for axis, size in source(metadata, shape).items():
            pixel_size.setdefault(axis, size)
    for axis, size in resolution_pixel_size(tif).items():
        pixel_size.setdefault(axis, size)
    return {axis: size for axis, size in pixel_size.items()
            if is_plausible_pixel_size(size)}


def is_plausible_pixel_size(size: Any) -> bool:
    """Return whether a pixel size can be a measurement at all."""
    low, high = PLAUSIBLE_PIXEL_SIZE_UM
    return isinstance(size, float) and low <= size <= high


def stated_pixel_size(metadata: Dict, shape: Dict[str, int]) -> Dict:
    """Return the pixel size from any field naming one.

    This is where OME PhysicalSizeX, and every vendor field named after
    the size or the spacing of a pixel, is picked up.
    """
    pixel_size = {}
    for key, value, siblings in walk_fields(metadata):
        name, axis, suffix = split_axis(normalise_name(key))
        if not (name.endswith(PIXEL_SIZE_NAMES)
                or (name.endswith('pixel') and suffix in DIMENSION_SUFFIXES)):
            continue
        size = parse_quantity(value, sibling_unit(key, siblings)
                              or implied_unit(name))
        if size is None:
            continue
        for each_axis in (axis,) if axis else ('x', 'y'):
            pixel_size.setdefault(each_axis, size)
    return pixel_size


def implied_unit(name: str) -> str:
    """Return the unit a pixel size with no unit beside it is in.

    OME states that PhysicalSize defaults to micrometres. Anywhere else
    a bare length is in the SI unit: Olympus writes 1.65e-9, which is
    metres or nothing.
    """
    return 'um' if name.endswith('physicalsize') else 'm'


def field_of_view_pixel_size(metadata: Dict,
                             shape: Dict[str, int]) -> Dict:
    """Return the pixel size from the width of the whole image.

    Electron microscopes state the field of view rather than the pixel,
    so divide it by the pixels across the image.
    """
    pixel_size = {}
    for key, value, siblings in walk_fields(metadata):
        name, axis, _ = split_axis(normalise_name(key))
        if not name.endswith(FIELD_OF_VIEW_NAMES):
            continue
        axis = axis or 'x'
        extent = parse_quantity(value, sibling_unit(key, siblings))
        pixels = shape.get(axis)
        if extent is None or not pixels:
            continue
        pixel_size.setdefault(axis, extent / pixels)
    # a field of view given for one axis describes square pixels
    if len(pixel_size) == 1:
        (size,) = pixel_size.values()
        pixel_size = {'x': size, 'y': size}
    return pixel_size


def resolution_pixel_size(tif: TiffFile) -> Dict:
    """Return the pixel size from the baseline TIFF resolution.

    The resolution is a rational, and a writer with nothing to say
    fills it in anyway, so read it only where it can mean something.
    """
    pixel_size = {}
    tags = tif.pages.first.tags
    unit = getattr(tags.valueof('ResolutionUnit'), 'name', '')
    for axis, name in (('x', 'XResolution'), ('y', 'YResolution')):
        value = tags.valueof(name)
        if not isinstance(value, tuple) or not all(value):
            continue
        numerator, denominator = value
        if numerator == denominator:
            # the rational cancels out, so the ratio says nothing and
            # only the numerator is left to read. A resolution that
            # large is pixels per metre, whatever the unit tag claims,
            # which is filled in by default as readily as the value.
            size = convert_to_um(1 / numerator, 'm')
        elif unit == 'CENTIMETER':
            size = convert_to_um(denominator / numerator, 'cm')
        elif unit == 'INCH' and numerator / denominator not in (
                SCREEN_RESOLUTIONS):
            size = convert_to_um(denominator / numerator, 'inch')
        else:
            # no unit at all, or one of the resolutions a writer fills
            # in to say how large to print the image on a screen
            size = None
        if size is not None:
            pixel_size[axis] = size
    return pixel_size


def imagej_pixel_size(tif: TiffFile) -> Dict:
    """Return the pixel size ImageJ writes, which names its own unit."""
    metadata = tif.imagej_metadata
    unit = metadata.get('unit', '')
    pixel_size = {}
    for axis, name in (('x', 'XResolution'), ('y', 'YResolution')):
        value = tif.pages.first.tags.valueof(name)
        if isinstance(value, tuple) and all(value):
            size = convert_to_um(value[1] / value[0], unit)
            if size is not None:
                pixel_size[axis] = size
    spacing = convert_to_um(metadata.get('spacing'), unit)
    if spacing is not None:
        pixel_size['z'] = spacing
    return pixel_size


def stage_travel_mm(metadata: Dict) -> Optional[float]:
    """Return how far the stage travels, where a field names it.

    A vendor writes the travel into the model of the stage, as in
    TMCM-GYLZ-4000X-FIVE-110x110x69, and quotes it in millimetres.
    """
    travels = []
    for key, value, _ in walk_fields(metadata):
        if 'stage' not in normalise_name(key) or not isinstance(value, str):
            continue
        for match in STAGE_TRAVEL_PATTERN.finditer(value):
            travels.extend(float(group) for group in match.groups() if group)
    return max(travels) if travels else None


def implied_position_unit(value: Any,
                          travel_mm: Optional[float]) -> str:
    """Return the unit a position with no unit beside it is in.

    A vendor leaving the unit out means the SI one, except where the
    file also names the travel of its stage: a coordinate that fits
    inside that travel read as millimetres is millimetres, because no
    stage of that size reaches metres away from its own origin.
    """
    if travel_mm is None:
        return 'm'
    if isinstance(value, dict):
        keyed = {normalise_name(key): item for key, item in value.items()}
        value = next((keyed[key] for key in VALUE_KEYS if key in keyed), None)
    try:
        magnitude = abs(float(value))
    except (TypeError, ValueError):
        return 'm'
    return 'mm' if magnitude <= travel_mm else 'm'


def get_position_um(metadata: Dict) -> Dict[str, float]:
    """Return the stage position in micrometres, per axis.

    A vendor writes the position as a structure named after the stage,
    holding one field per axis, or as flat fields with the axis on the
    end of the name. Where no unit is given the value is read as metres,
    unless the file says how far the stage travels.
    """
    position = {}
    travel_mm = stage_travel_mm(metadata)
    for key, value, siblings in walk_fields(metadata):
        name, axis, _ = split_axis(normalise_name(key))
        if isinstance(value, dict) and name.endswith(
                POSITION_CONTAINER_NAMES) and axis is None:
            for axis_key, axis_value in value.items():
                axis = normalise_name(axis_key)
                if axis in ('x', 'y', 'z'):
                    coordinate = parse_quantity(
                        axis_value,
                        implied_position_unit(axis_value, travel_mm))
                    if coordinate is not None:
                        position.setdefault(axis, coordinate)
        elif axis is not None and name.endswith(POSITION_NAMES):
            coordinate = parse_quantity(
                value, sibling_unit(key, siblings)
                or implied_position_unit(value, travel_mm))
            if coordinate is not None:
                position.setdefault(axis, coordinate)
    return position
