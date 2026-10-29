import os
import threading
from xml.etree import ElementTree

import dask.array as da
from dask.base import tokenize
from napari_meta_tiff.metadata import get_extra_metadata, get_pixel_size_um, get_position_um, get_rotation_deg
import zarr
from ome_zarr.scale import Scaler
from tifffile import TiffWriter, tifffile

try:
    # reused so our reads never disagree with ngff_zarr's; private, so guarded: without them the
    # fast reads decline and callers fall back to ngff_zarr
    from ngff_zarr.tiff_to_ngff_image import _map_tiff_axes_to_ngff as map_tiff_axes_to_ngff
    from ngff_zarr.tiff_to_ngff_image import _normalize_unit as normalize_ome_unit
    from ngff_zarr.tiff_to_ngff_image import _reshape_tiff_for_channels as reshape_tiff_for_channels
except ImportError:  # pragma: no cover - depends on the installed ngff_zarr
    map_tiff_axes_to_ngff = None
    reshape_tiff_for_channels = None

    def normalize_ome_unit(unit):
        return unit

from muvis_align.constants import default_chunk_size
from muvis_align.image.color_conversion import hexrgb_to_rgba, rgba_to_int
from muvis_align.util import *


def load_tiff(filename):
    return tifffile.imread(filename)



def extract_ome_translation_from_xml(ome_xml, **kwargs):
    """The first Image's first Plane position, in um - see extract_ome_image_metadata()."""
    return extract_ome_image_metadata(ome_xml, **kwargs)['position']


def extract_ome_image_metadata(ome_xml, chunk_size=64 * 1024):
    """The first Image's position (um), physical pixel size and units, channel names and colours, plus the
    document's Creator.

    Pull-parsed in chunks and stopped once settled: a multi-file OME-TIFF repeats the whole dataset's XML in
    every file. `position` is {} when the XML describes more than one Image (positions then come from
    source_metadata); scale, units and channels come from the first Image, as in ngff_zarr.
    """
    parser = ElementTree.XMLPullParser(events=['start'])
    position, scale, units = {}, {}, {}
    channel_names, channel_colors = [], []
    creator = ''
    images = 0
    for start in range(0, len(ome_xml), chunk_size):
        parser.feed(ome_xml[start:start + chunk_size])
        for _event, element in parser.read_events():
            # strip the OME namespace, e.g. '{http://...}Plane'
            tag = element.tag.rpartition('}')[2]
            if tag == 'OME':
                creator = element.get('Creator', '')
            elif tag == 'Image':
                images += 1
                if images > 1:
                    # nothing after a second Image belongs to the first
                    return {'position': {}, 'scale': scale, 'units': units,
                            'channel_names': channel_names, 'channel_colors': channel_colors,
                            'creator': creator}
            elif tag == 'Pixels':
                for dim in 'XYZ':
                    value = element.get(f'PhysicalSize{dim}')
                    if value is not None:
                        try:
                            scale[dim.lower()] = float(value)
                        except ValueError:
                            continue
                        unit = normalize_ome_unit(element.get(f'PhysicalSize{dim}Unit'))
                        if unit is not None:
                            units[dim.lower()] = unit
            elif tag == 'Channel':
                channel_names.append(element.get('Name', ''))
                channel_colors.append(element.get('Color'))
            elif tag == 'Plane' and not position:
                for dim in ['X', 'Y', 'Z']:
                    key = f'Position{dim}'
                    value = element.get(key)
                    if value is not None:
                        position[dim.lower()] = convert_to_um(float(value),
                                                              element.get(f'{key}Unit', 'um'))
    return {'position': position, 'scale': scale, 'units': units,
            'channel_names': channel_names, 'channel_colors': channel_colors, 'creator': creator}


def read_tiff_source_metadata(filename):
    """The metadata an ImageSource needs from a TIFF (per-level shapes and pixel sizes, dtype, dimension
    order, origin, channels, creator), read off tifffile without building a zarr store or dask arrays.

    Matches ngff_zarr.tiff_file_to_ngff_images(), reusing its axis mapping so dims can never be mis-ordered.
    Returns None if it cannot be done faithfully, leaving the caller on ngff_zarr's path.
    """
    if map_tiff_axes_to_ngff is None:
        return None
    with tifffile.TiffFile(filename) as tif:
        series = tif.series[0]
        axes = series.axes
        if not axes:
            return None
        level_shapes = [level.shape for level in series.levels]
        dtype = series.dtype
        # .is_ome/.ome_metadata read only the description tag, not the series structure
        ome = (extract_ome_image_metadata(tif.ome_metadata)
               if tif.is_ome and tif.ome_metadata is not None else None)
        if ome is None:
            # no OME: the pixel size and stage position wherever a vendor wrote them, in um, and its rotation
            vendor_metadata = get_extra_metadata(tif)
            ome = {'scale': get_pixel_size_um(tif, vendor_metadata, dict(zip(axes.lower(), series.shape))),
                   'position': get_position_um(vendor_metadata),
                   'rotation': get_rotation_deg(vendor_metadata)}

    dims, _, _, _ = map_tiff_axes_to_ngff(axes, series.shape)
    shapes = [map_tiff_axes_to_ngff(axes, shape)[1] for shape in level_shapes]
    spatial_dims = [dim for dim in dims if dim in 'xyz']
    if not shapes or not spatial_dims:
        return None

    # as ngff_zarr, OME PhysicalSize* with 1.0 for any spatial dim not given; without OME the vendor's
    # own fields or the TIFF resolution, which ngff_zarr ignores
    ome_scale = (ome or {}).get('scale') or {}
    ome_units = (ome or {}).get('units') or {}
    axis_of = {dim: index for index, dim in enumerate(dims)}
    pixel_sizes = []
    for shape in shapes:
        # by the realised extent ratio, not the nominal downsample factor, as ngff_zarr
        pixel_sizes.append({
            dim: convert_to_um(ome_scale.get(dim, 1.0) * shapes[0][axis_of[dim]] / shape[axis_of[dim]],
                               ome_units.get(dim, 'um'))
            for dim in spatial_dims})

    channels = []
    for index, (name, color) in enumerate(zip((ome or {}).get('channel_names') or [],
                                               (ome or {}).get('channel_colors') or [])):
        # a <Channel> without Name parses to '': generate a label so it never shows blank
        channel = {'label': name or f'channel {index}'}
        if color:
            channel['color'] = hexrgb_to_rgba(color)
        channels.append(channel)

    return {'dimension_order': ''.join(dims), 'shapes': shapes, 'dtype': dtype,
            'pixel_sizes': pixel_sizes, 'position': (ome or {}).get('position') or {},
            'rotation': (ome or {}).get('rotation'),
            'channels': channels, 'creator': (ome or {}).get('creator') or ''}


def read_tiff_creator(filename):
    """The OME-XML Creator attribute alone, for the ngff_zarr fallback path."""
    with tifffile.TiffFile(filename) as tif:
        if not tif.is_ome or tif.ome_metadata is None:
            return ''
        return extract_ome_image_metadata(tif.ome_metadata).get('creator', '')


def contiguous_layout(page, shape):
    """(file offset, byte order) of a level stored as one uncompressed run of rows in C order, as its array has
    them (`shape`), or None: only then can a slice be read as a byte range of the file."""
    is_plain = (page.compression == 1 and page.predictor == 1 and page.fillorder == 1 and not page.is_tiled
                and page.planarconfig == 1 and page.bitspersample == 8 * np.dtype(page.dtype).itemsize)
    offsets, counts = page.dataoffsets, page.databytecounts
    runs_on = all(offset + count == next_offset for offset, count, next_offset in zip(offsets, counts, offsets[1:]))
    if not (is_plain and runs_on and tuple(page.shape) == tuple(shape)
            and sum(counts) == int(np.prod(shape)) * np.dtype(page.dtype).itemsize):
        return None
    return int(offsets[0]), page.parent.byteorder


class PicklableTiffLevel:
    """A TIFF pyramid level's zarr array that pickles as its file name and level, reopening the file
    where it is unpickled: tifffile's store holds a lock, so worker processes could not be sent one.

    A level stored as one uncompressed run of rows is read as the byte range of the rows asked for, in the
    calling thread: through zarr every read of the process runs on zarr's one event-loop thread."""

    def __init__(self, filename, path, array=None, layout=None):
        self.filename, self.path, self._array, self.layout = filename, path, array, layout
        self.shape, self.dtype, self.chunks = array.shape, array.dtype, array.chunks
        self.ndim = len(self.shape)

    _open_lock = threading.Lock()

    @property
    def array(self):
        # locked: dask threads opening it concurrently could read from a closed file
        if self._array is None:
            with self._open_lock:
                if self._array is None:
                    with tifffile.TiffFile(self.filename) as tif:
                        array = zarr.open_group(store=tif.aszarr(series=0, multiscales=True), mode='r')[self.path]
                    # published once the file is closed: a thread reading it before would keep a handle about to close
                    self._array = array
        return self._array

    def __getitem__(self, key):
        if self.layout is None:
            return self.array[key]
        key = key if isinstance(key, tuple) else (key,)
        first = key[0] if key else slice(None)
        if isinstance(first, (int, np.integer)):
            rows = slice(int(first) % self.shape[0], int(first) % self.shape[0] + 1)
            rest = (0,) + key[1:]
        elif isinstance(first, slice) and (first.step or 1) > 0:
            # a strided slice reads only its own rows: a coarse level's chunk is a few rows, not its whole span
            start, stop, step = first.indices(self.shape[0])
            rows = slice(start, max(stop, start), step)
            rest = (slice(None),) + key[1:]
        else:
            rows = slice(0, self.shape[0])
            rest = key
        return self._read_rows(rows)[rest]

    def _read_rows(self, rows):
        offset, byteorder = self.layout
        row_shape = self.shape[1:]
        dtype = np.dtype(self.dtype).newbyteorder(byteorder)
        row_bytes = int(np.prod(row_shape)) * dtype.itemsize
        step = rows.step or 1
        starts = range(rows.start, rows.stop, step)
        data = np.empty((len(starts),) + tuple(row_shape), dtype=dtype)
        if len(starts) == 0:
            return data.astype(self.dtype, copy=False)
        # consecutive rows in one read, strided ones each on their own
        runs = [(rows.start, data)] if step == 1 else [(start, data[index]) for index, start in enumerate(starts)]
        with open(self.filename, 'rb') as file:
            for start, target in runs:
                file.seek(offset + start * row_bytes)
                if file.readinto(memoryview(target).cast('B')) != target.nbytes:
                    raise OSError(f'{self.filename}: fewer bytes than level {self.path} holds')
        return data.astype(self.dtype, copy=False)

    def __reduce__(self):
        return _unpickle_tiff_level, (self.filename, self.path, self.shape, self.dtype, self.chunks, self.layout)


def _unpickle_tiff_level(filename, path, shape, dtype, chunks, layout=None):
    level = PicklableTiffLevel.__new__(PicklableTiffLevel)
    level.filename, level.path, level._array, level.layout = filename, path, None, layout
    level.shape, level.dtype, level.chunks, level.ndim = shape, dtype, chunks, len(shape)
    return level


def read_tiff_level_arrays(filename):
    """One dask array per pyramid level of the file's first series, off tifffile's zarr store, as ngff_zarr
    builds them but without rebuilding the metadata. Returns None if it cannot be done faithfully.

    Wrapped in dask because slicing a zarr array reads at once, which would make strided level synthesis a
    full decode at load.
    """
    if map_tiff_axes_to_ngff is None or reshape_tiff_for_channels is None:
        return None
    with tifffile.TiffFile(filename) as tif:
        series = tif.series[0]
        axes = series.axes
        if not axes:
            return None
        # multiscales=True: a group even for one level, levels ordered finest first
        group = zarr.open_group(store=tif.aszarr(series=0, multiscales=True), mode='r')
        attributes = group.attrs.get('ome', group.attrs)
        try:
            datasets = attributes['multiscales'][0]['datasets']
        except (KeyError, IndexError, TypeError):
            return None
        arrays = []
        for dataset in datasets:
            array = group[dataset['path']]
            page = series.levels[int(dataset['path'])].keyframe if dataset['path'].isdigit() else None
            layout = contiguous_layout(page, array.shape) if page is not None else None
            arrays.append(PicklableTiffLevel(filename, dataset['path'], array, layout))
        # tif closes on exit: the store reopens the file per chunk read, so no handle is held per source
        dims, _, channel_indices, dropped_indices = map_tiff_axes_to_ngff(axes, series.shape)

    datas = []
    # names from file, size, mtime and level: cheaper than tokenizing the array, and new when rewritten in place
    stat = os.stat(filename)
    for level, array in enumerate(arrays):
        # NGFF shape: unsupported axes dropped, channel-like axes ('S' included) flattened onto 'c'
        shape = map_tiff_axes_to_ngff(axes, array.shape)[1]
        chunks = array.chunks
        if array.layout is not None:
            # any row span reads in one go: a one-row-strip file's own chunks would make a block thousands of reads
            chunks = (min(default_chunk_size, array.shape[0]),) + tuple(array.shape[1:])
        # meta given, so dask does not probe the dtype with a read
        data = da.from_array(array, chunks=chunks, meta=np.empty((0,) * array.ndim, dtype=array.dtype),
                             name='tiff-' + tokenize(filename, stat.st_size, stat.st_mtime_ns, level))
        if tuple(data.shape) != tuple(shape):
            data = reshape_tiff_for_channels(data, axes, tuple(array.shape), dims, tuple(shape),
                                             list(channel_indices), list(dropped_indices))
        if tuple(data.shape) != tuple(shape):
            # decline rather than return levels disagreeing with the metadata
            return None
        datas.append(data)
    return datas


def save_tiff(filename, data, dimension_order=None, pixel_size=None, tile_size=(default_chunk_size, default_chunk_size),
              compression='LZW'):
    _, resolution, resolution_unit = create_tiff_metadata(pixel_size, dimension_order)
    tifffile.imwrite(filename, data, tile=tile_size, compression=compression,
                     resolution=resolution, resolutionunit=resolution_unit)


def save_ome_tiff(filename, data, dimension_order, pixel_size, channels=[], positions=[], rotation=None,
                  tile_size=None, compression=None, pyramid_downsample=2, npyramid_add=None):

    ome_metadata, resolution0, resolution_unit0 = create_tiff_metadata(pixel_size, dimension_order,
                                                                       channels, positions, rotation, is_ome=True)
    # maximum size (w/o compression)
    max_size = data.size * data.itemsize
    size = max_size
    if pyramid_downsample is not None:
        scaler = Scaler(downscale=pyramid_downsample, max_layer=npyramid_add)
        npyramid_add = scaler.max_layer
        for i in range(npyramid_add):
            size //= (scaler.downscale ** 2)
            max_size += size
    bigtiff = (max_size > 2 ** 32)

    if tile_size:
        tile_size = tile_size[-2:]  # assume order zyx (inversed xyz)
        shape_yx = [data.shape[dimension_order.index(dim)] for dim in 'yx']
        if np.any(np.array(tile_size) > np.array(shape_yx)):
            tile_size = None

    with TiffWriter(filename, bigtiff=bigtiff) as writer:
        for i in range(npyramid_add + 1):
            if i == 0:
                subifds = npyramid_add
                subfiletype = None
                metadata = ome_metadata
                resolution = resolution0[:2]
                resolutionunit = resolution_unit0
            else:
                subifds = None
                subfiletype = 1
                metadata = None
                resolution = None
                resolutionunit = None
                data = scaler.resize_image(data)
                data.rechunk()
            writer.write(data, subifds=subifds, subfiletype=subfiletype,
                         tile=tile_size, compression=compression,
                         resolution=resolution, resolutionunit=resolutionunit, metadata=metadata)


def create_tiff_metadata(pixel_size, dimension_order=None, channels=[], positions=[], rotation=None, is_ome=False):
    ome_metadata = None
    resolution = None
    resolution_unit = None

    if pixel_size is not None:
        resolution_unit = 'CENTIMETER'
        resolution = [1e4 / size for size in dict_to_xyz(pixel_size, 'xy')]

    if is_ome:
        ome_metadata = {'Creator': 'muvis-align'}
        if dimension_order is not None:
            #ome_metadata['DimensionOrder'] = dimension_order[::-1].upper()
            ome_metadata['axes'] = dimension_order.upper()
        ome_channels = []
        if pixel_size is not None:
            ome_metadata['PhysicalSizeX'] = pixel_size['x']
            ome_metadata['PhysicalSizeXUnit'] = 'µm'
            ome_metadata['PhysicalSizeY'] = pixel_size['y']
            ome_metadata['PhysicalSizeYUnit'] = 'µm'
            if 'z' in pixel_size:
                ome_metadata['PhysicalSizeZ'] = pixel_size['z']
                ome_metadata['PhysicalSizeZUnit'] = 'µm'
        if positions is not None and len(positions) > 0:
            plane_metadata = {}
            plane_metadata['PositionX'] = [position['x'] for position in positions]
            plane_metadata['PositionXUnit'] = ['µm' for _ in positions]
            plane_metadata['PositionY'] = [position['y'] for position in positions]
            plane_metadata['PositionYUnit'] = ['µm' for _ in positions]
            if 'z' in positions[0]:
                plane_metadata['PositionZ'] = [position['z'] for position in positions]
                plane_metadata['PositionZUnit'] = ['µm' for _ in positions]
            ome_metadata['Plane'] = plane_metadata
        if rotation is not None:
            ome_metadata['StructuredAnnotations'] = {'CommentAnnotation': {'Value': f'Angle: {rotation} degrees'}}
        for channeli, channel in enumerate(channels):
            ome_channel = {'Name': channel.get('label', str(channeli))}
            if 'color' in channel:
                ome_channel['Color'] = rgba_to_int(channel['color'])
            ome_channels.append(ome_channel)
        if ome_channels:
            ome_metadata['Channel'] = ome_channels
    return ome_metadata, resolution, resolution_unit
