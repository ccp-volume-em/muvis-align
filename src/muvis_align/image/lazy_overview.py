"""An overview of the sources (raw or pre-processed), one plane per section (or z of a volume), each built only when
it is viewed: the plane's own tiles read at the coarsest level no coarser than the plane and pasted at their position.
Opening a large project then shows image data as soon as the shapes."""
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import dask
import dask.array as da
import numpy as np
import xarray as xr
from multiview_stitcher import msi_utils, mv_graph
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.constants import default_interactive_preview_scale
from muvis_align.image.util import (build_source_stack_props, coarsest_level_within, combine_msims_as_channels,
                                   get_display_level, get_display_range, get_dtype_contrast_limits,
                                   wrap_sims_as_msims)
from muvis_align.util import parse_scale, pixel_size_to_um, thread_map

# tiles read at once while a plane is built: on a network filesystem each costs a round trip, not CPU
default_plane_readers = 32
# planes kept once built, the least recently viewed dropped first: all planes of a large stack would not fit
default_planes_max_bytes = 1_000_000_000
# sections on either side of a viewed one built in the background, so scrolling finds them ready
default_prefetch_radius = 4
# a plane coarsened until it fits, so a viewed section and its prefetched neighbours stay within the budget
default_plane_max_bytes = default_planes_max_bytes // (2 * default_prefetch_radius + 1)
# sources whose coarsest level sets a channel's contrast: read up front, before any plane
default_contrast_sources = 16
# decoded z chunks of volumes kept for the next planes: a chunk spans many planes, each would decode it again
default_chunks_max_bytes = 256_000_000


class SourceLevels:
    """A source's stored levels, read as its own raw arrays (no msim built) when it has them, else its msim's."""

    def __init__(self, source):
        self.source, self.pixel_sizes, self.dtype = source, source.pixel_sizes, source.dtype

    def level_data(self, level):
        data = self.source.data
        if data and level < len(data):
            return xr.DataArray(data[level], dims=list(self.source.dimension_order))
        return self.source.msim[f'scale{level}'].ds['image']


class MsimLevels:
    """The levels of a (lazy) msim, as a pre-processed source: its pixels computed only for the level read."""

    def __init__(self, msim):
        self.sims = [msi_utils.get_sim_from_msim(msim, scale=key) for key in msi_utils.get_sorted_scale_keys(msim)]
        self.pixel_sizes = [si_utils.get_spacing_from_sim(sim) for sim in self.sims]
        self.dtype = self.sims[0].dtype

    def level_data(self, level):
        return self.sims[level]


class SectionPlanes:
    """The (c, y, x) planes of a sectioned overview, each built on first request and kept within a memory budget, the
    planes around a requested one built in the background - none with one worker, which also reads in this thread."""

    def __init__(self, entries, shape, spacing, origin, dtype, nplanes, nchannels=1, workers=default_plane_readers,
                 max_bytes=default_planes_max_bytes, prefetch_radius=default_prefetch_radius):
        self.entries, self.shape, self.spacing, self.origin, self.dtype = entries, shape, spacing, origin, dtype
        self.nplanes, self.nchannels, self.workers, self.prefetch_radius = nplanes, nchannels, workers, prefetch_radius
        plane_bytes = nchannels * shape['y'] * shape['x'] * np.dtype(dtype).itemsize
        self.max_planes = max(2 * prefetch_radius + 1, int(max_bytes // max(plane_bytes, 1)))
        self._planes = OrderedDict()
        self._chunks = OrderedDict()
        self._building = {}
        self._lock = threading.Lock()
        self._background = ThreadPoolExecutor(max_workers=1) if workers > 1 else None

    def plane(self, index, prefetch=True):
        plane = self._get(index)
        if prefetch and self._background is not None:
            for offset in range(1, self.prefetch_radius + 1):
                for neighbour in (index + offset, index - offset):
                    if 0 <= neighbour < self.nplanes:
                        self._background.submit(self._get, neighbour)
        return plane

    def _get(self, index):
        with self._lock:
            if index in self._planes:
                self._planes.move_to_end(index)
                return self._planes[index]
            # a plane asked for while another thread builds it waits for that build
            event = self._building.get(index)
            if event is None:
                self._building[index] = event = threading.Event()
                owner = True
            else:
                owner = False
        if not owner:
            event.wait()
            with self._lock:
                if index in self._planes:
                    return self._planes[index]
            return self._get(index)
        try:
            plane = self._build(index)
            with self._lock:
                self._planes[index] = plane
                while len(self._planes) > self.max_planes:
                    self._planes.popitem(last=False)
        finally:
            with self._lock:
                self._building.pop(index, None)
            event.set()
        return plane

    def _build(self, index):
        plane = np.zeros((self.nchannels, self.shape['y'], self.shape['x']), dtype=self.dtype)
        entries = self.entries.get(index, [])
        datas = thread_map(self._source_plane, entries, self.workers)
        # in source order: a later source covers an earlier one
        for entry, data in zip(entries, datas):
            _paste(plane, data, entry['edge'], entry['spacing'], self.spacing)
        return plane


    def _source_plane(self, entry):
        """An entry's z of its level, (c, y, x): read with the rest of its z chunk, kept for the planes after it."""
        image = entry.get('image')
        if image is None:
            image = _level_image(entry['reader'], entry['level'])
        z_chunks = getattr(image.data, 'chunks', [None, None])[1]
        if not z_chunks or max(z_chunks) == 1:
            return _read(image.isel(z=entry['z_index']))
        bounds = np.cumsum((0,) + tuple(z_chunks))
        chunk = int(np.searchsorted(bounds, entry['z_index'], side='right')) - 1
        key = (id(entry['reader']), entry['level'], chunk)
        with self._lock:
            data = self._chunks.get(key)
            if data is not None:
                self._chunks.move_to_end(key)
        if data is None:
            data = _read(image.isel(z=slice(bounds[chunk], bounds[chunk + 1])))
            with self._lock:
                self._chunks[key] = data
                while sum(kept.nbytes for kept in self._chunks.values()) > default_chunks_max_bytes:
                    self._chunks.popitem(last=False)
        return data[:, entry['z_index'] - bounds[chunk]]


def _level_image(reader, level):
    """A reader's level as a lazy (c, z, y, x) DataArray, whatever its own dims: any other dim at its first index, a
    missing c or z added with size 1. A plain array is taken as (y, x)."""
    image = reader.level_data(level)
    if not isinstance(image, xr.DataArray):
        image = xr.DataArray(image, dims=['y', 'x'])
    image = image.isel({dim: 0 for dim in image.dims if dim not in 'czyx'})
    for dim in 'cz':
        if dim not in image.dims:
            image = image.expand_dims(dim)
    return image.transpose('c', 'z', 'y', 'x')


def _read(image):
    data = image.data
    if hasattr(data, 'compute'):
        # own (no) callbacks: dask's global ones, napari's slicing cache among them, are shared by all threads
        data = data.compute(scheduler='synchronous', callbacks=())
    return np.asarray(data)


def _contrast_limits(readers, max_sources=default_contrast_sources, workers=default_plane_readers):
    """Per channel, the display range (get_display_range) and level (get_display_level) of the coarsest level of up
    to `max_sources` readers spread evenly, None where there is nothing to set them: (ranges, levels), empty without
    readers."""
    readers = [reader for reader in readers if reader is not None]
    if not readers:
        return [], []
    sample = [readers[int(index)] for index in np.linspace(0, len(readers) - 1, min(len(readers), max_sources))]
    datas = thread_map(lambda reader: _read(_level_image(reader, len(reader.pixel_sizes) - 1)), sample, workers)
    channel_values = [np.concatenate([data[channel].ravel() for data in datas]) for channel in range(datas[0].shape[0])]
    ranges = [get_display_range(values) for values in channel_values]
    return ranges, [get_display_level(values, display_range) for values, display_range in zip(channel_values, ranges)]


def _source_indices(count, plane_spacing, edge, spacing, size):
    """Along one axis of `count` plane pixels: the first plane pixel a source covers, and from it on, the index of the
    source pixel under each plane pixel's centre. `edge` is the source's outer edge from the first plane pixel's
    centre; the tolerances keep a centre falling exactly on a pixel edge on the later pixel."""
    first = max(int(np.ceil(edge / plane_spacing - 1e-9)), 0)
    centres = np.arange(first, count) * plane_spacing
    indices = np.floor((centres - edge) / spacing + 1e-9).astype(int)
    return first, indices[indices < size]


def _paste(plane, data, edge, spacing, plane_spacing):
    """`data` (c, y, x) sampled nearest-neighbour onto `plane` (c, y, x) at any ratio of their pixel sizes: `edge` and
    `spacing` are the source's (y, x) outer edge and pixel size, in plane units from the plane's first pixel centre."""
    first_row, rows = _source_indices(plane.shape[1], plane_spacing['y'], edge[0], spacing[0], data.shape[1])
    first_col, cols = _source_indices(plane.shape[2], plane_spacing['x'], edge[1], spacing[1], data.shape[2])
    if len(rows) and len(cols):
        target = (slice(None), slice(first_row, first_row + len(rows)), slice(first_col, first_col + len(cols)))
        plane[target] = data[:, rows[:, np.newaxis], cols]


def _plane_spacing(preview_scale, level0_spacing):
    """The plane's pixel size: `preview_scale` as a pixel size with its unit, else as a factor of the sources' own."""
    scale = parse_scale(preview_scale, default=default_interactive_preview_scale)
    if isinstance(scale, str):
        return {dim: pixel_size_to_um(scale) for dim in 'yx'}
    return {dim: level0_spacing[dim] * scale for dim in 'yx'}


def lazy_section_overview(sources, translations, transforms, output_order, transform_key, z_scale=None,
                          preview_scale=default_interactive_preview_scale, max_plane_bytes=default_plane_max_bytes,
                          readers=None, label='Overview', channel_labels=None, workers=default_plane_readers):
    """The overview msim (c, z, y, x) of the sources, each z-plane computed on demand at `preview_scale`: 2D sources
    a plane per section, volumes a plane per z at the finest z spacing read. None for anything it cannot place
    faithfully: a rotated transform, 2D and 3D sources together, or sources with differing channels.
    `readers` (one a source, None to leave it out) give the pixels instead of the sources, at the sources' place.
    `channel_labels` (one a single-channel source) make each source a channel of its own, on the same grid, as fuse()
    does. attrs['contrast_limits'] holds each channel's display range, the dtype's where it has no pixels to set one;
    attrs['display_levels'] where its typical value shows in it (get_display_level).
    `workers`: tile reads at once (one: none in other threads, and no planes built ahead)."""
    if readers is None:
        readers = [SourceLevels(source) for source in sources]
    if not sources or all(reader is None for reader in readers):
        return None
    is_volume = sources[0].get_size().get('z', 1) > 1
    source_channels = sources[0].get_nchannels()
    if (any((source.get_size().get('z', 1) > 1) != is_volume or source.get_nchannels() != source_channels
            for source in sources) or (channel_labels and source_channels > 1)):
        logging.info(f'{label}: sources differ in dimensions or channels - no lazy overview')
        return None
    geometry = []
    for source, reader, translation, transform in zip(sources, readers, translations, transforms):
        props = build_source_stack_props(source, output_order, translation, transform, transform_key,
                                         z_scale=z_scale, promote_z=True)
        affine = np.asarray(props['transform'].squeeze()) if 'transform' in props else np.eye(4)
        if not np.allclose(affine[:-1, :-1], np.eye(affine.shape[0] - 1), atol=1e-6):
            logging.info(f'{label}: source transforms are not translations only - no lazy overview')
            return None
        vertices = mv_graph.get_vertices_from_stack_props(props)
        geometry.append((reader, props, vertices.min(axis=0), vertices.max(axis=0)))

    dims = ['z', 'y', 'x']
    lower = np.min([low for *_, low, _ in geometry], axis=0)
    upper = np.max([high for *_, high in geometry], axis=0)
    level0_spacing = {dim: float(np.median([props['spacing'][dim] for _, props, _, _ in geometry])) for dim in 'yx'}
    spacing = _plane_spacing(preview_scale, level0_spacing)
    dtype = next(reader.dtype for reader in readers if reader is not None)
    itemsize = np.dtype(dtype).itemsize
    # every channel has its planes within the one budget
    nchannels = len(channel_labels) if channel_labels else source_channels
    max_plane_bytes //= nchannels

    def plane_shape_at(spacing):
        return {dim: int(np.ceil((upper[axis] - lower[axis]) / spacing[dim])) + 1
                for axis, dim in enumerate(dims) if dim != 'z'}

    shape = plane_shape_at(spacing)
    while shape['y'] * shape['x'] * itemsize > max_plane_bytes:
        spacing = {dim: value * 2 for dim, value in spacing.items()}
        shape = plane_shape_at(spacing)

    placed = [(source_index, reader, props, low, coarsest_level_within(reader.pixel_sizes, spacing))
              for source_index, (reader, props, low, _) in enumerate(geometry) if reader is not None]
    if is_volume:
        z_spacing = min(float(reader.pixel_sizes[level].get('z', props['spacing']['z']))
                        for _, reader, props, _, level in placed)
        nplanes = int(np.ceil((upper[0] - lower[0]) / z_spacing)) + 1
        z_origin = float(lower[0])
    else:
        z_values = sorted({round(float(low[0]), 9) for *_, low, _ in geometry})
        z_spacing = float(np.min(np.diff(z_values))) if len(z_values) > 1 else float(z_scale or 1)
        nplanes = len(z_values)
        z_origin = z_values[0]

    channel_entries = [{} for _ in range(len(channel_labels) if channel_labels else 1)]
    for source_index, reader, props, low, level in placed:
        level_spacing = reader.pixel_sizes[level]
        # vertices are pixel centres of the source's level 0: its outer edge is half a pixel before the first
        edge = {dim: float(low[axis] - lower[axis] - props['spacing'][dim] / 2) for axis, dim in enumerate(dims)}
        entry = {'reader': reader, 'level': level, 'edge': (edge['y'], edge['x']),
                 'spacing': tuple(float(level_spacing.get(dim, props['spacing'][dim])) for dim in 'yx')}
        entries = channel_entries[source_index if channel_labels else 0]
        if is_volume:
            # built once here for its z size, then shared by the volume's planes
            entry['image'] = _level_image(reader, level)
            first_plane, z_indices = _source_indices(
                nplanes, z_spacing, edge['z'], float(level_spacing.get('z', props['spacing']['z'])),
                entry['image'].sizes['z'])
            for offset, z_index in enumerate(z_indices):
                entries.setdefault(first_plane + offset, []).append(entry | {'z_index': int(z_index)})
        else:
            entries.setdefault(z_values.index(round(float(low[0]), 9)), []).append(entry | {'z_index': 0})

    plane_shape = (1 if channel_labels else nchannels, shape['y'], shape['x'])
    c_coords = [channel.get('label', '') for channel in sources[0].get_channels()] if source_channels > 1 else None
    section_planes, sims = [], []
    for entries in channel_entries:
        planes = SectionPlanes(entries, shape, spacing, {'y': lower[1], 'x': lower[2]}, dtype, nplanes,
                               nchannels=plane_shape[0], max_bytes=default_planes_max_bytes // len(channel_entries),
                               workers=workers)
        stacked = da.stack([da.from_delayed(dask.delayed(planes.plane)(index), plane_shape, dtype=dtype)
                            for index in range(nplanes)], axis=1)
        section_planes.append(planes)
        sims.append(si_utils.get_sim_from_array(stacked, dims=['c'] + dims, c_coords=c_coords,
                                                scale={'z': z_spacing, 'y': spacing['y'], 'x': spacing['x']},
                                                translation={'z': z_origin, 'y': float(lower[1]),
                                                             'x': float(lower[2])},
                                                transform_key=transform_key))
    logging.info(f'{label}: {len(sources)} sources in {nplanes} {"planes" if is_volume else "sections"} of'
                 f' {shape["y"]}x{shape["x"]}{f" in {nchannels} channels" if nchannels > 1 else ""}'
                 f' at {spacing["x"]:.4g}um, each built when viewed')
    msims = wrap_sims_as_msims(sims)
    msim = combine_msims_as_channels(msims, channel_labels) if channel_labels else msims[0]
    msim.attrs['section_planes'] = section_planes
    dtype_limits = get_dtype_contrast_limits(dtype)
    channel_readers = [[reader] for reader in readers] if channel_labels else [readers]
    ranges, levels = [], []
    for group in channel_readers:
        group_ranges, group_levels = _contrast_limits(group, workers=workers)
        ranges += group_ranges or [None] * plane_shape[0]
        levels += group_levels or [None] * plane_shape[0]
    msim.attrs['contrast_limits'] = [display_range or dtype_limits for display_range in ranges]
    msim.attrs['display_levels'] = levels
    return msim
