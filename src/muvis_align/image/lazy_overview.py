"""An overview of the sources (raw or pre-processed), one plane per section, each built only when it is viewed: the
plane's own tiles read at the coarsest level no coarser than the plane and pasted at their position. Opening a large
project then shows image data as soon as the shapes, instead of pasting every source (minutes at 34k) before any."""
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import dask
import dask.array as da
import numpy as np
from multiview_stitcher import msi_utils, mv_graph
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.constants import default_interactive_preview_scale
from muvis_align.image.util import build_source_stack_props, wrap_sims_as_msims
from muvis_align.util import parse_scale, pixel_size_to_um

# tiles read at once while a plane is built: on a network filesystem each costs a round trip, not CPU
default_plane_readers = 32
# planes kept once built, the least recently viewed dropped first: all 1081 of a 34k-source stack would be ~17GB
default_planes_max_bytes = 1_000_000_000
# sections on either side of a viewed one built in the background, so scrolling finds them ready
default_prefetch_radius = 4
# a plane coarsened until it fits, so a viewed section and its prefetched neighbours stay within the budget
default_plane_max_bytes = default_planes_max_bytes // (2 * default_prefetch_radius + 1)


class SourceLevels:
    """A source's stored levels, read as its own raw arrays (no msim built) when it has them, else its msim's."""

    def __init__(self, source):
        self.source, self.pixel_sizes, self.dtype = source, source.pixel_sizes, source.dtype

    def level_data(self, level):
        data = self.source.data
        if data and level < len(data):
            return data[level]
        return self.source.get_level_data(level)


class MsimLevels:
    """The levels of a (lazy) msim, as a pre-processed source: its pixels computed only for the level read."""

    def __init__(self, msim):
        self.sims = [msi_utils.get_sim_from_msim(msim, scale=key) for key in msi_utils.get_sorted_scale_keys(msim)]
        self.pixel_sizes = [si_utils.get_spacing_from_sim(sim) for sim in self.sims]
        self.dtype = self.sims[0].dtype

    def level_data(self, level):
        return self.sims[level].data


class SectionPlanes:
    """The planes of a sectioned overview, each built on first request and kept within a memory budget, the sections
    around a requested one built in the background."""

    def __init__(self, entries, shape, spacing, origin, dtype, nplanes, workers=default_plane_readers,
                 max_bytes=default_planes_max_bytes, prefetch_radius=default_prefetch_radius):
        self.entries, self.shape, self.spacing, self.origin, self.dtype = entries, shape, spacing, origin, dtype
        self.nplanes, self.workers, self.prefetch_radius = nplanes, workers, prefetch_radius
        plane_bytes = shape['y'] * shape['x'] * np.dtype(dtype).itemsize
        self.max_planes = max(2 * prefetch_radius + 1, int(max_bytes // max(plane_bytes, 1)))
        self._planes = OrderedDict()
        self._building = {}
        self._lock = threading.Lock()
        self._background = ThreadPoolExecutor(max_workers=1)

    def plane(self, index, prefetch=True):
        plane = self._get(index)
        if prefetch:
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
        plane = np.zeros((self.shape['y'], self.shape['x']), dtype=self.dtype)
        entries = self.entries.get(index, [])

        def read(entry):
            with dask.config.set(scheduler='synchronous'):
                return np.squeeze(np.asarray(entry['reader'].level_data(entry['level'])))

        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(entries)))) as executor:
            datas = list(executor.map(read, entries))
        # in source order, as the pasted overview did: a later source covers an earlier one
        for entry, data in zip(entries, datas):
            _paste(plane, data, entry['edge'], entry['spacing'], self.spacing)
        return plane


def _source_indices(count, plane_spacing, edge, spacing, size):
    """The first plane pixel a source covers and, from there on, the source pixel under each plane pixel's centre."""
    first = max(int(np.ceil(edge / plane_spacing - 1e-9)), 0)
    centres = np.arange(first, count) * plane_spacing
    indices = np.floor((centres - edge) / spacing + 1e-9).astype(int)
    return first, indices[indices < size]


def _paste(plane, data, edge, spacing, plane_spacing):
    """`data` nearest-neighbour sampled onto the plane: its first pixel's outer edge at `edge` (plane units from the
    plane's first pixel centre), any ratio of its spacing to the plane's."""
    first_row, rows = _source_indices(plane.shape[0], plane_spacing['y'], edge[0], spacing[0], data.shape[0])
    first_col, cols = _source_indices(plane.shape[1], plane_spacing['x'], edge[1], spacing[1], data.shape[1])
    if len(rows) and len(cols):
        plane[first_row:first_row + len(rows), first_col:first_col + len(cols)] = data[np.ix_(rows, cols)]


def _plane_spacing(preview_scale, level0_spacing):
    """The plane's pixel size: `preview_scale` as a pixel size with its unit, else as a factor of the sources' own."""
    scale = parse_scale(preview_scale, default=default_interactive_preview_scale)
    if isinstance(scale, str):
        return {dim: pixel_size_to_um(scale) for dim in 'yx'}
    return {dim: level0_spacing[dim] * scale for dim in 'yx'}


def _coarsest_level_within(reader, spacing):
    """The reader's coarsest level no coarser than `spacing`, else its finest."""
    best = 0
    for level, pixel_size in enumerate(reader.pixel_sizes):
        if all(pixel_size.get(dim, 0) <= spacing[dim] * (1 + 1e-6) for dim in 'yx'):
            best = level
    return best


def lazy_section_overview(sources, translations, transforms, output_order, transform_key, z_scale=None,
                          preview_scale=default_interactive_preview_scale, max_plane_bytes=default_plane_max_bytes,
                          readers=None, label='Overview'):
    """The overview msim (z, y, x) of 2D, single-channel sources at one or more z, each z-plane computed on demand at
    `preview_scale`; None for anything it cannot place faithfully (a rotated transform, a 3D or multichannel source).
    `readers` (one a source, None to leave it out) give the pixels instead of the sources, at the sources' place."""
    if readers is None:
        readers = [SourceLevels(source) for source in sources]
    if all(reader is None for reader in readers):
        return None
    if not sources or any(source.get_nchannels() > 1 or source.get_size().get('z', 1) > 1 for source in sources):
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

    def plane_shape_at(spacing):
        return {dim: int(np.ceil((upper[axis] - lower[axis]) / spacing[dim])) + 1
                for axis, dim in enumerate(dims) if dim != 'z'}

    shape = plane_shape_at(spacing)
    while shape['y'] * shape['x'] * itemsize > max_plane_bytes:
        spacing = {dim: value * 2 for dim, value in spacing.items()}
        shape = plane_shape_at(spacing)

    z_values = sorted({round(float(low[0]), 9) for *_, low, _ in geometry})
    entries = {}
    for reader, props, low, _ in geometry:
        if reader is not None:
            level = _coarsest_level_within(reader, spacing)
            level_spacing = [float(reader.pixel_sizes[level].get(dim, props['spacing'][dim])) for dim in 'yx']
            # vertices are pixel centres of the source's level 0: its outer edge is half a pixel before the first
            edge = tuple(float(low[axis] - lower[axis] - props['spacing'][dim] / 2)
                         for axis, dim in ((1, 'y'), (2, 'x')))
            entries.setdefault(z_values.index(round(float(low[0]), 9)), []).append(
                {'reader': reader, 'level': level, 'edge': edge, 'spacing': tuple(level_spacing)})

    planes = SectionPlanes(entries, shape, spacing, {'y': lower[1], 'x': lower[2]}, dtype, len(z_values))
    plane_shape = (shape['y'], shape['x'])
    stacked = da.stack([da.from_delayed(dask.delayed(planes.plane)(index), plane_shape, dtype=dtype)
                        for index in range(len(z_values))])
    z_spacing = float(np.min(np.diff(z_values))) if len(z_values) > 1 else float(z_scale or 1)
    sim = si_utils.get_sim_from_array(stacked, dims=dims,
                                      scale={'z': z_spacing, 'y': spacing['y'], 'x': spacing['x']},
                                      translation={'z': z_values[0], 'y': float(lower[1]), 'x': float(lower[2])},
                                      transform_key=transform_key)
    logging.info(f'{label}: {len(sources)} sources in {len(z_values)} sections of {shape["y"]}x{shape["x"]}'
                 f' at {spacing["x"]:.4g}um, each built when viewed')
    msim = wrap_sims_as_msims([sim])[0]
    msim.attrs['section_planes'] = planes
    return msim
