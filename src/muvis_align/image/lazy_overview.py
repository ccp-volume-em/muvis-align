"""An overview of the raw sources, one plane per section, each built only when it is viewed: the plane's own tiles
read at their coarsest level and pasted at their position. Opening a large project then shows image data as soon as
the shapes, instead of pasting every source (minutes at 34k) before showing any."""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import dask
import dask.array as da
import numpy as np
from multiview_stitcher import mv_graph
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.image.util import build_source_stack_props, wrap_sims_as_msims

# the longest side of a section's plane, in pixels
default_plane_size = 4096
# tiles read at once while a plane is built: on a network filesystem each costs a round trip, not CPU
default_plane_readers = 32


def _coarsest_level_data(source, level):
    """The source's level as its own raw array (no msim built) when it has one, else its msim's."""
    data = source.data
    if data and level < len(data):
        return data[level]
    return source.get_level_data(level)


class SectionPlanes:
    """The planes of a sectioned overview, each built on first request and kept."""

    def __init__(self, entries, shape, spacing, origin, dtype, workers=default_plane_readers):
        self.entries, self.shape, self.spacing, self.origin, self.dtype = entries, shape, spacing, origin, dtype
        self.workers = workers
        self._planes = {}
        self._lock = threading.Lock()

    def plane(self, index):
        with self._lock:
            plane = self._planes.get(index)
        if plane is None:
            plane = self._build(index)
            with self._lock:
                self._planes.setdefault(index, plane)
        return plane

    def _build(self, index):
        plane = np.zeros((self.shape['y'], self.shape['x']), dtype=self.dtype)
        entries = self.entries.get(index, [])

        def read(entry):
            with dask.config.set(scheduler='synchronous'):
                return np.squeeze(np.asarray(_coarsest_level_data(entry['source'], entry['level'])))

        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(entries)))) as executor:
            datas = list(executor.map(read, entries))
        # in source order, as the pasted overview did: a later source covers an earlier one
        for entry, data in zip(entries, datas):
            _paste(plane, data, entry['start'], entry['stride'], entry['repeat'])
        return plane


def _paste(plane, data, start, stride, repeat):
    data = data[::stride[0], ::stride[1]]
    if repeat != (1, 1):
        data = np.repeat(np.repeat(data, repeat[0], axis=0), repeat[1], axis=1)
    target, source = [], []
    for axis in range(2):
        stop = min(start[axis] + data.shape[axis], plane.shape[axis])
        source.append(slice(max(-start[axis], 0), max(stop - start[axis], 0)))
        target.append(slice(max(start[axis], 0), max(stop, 0)))
    if all(piece.stop > piece.start for piece in target):
        plane[tuple(target)] = data[tuple(source)]


def lazy_section_overview(sources, translations, transforms, output_order, transform_key, z_scale=None,
                          max_plane_size=default_plane_size, label='Overview'):
    """The overview msim (z, y, x) of 2D, single-channel sources at one or more z, each z-plane computed on demand;
    None for anything it cannot place faithfully (a rotated transform, a 3D or multichannel source)."""
    if not sources or any(source.get_nchannels() > 1 or source.get_size().get('z', 1) > 1 for source in sources):
        return None
    geometry = []
    for source, translation, transform in zip(sources, translations, transforms):
        level = len(source.shapes) - 1
        props = build_source_stack_props(source, output_order, translation, transform, transform_key,
                                         z_scale=z_scale, level=level, promote_z=True)
        affine = np.asarray(props['transform'].squeeze()) if 'transform' in props else np.eye(4)
        if not np.allclose(affine[:-1, :-1], np.eye(affine.shape[0] - 1), atol=1e-6):
            logging.info(f'{label}: source transforms are not translations only - no lazy overview')
            return None
        vertices = mv_graph.get_vertices_from_stack_props(props)
        geometry.append((source, level, props, vertices.min(axis=0), vertices.max(axis=0)))

    dims = ['z', 'y', 'x']
    lower = np.min([low for *_, low, _ in geometry], axis=0)
    upper = np.max([high for *_, high in geometry], axis=0)
    spacing = {dim: float(np.median([props['spacing'][dim] for _, _, props, _, _ in geometry])) for dim in 'yx'}
    shape = {dim: int(np.ceil((upper[axis] - lower[axis]) / spacing[dim])) + 1
             for axis, dim in enumerate(dims) if dim != 'z'}
    while max(shape.values()) > max_plane_size:
        for dim in 'yx':
            spacing[dim] *= 2
            shape[dim] = max(int(np.ceil(shape[dim] / 2)), 1)

    z_values = sorted({round(float(low[0]), 9) for *_, low, _ in geometry})
    entries = {}
    for source, level, props, low, _ in geometry:
        start, stride, repeat = [], [], []
        for axis, dim in ((1, 'y'), (2, 'x')):
            factor = spacing[dim] / props['spacing'][dim]
            stride.append(max(int(round(factor)), 1) if factor >= 1 else 1)
            repeat.append(max(int(round(1 / factor)), 1) if factor < 1 else 1)
            start.append(int(round((low[axis] - lower[axis]) / spacing[dim])))
        entries.setdefault(z_values.index(round(float(low[0]), 9)), []).append(
            {'source': source, 'level': level, 'start': tuple(start), 'stride': tuple(stride),
             'repeat': tuple(repeat)})

    dtype = sources[0].dtype
    planes = SectionPlanes(entries, shape, spacing, {'y': lower[1], 'x': lower[2]}, dtype)
    plane_shape = (shape['y'], shape['x'])
    stacked = da.stack([da.from_delayed(dask.delayed(planes.plane)(index), plane_shape, dtype=dtype)
                        for index in range(len(z_values))])
    z_spacing = float(np.min(np.diff(z_values))) if len(z_values) > 1 else float(z_scale or 1)
    sim = si_utils.get_sim_from_array(stacked, dims=dims,
                                      scale={'z': z_spacing, 'y': spacing['y'], 'x': spacing['x']},
                                      translation={'z': z_values[0], 'y': float(lower[1]), 'x': float(lower[2])},
                                      transform_key=transform_key)
    logging.info(f'{label}: {len(sources)} sources in {len(z_values)} sections of {shape["y"]}x{shape["x"]},'
                 f' each built when viewed')
    msim = wrap_sims_as_msims([sim])[0]
    msim.attrs['section_planes'] = planes
    return msim
