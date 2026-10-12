import logging
from math import ceil

import numpy as np
from multiview_stitcher import msi_utils, param_utils
from multiview_stitcher import spatial_image_utils as si_utils

from muvis_align.constants import default_transform_key
from muvis_align.image.util import (build_source_redimensioned_msim, build_missing_pyramid_levels,
                                    calc_pyramid_level_factors, complete_translation, get_properties_from_transform)
from muvis_align.util import (find_all_numbers, split_numeric_dict, eval_context, check_contains_value,
                              is_source_value, load_sbemimage_best_config, adjust_sbemimage_properties,
                              find_sbemimage_meta_dir, xyz_to_dict)


class ImageSource:
    default_physical_unit = 'µm'

    def __init__(self, filename, source_metadata=None, file_label=None, index=None,
                 transform_key=default_transform_key):
        self.filename = filename
        self.file_label = file_label
        self.index = index
        self.transform_key = transform_key
        self.dimension_order = ''
        self.is_rgb = False
        self.shapes = []
        self.shape = []
        self.dtype = None
        self.pixel_sizes = []
        self.pixel_size = {}
        self.scale_factors = []
        self.position = {}
        self.rotation = None
        self.channels = []
        self.creator = ''
        self._data = []
        self._metadata = None
        self._msim = None
        self._redimensioned_msims = {}
        self._data_loaded = False
        self.init_metadata()
        self.fix_metadata(source_metadata)
        # settle synthesized levels' metadata up front so get_shape/get_pixel_size answer the same
        # whenever called; their arrays are built on first access to self.data
        self._add_missing_pyramid_level_metadata()

    @property
    def msim(self):
        if self._msim is None:
            self._build_msim()
        return self._msim

    @property
    def data(self):
        """One array per pyramid level, loaded on first access, never during __init__: project load
        only reads metadata."""
        if not self._data_loaded:
            self._data_loaded = True
            self._load_data()
            self._add_missing_pyramid_level_data()
        return self._data

    @data.setter
    def data(self, value):
        # marks data loaded so a later _load_data() cannot overwrite what a subclass set
        self._data_loaded = True
        self._data = value

    @property
    def metadata(self):
        """The source's full-format metadata, read lazily (see _read_metadata)."""
        if self._metadata is None:
            self._metadata = self._read_metadata()
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def _read_metadata(self):
        return {}

    def get_msim(self, output_order, from_level=0, ends_only=False):
        """self.msim redimensioned to `output_order`, cached per (output_order, from_level, ends_only) as it
        never depends on per-run geometry. `from_level` starts the pyramid coarser than level 0, `ends_only`
        keeps just that level and the coarsest.
        """
        key = (output_order, from_level, ends_only)
        if key not in self._redimensioned_msims:
            self._redimensioned_msims[key] = build_source_redimensioned_msim(
                self, output_order, from_level=from_level, ends_only=ends_only)
        return self._redimensioned_msims[key]

    def init_metadata(self):
        raise NotImplementedError("Image source should implement init_metadata() to initialize metadata,"
                                  " populating self.data with one dask array per pyramid level")

    def fix_metadata(self, source_metadata=None):
        source_metadata = self.select_source_metadata(source_metadata)
        if isinstance(source_metadata, dict):
            filename_numeric = find_all_numbers(self.filename)
            filename_dict = {key: int(value) for key, value in split_numeric_dict(self.filename).items()}
            context = {'filename_numeric': filename_numeric, 'fn': filename_numeric} | filename_dict
            if 'position' in source_metadata:
                translation = source_metadata['position']
                for dim in 'xyz':
                    if dim in translation:
                        if not is_source_value(translation[dim]):
                            self.position[dim] = eval_context(translation, dim, 0, context)
                        if check_contains_value(translation[dim], 'invert') and dim in self.position:
                            if isinstance(self.position[dim], (tuple, list)):
                                self.position[dim] = -self.position[dim][0], self.position[dim][1]
                            else:
                                self.position[dim] = -self.position[dim]
            if 'scale' in source_metadata:
                scale = source_metadata['scale']
                if 'x' in scale:
                    if not check_contains_value(scale['x'], 'source'):
                        self.pixel_sizes[0]['x'] = eval_context(scale, 'x', 1, context)
                if 'y' in scale:
                    if not check_contains_value(scale['y'], 'source'):
                        self.pixel_sizes[0]['y'] = eval_context(scale, 'y', 1, context)
                if 'z' in scale:
                    if not check_contains_value(scale['z'], 'source'):
                        self.pixel_sizes[0]['z'] = eval_context(scale, 'z', 1, context)
            if 'rotation' in source_metadata:
                if not is_source_value(source_metadata['rotation']):
                    self.rotation = eval_context(source_metadata, 'rotation', None, context)
                if check_contains_value(source_metadata['rotation'], 'invert') and self.rotation:
                    self.rotation = -self.rotation
            if 'channels' in source_metadata and source_metadata['channels']:
                self.channels = source_metadata['channels']

        self.scale_factors = [{dim: value0 / value for dim, value, value0
                               in zip(self.dimension_order, shape, self.shape) if dim in 'xyz'}
                               for shape in self.shapes]
        # re-derive every level from the level 0 value so a 'scale' override reaches all levels
        self.pixel_sizes = [{dim: self.pixel_sizes[0][dim] * factor.get(dim, 1) for dim in self.pixel_sizes[0]}
                            for factor in self.scale_factors]
        self.pixel_size = self.pixel_sizes[0]

        if isinstance(source_metadata, dict):
            if 'is_center' in source_metadata:
                self.position = {dim: self.position[dim] - self.get_physical_size().get(dim, 0) / 2
                                 for dim in self.position}

        if 'SBEMimage' in self.creator:
            source_version = self.creator
            if '2025' in source_version:
                metapath = find_sbemimage_meta_dir(self.filename)
                if metapath:
                    sbemimage_config = load_sbemimage_best_config(metapath, self.filename)
                    if sbemimage_config:
                        size = self.get_size()
                        translation, scale0 = adjust_sbemimage_properties(
                            self.position, self.pixel_size, size, self.filename, sbemimage_config)
                        self.position = translation
                        if scale0:
                            self.pixel_sizes[0] = scale0
                            self.pixel_size = self.pixel_sizes[0]
                        elif self.pixel_size.get('x') != self.pixel_size.get('y'):
                            logging.warning('SBEMimage pixel size requires correction,'
                                            ' please provide in source metadata.')
                        logging.debug(f'Adjusted SBEMimage properties for {self.filename}')
                    else:
                        logging.warning(f'Could not find SBEMimage config for {self.filename}.')
                else:
                    logging.warning(f'Could not find SBEMimage config for {self.filename}.')

        if isinstance(source_metadata, dict):
            if 'invert' in source_metadata:
                if 'x' in self.position:
                    self.position['x'] = -self.position['x']
                if 'y' in self.position:
                    self.position['y'] = -self.position['y']

    def select_source_metadata(self, source_metadata):
        """This file's own entry of a run's source_metadata: its label's matrix, else the run-level values, a list
        value taking this file's index."""
        if not isinstance(source_metadata, dict):
            return source_metadata
        if self.file_label is not None and self.file_label in source_metadata:
            transform = param_utils.affine_to_xaffine(np.array(source_metadata[self.file_label]))
            position, rotation, scale = get_properties_from_transform(transform)
            selected = {'position': position, 'rotation': rotation, 'scale': xyz_to_dict([scale, scale])}
        else:
            selected = {key: value[self.index] if isinstance(value, list) and self.index is not None else value
                        for key, value in source_metadata.items()
                        if key in ('position', 'scale', 'rotation', 'channels')}
        # run-level flags apply to every file, whatever its entry
        for flag in ('invert', 'is_center'):
            if flag in source_metadata:
                selected[flag] = source_metadata[flag]
        return selected

    def _load_data(self):
        """Populate self._data with one array per level of the source's own pyramid, on first access to
        self.data. A subclass with no raw arrays to offer leaves it empty."""

    def _synthesized_level_factors(self):
        """Cumulative downsample factors of the coarse levels to synthesize; empty unless single-resolution.

        Without a coarse level, a zoomed-out view or a reduced-resolution fusion has to compute the whole
        finest level.
        """
        if len(self.shapes) != 1:
            return []
        sizes = {dim: size for dim, size in zip(self.dimension_order, self.shape) if dim in 'xyz'}
        return calc_pyramid_level_factors(sizes)

    def _add_missing_pyramid_level_metadata(self):
        # from the finest level's shape alone, so it needs no arrays and can run during __init__
        axes = {dim: axis for axis, dim in enumerate(self.dimension_order)}
        for level_factors in self._synthesized_level_factors():
            shape = tuple(ceil(size / level_factors.get(dim, 1))
                          for dim, size in zip(self.dimension_order, self.shape))
            self.shapes.append(shape)
            # from the realised extent ratio, not the factor: strided slicing rounds up, so odd
            # extents shrink by less than their factor
            self.pixel_sizes.append({dim: value * self.shape[axes[dim]] / shape[axes[dim]]
                                     for dim, value in self.pixel_sizes[0].items()
                                     if dim in axes})
        if len(self.shapes) > 1:
            self.scale_factors = [{dim: value0 / value for dim, value, value0
                                   in zip(self.dimension_order, shape, self.shape) if dim in 'xyz'}
                                   for shape in self.shapes]

    def _add_missing_pyramid_level_data(self):
        if len(self._data) != 1 or len(self.shapes) == 1:
            return
        datas, _ = build_missing_pyramid_levels(
            self._data[0], self.dimension_order, self.pixel_sizes[0])
        # metadata and arrays come from the same rule, so they must describe the same pyramid
        assert [tuple(data.shape) for data in datas] == [tuple(shape) for shape in self.shapes], \
            (f'{self.filename}: synthesized levels {[tuple(d.shape) for d in datas]} do not'
             f' match the shapes settled at init {self.shapes}')
        self._data = datas

    def _build_msim(self):
        # get_sim_from_array always adds a 'c' dim; label it so a channel can be selected by name
        c_coords = [channel.get('label', '') for channel in self.get_channels()]
        # get_sim_from_array needs a translation covering every spatial dim, or None
        translation_arg = complete_translation(self.position) or None
        # unplaced (init_data places it), z only with more than one plane
        identity = np.eye(4 if self.get_size().get('z', 0) > 1 else 3)
        sims = []
        for level, data in enumerate(self.data):
            scale = self.pixel_sizes[level]
            scale_arg = scale if scale else None
            sims.append(si_utils.get_sim_from_array(
                data, dims=list(self.dimension_order),
                scale=scale_arg, translation=translation_arg, affine=identity, transform_key=self.transform_key,
                c_coords=c_coords))
        self._msim = msi_utils.get_msim_from_sims(sims)

    def _restamp_msim(self):
        # a native reader's transform is always 4x4; ours has z only where there is more than one plane
        xaffine = param_utils.identity_transform(3 if self.get_size().get('z', 0) > 1 else 2)
        for scale_key in msi_utils.get_sorted_scale_keys(self.msim):
            ds = self.msim[scale_key].to_dataset()
            # drop x_in/x_out too, else a differently-shaped transform gets NaN-padded against them
            ds = ds.drop_vars([self.transform_key, 'x_in', 'x_out'], errors='ignore')
            self.msim[scale_key] = ds.assign({self.transform_key: xaffine})

    def get_level_data(self, level=0):
        return self.msim[f'scale{level}'].ds['image'].data

    def get_shape(self, level=0):
        return self.shapes[level]

    def get_size(self, level=0, asarray=False, axes='zyx'):
        size = {dim: size for dim, size in zip(self.dimension_order, self.get_shape(level))}
        if asarray:
            return np.array([size[dim] for dim in axes if dim in size])
        else:
            return size

    def get_pixel_size(self, level=0, asarray=False, axes='zyx'):
        # pixel size in micrometers
        pixel_size = self.pixel_sizes[level]
        if asarray:
            return np.array([pixel_size[dim] for dim in axes if dim in pixel_size])
        else:
            return pixel_size

    def get_physical_size(self, asarray=False, axes='zyx'):
        pixel_size = self.get_pixel_size()
        size = self.get_size()
        physical_size = {dim: size[dim] * pixel_size[dim] for dim in size if dim in pixel_size}
        if asarray:
            return np.array([physical_size[dim] for dim in axes if dim in physical_size])
        else:
            return physical_size

    def get_position(self, asarray=False, axes='zyx'):
        # position in micrometers
        if asarray:
            return np.array([self.position[dim] for dim in axes if dim in self.position])
        else:
            return self.position

    def get_rotation(self):
        # rotation in degrees
        return self.rotation

    def get_nchannels(self):
        return self.get_size().get('c', 1)

    def get_channels(self):
        if not self.channels:
            if self.is_rgb:
                return [{'label': ''}]
            else:
                return [{'label': f'channel {index}'} for index in range(self.get_nchannels())]
        return self.channels
