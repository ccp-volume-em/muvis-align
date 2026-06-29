# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

from contextlib import nullcontext
import dask
from dask.diagnostics import ProgressBar
from enum import Enum, auto
import logging
from multiview_stitcher import registration, vis_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
from multiview_stitcher.param_resolution import groupwise_resolution
from multiview_stitcher.registration import compute_pairwise_registrations, _plot_registration_summaries
import networkx as nx
import numpy as np
import os.path
import shutil
from skimage.transform import resize
import xarray as xr

from src.muvis_align.constants import *
from src.muvis_align.file.rocrate_utils import create_ro_crate, create_zarr_ro_crate
from src.muvis_align.image.Video import Video
from src.muvis_align.image.flatfield import flatfield_correction
from src.muvis_align.image.ome_helper import save_image
from src.muvis_align.image.ome_tiff_helper import save_tiff
from src.muvis_align.image.source_helper import create_dask_source
from src.muvis_align.image.util import *
from src.muvis_align.metrics import calc_pair_metrics, calc_global_metrics
from src.muvis_align.Timer import Timer
from src.muvis_align.util import *


class RegState(Enum):
    UNINIT = auto()
    INIT = auto()
    SIMS_INIT = auto()
    PAIRS_REG = auto()
    GLOBAL_REG = auto()
    FUSED = auto()


class MVSRegistration:
    def __init__(self, operation='register', label='', input_path=None, output_path=None,
                 source_metadata={}, extra_metadata={},
                 global_rotation=None, global_center=None,
                 overwrite=True, clear=False, ui='', verbose=False, debug=False):
        self.reset()

        if input_path is not None:
            self.init(operation=operation, label=label, input_path=input_path, output_path=output_path,
                      source_metadata=source_metadata, extra_metadata=extra_metadata,
                      global_rotation=global_rotation, global_center=global_center,
                      overwrite=overwrite, clear=clear, ui=ui, verbose=verbose, debug=debug)

    def reset(self):
        self.state = RegState.UNINIT
        self.source_metadata = {}
        self.extra_metadata = {}
        self.sims = []
        self.register_sims = []
        self.sources = []
        self.metrics = {}
        self.register_indices = None
        self.output_params = {}

    def is_initialised(self):
        return self.state.value >= RegState.INIT.value

    def is_pairs_registered(self):
        return self.state.value >= RegState.PAIRS_REG.value

    def is_global_registered(self):
        return self.state.value >= RegState.GLOBAL_REG.value

    def is_fused(self):
        return self.state.value >= RegState.FUSED.value

    def init_params(self, params_general, params, label='', input_path=None, global_rotation=None, global_center=None):
        self.params_general = params_general
        self.params = params
        self.input_params = params.get('input')
        if isinstance(self.input_params, (str, list)):
            self.input_params = {'path': self.input_params}
        if input_path is None:
            input_path = self.input_params.get('path')
        self.output_params = params.get('output')
        if isinstance(self.output_params, str):
            self.output_params = {'path': self.output_params}
        self.preprocess_params = params.get('preprocessing', {})
        self.register_params = params.get('registration', {})
        self.fusion_params = params.get('fusion', {})

        return self.init(operation=params.get('operation'), label=label, input_path=input_path,
                         input_labels=self.input_params.get('labels'),
                         output_path=self.output_params.get('path'),
                         source_metadata=self.input_params.get('source_metadata', {}),
                         extra_metadata=self.input_params.get('extra_metadata', {}),
                         global_rotation=global_rotation, global_center=global_center,
                         overwrite=params_general.get('overwrite', False), clear=params_general.get('clear', False),
                         ui=params_general.get('ui', ''),
                         verbose=params_general.get('verbose', False), debug=params_general.get('debug', False))

    def init(self, operation='', label='', input_path=None, input_labels=None, output_path=None,
             source_metadata={}, extra_metadata={}, global_rotation=None, global_center=None,
             overwrite=True, clear=False, ui='', verbose=False, debug=False):
        self.overwrite = overwrite
        self.clear = clear
        self.ui = ui
        self.verbose = verbose
        self.debug = debug
        self.logging_dask = self.verbose
        self.logging_time = self.verbose
        self.mpl_ui = ('mpl' in self.ui or 'plot' in self.ui)
        self.operation = operation
        self.fileset_label = label
        self.global_rotation = global_rotation
        self.global_center = global_center
        self.source_transform_key = 'source_metadata'
        self.reg_transform_key = 'registered'
        self.transition_transform_key = 'transition'
        self.sims = []
        self.sources = []
        self.state = RegState.INIT

        self.input_path = input_path
        if isinstance(input_path, list):
            self.filenames = input_path
            self.input_dir = os.path.dirname(input_path[0])
        else:
            self.filenames = dir_regex(input_path)
            self.input_dir = os.path.dirname(input_path)
        if not self.filenames:
            return False

        if input_labels:
            self.file_labels = input_labels
        else:
            self.file_labels = get_unique_file_labels(self.filenames)

        self.source_metadata = source_metadata
        self.extra_metadata = extra_metadata

        output_path = output_path.format_map(split_numeric_dict(self.filenames[0]))
        self.output = os.path.join(self.input_dir, output_path)    # preserve trailing slash: do not use os.path.normpath()
        output_dir = os.path.dirname(self.output)
        if self.clear:
            shutil.rmtree(output_dir, ignore_errors=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return True

    def run(self):
        with ProgressBar(minimum=60, dt=1) if self.logging_dask else nullcontext():
            return self._run()

    def _run(self):
        filenames = self.filenames
        file_labels = self.file_labels

        output = self.output
        operation = self.operation
        source_metadata = self.source_metadata
        extra_metadata = self.extra_metadata
        if isinstance(extra_metadata, dict):
            z_scale = extra_metadata.get('scale', {}).get('z')
            channels = extra_metadata.get('channels', [])
        else:
            z_scale = None
            channels = []
        normalise_orientation = 'norm' in source_metadata
        output_params = self.output_params
        general_output_params = self.params_general.get('output', {})
        overlap_threshold = self.register_params.get('overlap_threshold', self.params.get('overlap_threshold', 0.5))
        save_images = self.output_params.get('save_images', self.params.get('save_images', True))

        output_format = output_params.get('format', general_output_params.get('format', zarr_extension))
        output_tile_size = output_params.get('tile_size', general_output_params.get('tile_size'))
        output_compression = output_params.get('compression', general_output_params.get('compression'))
        output_pyramid_downsample = output_params.get('pyramid_downsample', general_output_params.get('pyramid_downsample', 2))
        output_npyramid_add = output_params.get('npyramid_add', general_output_params.get('npyramid_add', 0))
        output_ome_version = output_params.get('ome_version', general_output_params.get('ome_version', default_ome_zarr_version))

        mappings_header = ['id','x_pixels', 'y_pixels', 'z_pixels', 'x', 'y', 'z', 'rotation']

        if len(filenames) == 0:
            logging.warning('Skipping (no images)')
            return False

        output_filename = operation.split()[0] + 'ed'

        self.check_progress(output_filename, output_format)

        if self.is_fused() and not self.overwrite:
            logging.warning(f'Skipping existing output {output_filename}')
            return False

        with Timer('init sims', self.logging_time):
            sims = self.init_sims(target_scale=self.preprocess_params.get('scale'))
        self.sims = sims

        is_3d = (self.sources[0].get_size().get('z', 0) > 1)
        is_stack = ('stack' in operation)
        is_simple_stack = is_stack and not is_3d
        is_transition = ('transition' in operation)
        is_channel_overlay = (len(channels) > 1)
        if not z_scale:
            z_scale = self.scales[0].get('z', 1)

        with Timer('pre-process', self.logging_time):
            register_sims, register_indices, _ = self.preprocess(sims, **self.preprocess_params)

        self.init_progress(output_filename, output_format)

        data = []
        for label, sim, scale in zip(file_labels, sims, self.scales):
            position, rotation = get_data_mapping(sim, transform_key=self.source_transform_key)
            position_pixels = {dim: position[dim] / float(scale.get(dim, 1)) for dim in position.keys()}
            row = [label] + dict_to_xyz(position_pixels, add_zeros=True) + dict_to_xyz(position, add_zeros=True) + [rotation]
            data.append(row)
        export_csv(output + prereg_mappings_name, data, header=mappings_header)

        if len(filenames) == 1 and save_images and not 'register' in operation and not 'stack' in operation:
            logging.warning('Skipping operation (single image)')
            self.save(output_filename, sims[0], translations0=self.positions,
                      format=output_format,
                      tile_size=output_tile_size,
                      pyramid_downsample=output_pyramid_downsample,
                      npyramid_add=output_npyramid_add,
                      ome_version=output_ome_version)
            return False

        _, has_overlaps = self.validate_overlap(sims, file_labels, is_stack=is_simple_stack,
                                                expect_large_overlap=is_simple_stack or is_channel_overlay)
        overall_overlap = np.mean(has_overlaps)
        if overall_overlap < overlap_threshold:
            raise ValueError(f'Not enough overlap: {overall_overlap * 100:.1f}%')

        if not self.is_global_registered() or self.overwrite:
            if 'register' in operation:
                with Timer('register', self.logging_time):
                    results = self.register(sims, register_sims, register_indices, self.register_params)
                reg_result = results['reg_result']
                mappings = results['mappings']
                metrics = results['metrics']

            if is_stack:
                sims = make_sims_3d(sims, z_scale, self.positions)

            if 'register' in operation:
                logging.info(metrics['summary'])
                data = []
                for label, sim, mapping, scale, position, rotation\
                        in zip(file_labels, sims, mappings.values(), self.scales, self.positions, self.rotations):
                    if not normalise_orientation:
                        # rotation already in msim affine transform
                        rotation = None
                    position, rotation = get_data_mapping(sim, transform_key=self.reg_transform_key,
                                                          transform=mapping,
                                                          translation0=position,
                                                          rotation=rotation)
                    position_pixels = {dim: position[dim] / float(scale.get(dim, 1)) for dim in position.keys()}
                    row = [label] + dict_to_xyz(position_pixels, add_zeros=True) + dict_to_xyz(position, add_zeros=True) + [rotation]
                    data.append(row)
                export_csv(output + metrics_tabular_name, data, header=mappings_header)

                for reg_label, reg_item in reg_result.items():
                    if isinstance(reg_item, dict):
                        summary_plot = reg_item.get('summary_plot')
                        if summary_plot is not None:
                            figure, axes = summary_plot
                            summary_plot_filename = output + f'{reg_label}.pdf'
                            figure.savefig(summary_plot_filename)

        self.sims = sims
        registered_positions_filename = output + registered_positions_name
        if self.reg_transform_key in sims[0].transforms:
            transform_key = self.reg_transform_key
            with Timer('plot positions', self.logging_time):
                vis_utils.plot_positions(sims, transform_key=transform_key,
                                         use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                         show_plot=self.mpl_ui, output_filename=registered_positions_filename)
                plt_close()
        else:
            transform_key = self.source_transform_key

        logging.info('Exporting...')

        image_paths = []
        if save_images:
            if self.output_params.get('preview'):
                with Timer('create preview', self.logging_time):
                    self.create_preview('preview_' + output_filename,
                                        nom_sims=sims,
                                        transform_key=transform_key)

            if 'register' in operation or 'stack' in operation:
                with Timer('fuse image', self.logging_time):
                    if isinstance(self.fusion_params, dict):
                        fusion_method = self.fusion_params.get('method', '')
                        output_spacing = self.fusion_params.get('output_spacing', 'mean')
                    else:
                        fusion_method = self.fusion_params
                        output_spacing = self.params.get('output_spacing', 'mean')
                    fused_image, is_saved = self.fuse(sims, fusion_method=fusion_method, output_spacing=output_spacing,
                                                      transform_key=transform_key, output_filename=output_filename,
                                                      tile_size=output_tile_size, ome_version=output_ome_version)
                    self.state = RegState.FUSED
            else:
                fused_image = sims
                is_saved = False

            if not is_saved or 'tif' in output_format:
                logging.info('Saving fused image...')
                with Timer('save fused image', self.logging_time):
                    self.save(output_filename, fused_image,
                              transform_key=transform_key, translations0=self.positions,
                              format = output_format,
                              tile_size = output_tile_size,
                              compression = output_compression,
                              pyramid_downsample = output_pyramid_downsample,
                              npyramid_add = output_npyramid_add,
                              ome_version = output_ome_version)

            if 'tif' in output_format:
                filename = output_filename + tiff_extension
                image_paths.append(filename)
            if 'zar' in output_format:
                filename = output_filename + zarr_extension
                image_paths.append(filename)
                create_zarr_ro_crate(self.output + filename)

        create_ro_crate(fused_image, self.output, image_paths)

        if is_transition:
            self.save_video(output, sims, fused_image)

        return True

    def init_sources(self):
        source_metadata0 = self.source_metadata
        source_metadata = {}
        self.sources = []
        for index, (filename, label) in enumerate(zip(self.filenames, self.file_labels)):
            if isinstance(source_metadata0, dict) and label in source_metadata0:
                source_metadata = source_metadata0[label]
                position, rotation, scale = get_properties_from_transform(param_utils.affine_to_xaffine(np.array(source_metadata)))
                source_metadata = {'position': position, 'rotation': rotation, 'scale': xyz_to_dict([scale, scale])}
            else:
                if 'position' in source_metadata0:
                    translation = source_metadata0['position']
                    if isinstance(translation, list):
                        translation = translation[index]
                    source_metadata['position'] = translation
                if 'scale' in source_metadata0:
                    scale = source_metadata0['scale']
                    if isinstance(scale, list):
                        scale = scale[index]
                    source_metadata['scale'] = scale
                if 'rotation' in source_metadata0:
                    source_metadata['rotation'] = source_metadata0['rotation']
            self.sources.append(create_dask_source(filename, source_metadata))

    def init_sims(self, source_metadata={}, extra_metadata={}, z_scale=None, chunk_size=default_chunk_size,
                  target_scale=None, store=True):
        if not source_metadata:
            source_metadata = self.source_metadata
        if not extra_metadata:
            extra_metadata = self.extra_metadata
        source_metadata = import_metadata(source_metadata, input_path=self.input_path)
        extra_metadata = import_metadata(extra_metadata, input_path=self.input_path)
        source_metadata_changed = (source_metadata != self.source_metadata)
        self.source_metadata = source_metadata
        self.extra_metadata = extra_metadata
        if isinstance(source_metadata, dict):
            z_scale = source_metadata.get('scale', {}).get('z')
        if not z_scale and isinstance(extra_metadata, dict):
            z_scale = extra_metadata.get('scale', {}).get('z')

        if len(self.filenames) == 0:
            raise ValueError('No input files')

        logging.info('Initialising sims...')
        if not self.sources or source_metadata_changed:
            self.init_sources()
        sources = self.sources
        source0 = sources[0]
        images = []
        sims = []
        scales = []
        translations = []
        rotations = []

        is_3d = (source0.get_size().get('z', 0) > 1)
        is_stack = ('stack' in self.operation)
        output_order = 'zyx' if is_3d else 'yx'

        ndims = len(output_order)
        if source0.get_nchannels() > 1:
            output_order += 'c'

        last_z_position = None
        different_z_positions = False
        delta_zs = []
        for filename, source in zip(self.filenames, sources):
            scale = source.get_pixel_size()
            translation = source.get_position()
            rotation = source.get_rotation()

            if 'is_center' in source_metadata:
                translation = {dim: translation[dim] - source.get_physical_size().get(dim, 0) / 2 for dim in translation}

            if 'sbem' in source_metadata:
                source_version = source.metadata.get('Creator', source.metadata.get('creator', ''))
                if '2025' in source_version:
                    path = os.path.dirname(self.filenames[0])
                    metapath = None
                    attempts = 0
                    while attempts < 3:
                        metapath = os.path.join(path, 'meta')
                        if os.path.exists(metapath):
                            break
                        path = os.path.join(path, '..')
                        attempts += 1
                    if metapath:
                        sbemimage_config = load_sbemimage_best_config(metapath, filename)
                        if sbemimage_config:
                            size = source.get_size()
                            translation, scale0 = adjust_sbemimage_properties(translation, scale, size,
                                                                              filename, sbemimage_config)
                            if scale0:
                                scale = scale0
                            elif scale.get('x') != scale.get('y'):
                                logging.warning('SBEMimage pixel size requires correction, please provide in source metadata.')
                            logging.debug(f'Adjusted SBEMimage properties for {filename}')
                        else:
                            logging.warning(f'Could not find SBEMimage config for {filename}.')

            level = 0
            rescale = 1
            if target_scale:
                # Only downscaling
                level, rescale, scale = get_level_from_scale(source, target_scale)
            if 'invert' in source_metadata:
                translation['x'] = -translation['x']
                translation['y'] = -translation['y']
            if 'z' in translation:
                z_position = translation['z']
            else:
                z_position = 0
            if last_z_position is not None and z_position != last_z_position:
                delta_zs.append(z_position - last_z_position)
            if 'rotation' in source_metadata:
                rotation = source_metadata['rotation']
            if self.global_rotation is not None:
                rotation = self.global_rotation

            dask_data = source.get_data(level=level)
            if any(value != 1 for value in rescale.values()):
                new_shape = [int(size / rescale[dim]) if dim in 'xyz' else 1
                             for dim, size in zip(source.dimension_order, dask_data.shape)]
                dask_data = resize(dask_data, new_shape, preserve_range=True).astype(dask_data.dtype)
            image = redimension_data(dask_data, source.dimension_order, output_order)

            scales.append(scale)
            translations.append(translation)
            rotations.append(rotation)
            images.append(image)
            last_z_position = z_position

        if 'z' in output_order and z_scale is None:
            if len(delta_zs) > 0:
                z_scale = np.min(delta_zs)
            else:
                z_scale = 1

        if 'norm' in source_metadata:
            sizes = [source.get_physical_size() for source in sources]
            center = {dim: 0 for dim in output_order}
            if 'center' in source_metadata:
                if 'global' in source_metadata:
                    center = self.global_center
                else:
                    center = {dim: float(np.mean([translation[dim] for translation in translations])) for dim in translations[0]}
            translations, rotations = normalise_rotated_positions(translations, rotations, sizes, center, len(output_order))

        #translations = [np.array(translation) * 1.25 for translation in translations]

        increase_z_positions = is_stack and not different_z_positions

        z_position = 0
        final_scales = []
        final_translations = []
        for source, image, scale, translation, rotation, file_label in zip(sources, images, scales, translations, rotations, self.file_labels):
            # transform #dimensions need to match
            if 'z' in output_order:
                if len(scale) > 0 and 'z' not in scale:
                    scale['z'] = abs(z_scale)
                if (len(translation) > 0 and 'z' not in translation) or increase_z_positions:
                    translation['z'] = z_position
                if increase_z_positions:
                    z_position += z_scale
            channel_labels = [channel.get('label', '') for channel in source.get_channels()]
            if rotation is None or 'norm' in source_metadata:
                # if positions are normalised, don't use rotation
                transform = None
            else:
                transform = param_utils.invert_coordinate_order(
                    create_transform(translation, rotation, matrix_size=ndims + 1)
                )
            if file_label in extra_metadata:
                transform2 = extra_metadata[file_label]
                if transform is None:
                    transform = np.array(transform2)
                else:
                    transform = np.array(combine_transforms([transform, transform2]))

            # fix empty dictionaries args
            scale_arg = scale if scale else None
            translation_arg = translation if translation else None

            sim = si_utils.get_sim_from_array(
                image,
                dims=list(output_order),
                scale=scale_arg,
                translation=translation_arg,
                affine=transform,
                transform_key=self.source_transform_key,
                c_coords=channel_labels
            )
            if len(sim.chunksizes.get('x')) == 1 and len(sim.chunksizes.get('y')) == 1:
                if isinstance(chunk_size, int):
                    chunk_size = [chunk_size] * 2
                sim = sim.chunk(xyz_to_dict(chunk_size))
            sims.append(sim)
            final_scales.append(scale)
            final_translations.append(translation)

        if store:
            self.sims = sims
            self.scales = final_scales
            self.positions = final_translations
            self.rotations = rotations
            self.state = RegState.SIMS_INIT

        #print_sim_info(sims[0])
        return sims

    def check_progress(self, output_filename, output_format):
        pair_mappings_filename = self.output + self.output_params.get('pair_mappings', default_pair_mappings_name)
        mappings_filename = self.output + self.output_params.get('mappings', default_mappings_name)
        if self.output_exists(output_filename, output_format):
            self.state = RegState.FUSED
        elif os.path.exists(mappings_filename):
            self.state = RegState.GLOBAL_REG
        elif os.path.exists(pair_mappings_filename):
            self.state = RegState.PAIRS_REG

    def init_progress(self, output_filename, output_format):
        pair_mappings_filename = self.output + self.output_params.get('pair_mappings', default_pair_mappings_name)
        mappings_filename = self.output + self.output_params.get('mappings', default_mappings_name)
        metrics_filename = self.output + metrics_name
        is_3d = (self.sources[0].get_size().get('z', 0) > 1)
        self.check_progress(output_filename, output_format)

        if self.is_pairs_registered():
            # load pair mapping and initialise pair_graph
            pairs = import_json(pair_mappings_filename)
            indexed_pair_transforms = {}
            indexed_qualities = {}
            indexed_bboxes = {}
            for key, value in pairs.items():
                key1, key2 = key.split('-')
                indexed_key = self.file_labels.index(key1), self.file_labels.index(key2)
                indexed_pair_transforms[indexed_key] = (
                    param_utils.affine_to_xaffine(np.array(value['mapping'])).expand_dims({'t': [0]}))
                indexed_qualities[indexed_key] = np.array(value[default_quality_key])
                indexed_bboxes[indexed_key] = xr.DataArray(value['bbox'])
            if not is_3d:
                self.sims = make_sims_2d(self.sims)
            self.msims = [msi_utils.get_msim_from_sim(sim) for sim in self.sims]
            self.pairs = list(indexed_pair_transforms.keys())
            self.metrics = {
                'summary': {default_transform_key: {default_quality_key: np.mean(list(indexed_qualities.values()))}},
                'pairs': {key: {default_transform_key: {default_quality_key: value.item()}}
                          for key, value in indexed_qualities.items()}
            }
            with dask.config.set(scheduler='single-threaded'):
                self.pairs_graph = mv_graph.build_view_adjacency_graph_from_msims(
                    self.msims,
                    transform_key=self.source_transform_key,
                    pairs=self.pairs
                )
            nx.set_edge_attributes(self.pairs_graph, indexed_pair_transforms, default_transform_key)
            nx.set_edge_attributes(self.pairs_graph, indexed_qualities, default_quality_key)
            nx.set_edge_attributes(self.pairs_graph, indexed_bboxes, 'bbox')

        if self.is_global_registered():
            # load mapping
            sims = self.sims

            is_stack = ('stack' in self.operation)
            z_positions = set([source.get_position().get('z', 0) for source in self.sources])
            make_3d = len(z_positions) > 1 or is_stack
            if isinstance(self.extra_metadata, dict):
                z_scale = self.extra_metadata.get('scale', {}).get('z')
            else:
                z_scale = None

            mappings = import_json(mappings_filename)
            # copy transforms to sims
            for sim, label in zip(sims, self.file_labels):
                mapping = param_utils.affine_to_xaffine(np.array(mappings[label]))
                if make_3d:
                    transform = param_utils.identity_transform(ndim=3)
                    transform.loc[{dim: mapping.coords[dim] for dim in mapping.dims}] = mapping
                else:
                    transform = mapping
                si_utils.set_sim_affine(sim, transform, transform_key=self.reg_transform_key)
            if make_3d:
                sims = make_sims_3d(sims, z_scale, self.positions)
            if not is_3d:
                self.sims = make_sims_2d(self.sims)
            self.sims = sims
            self.msims = [msi_utils.get_msim_from_sim(sim) for sim in sims]
            metrics = import_json(metrics_filename)
            indexed_metrics = {}
            for key, value in metrics.items():
                key1, key2 = key.split('-')
                indexed_key = self.file_labels.index(key1), self.file_labels.index(key2)
                indexed_metrics[indexed_key] = value
            self.metrics = {
                'summary': {default_transform_key:
                                {self.reg_transform_key: np.mean([value[default_quality_key]
                                                                  for value in indexed_metrics.values()
                                                                  if default_quality_key in value])}},
                'pairs': {key: {self.reg_transform_key: value} for key, value in indexed_metrics.items()}
            }

    def validate_overlap(self, sims, labels, is_stack=False, expect_large_overlap=False):
        min_dists = []
        has_overlaps = []
        n = len(sims)
        positions = [get_sim_position_final(sim, get_center=True) for sim in sims]
        sizes = [float(np.linalg.norm(list(get_sim_physical_size(sim).values()))) for sim in sims]
        for i in range(n):
            norm_dists = []
            # check if only single z slices
            if is_stack:
                if i + 1 < n:
                    compare_indices = [i + 1]
                else:
                    compare_indices = []
            else:
                compare_indices = range(n)
            for j in compare_indices:
                if not j == i:
                    distance = math.dist(positions[i].values(), positions[j].values())
                    norm_dist = distance / np.mean([sizes[i], sizes[j]])
                    norm_dists.append(norm_dist)
            if len(norm_dists) > 0:
                norm_dist = min(norm_dists)
                min_dists.append(float(norm_dist))
                if norm_dist >= 1:
                    logging.warning(f'{labels[i]} has no overlap')
                    has_overlaps.append(False)
                elif expect_large_overlap and norm_dist > 0.5:
                    logging.warning(f'{labels[i]} has small overlap')
                    has_overlaps.append(False)
                else:
                    has_overlaps.append(True)
        return min_dists, has_overlaps

    def check_preprocess(self,
                         flatfield_quantiles=None, normalisation=None, gaussian_sigma=None, filter_foreground=False):
        if normalisation:
            if isinstance(normalisation, str) and normalisation.lower() in ['false', 'no', 'none', '']:
                normalisation = None
            elif isinstance(normalisation, bool) and normalisation == False:
                normalisation = None
        if flatfield_quantiles or normalisation or gaussian_sigma or filter_foreground:
            return True
        else:
            return False

    def preprocess(self, sims,
                   flatfield_quantiles=None, normalisation=None, gaussian_sigma=None, filter_foreground=False,
                   **kwargs):
        modified = False
        # normalise pixel size: take max pixel size
        max_scale = {dim: max(scale.get(dim, 1) for scale in self.scales) for dim in 'xy'}
        scales0 = self.scales

        if filter_foreground:
            foreground_map = calc_foreground_map(sims)
            modified = True
        else:
            foreground_map = None
        if flatfield_quantiles:
            logging.info('Flat-field correction...')
            if isinstance(flatfield_quantiles, str):
                flatfield_quantiles = [float(quantile.strip()) for quantile in flatfield_quantiles.split(',')]
            new_sims = [None] * len(sims)
            for sim_indices in group_sims_by_z(sims, self.positions):
                sims_z_set = [sims[i] for i in sim_indices]
                foreground_map_z_set = [foreground_map[i] for i in sim_indices] if foreground_map is not None else None
                new_sims_z_set = flatfield_correction(sims_z_set, self.source_transform_key, flatfield_quantiles,
                                                      foreground_map=foreground_map_z_set)
                for sim_index, sim in zip(sim_indices, new_sims_z_set):
                    new_sims[sim_index] = sim
            sims = new_sims
            modified = True

        if gaussian_sigma:
            logging.info('Applying Gaussian filtering...')
            new_sims = []
            for sim, scale0 in zip(sims, scales0):
                # factor in original pixel size for gaussian sigma value
                scale = np.mean(list(scale0.values())) / np.mean(list(max_scale.values()))
                sigma = gaussian_sigma * (scale ** (1 / 3))
                new_sims.append(gaussian_filter_sim(sim, self.source_transform_key, sigma))
            sims = new_sims
            modified = True

        if normalisation:
            if isinstance(normalisation, str) and normalisation.lower() in ['false', 'no', 'none', '']:
                normalisation = None
            elif isinstance(normalisation, bool) and normalisation == False:
                normalisation = None
        if normalisation:
            use_global = ('global' in str(normalisation).lower())
            if use_global:
                logging.info('Normalising (global)...')
            else:
                logging.info('Normalising (individual)...')
            sims = normalise_sims(sims, self.source_transform_key, use_global=use_global)
            modified = True

        if filter_foreground:
            logging.info('Filtering foreground images...')
            #tile_vars = np.array([np.asarray(np.std(sim)).item() for sim in sims])
            #threshold1 = np.mean(tile_vars)
            #threshold2 = np.median(tile_vars)
            #threshold3, _ = cv.threshold(np.array(tile_vars).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
            #threshold = min(threshold1, threshold2, threshold3)
            #foregrounds = (tile_vars >= threshold)
            new_sims = [sim for sim, is_foreground in zip(sims, foreground_map) if is_foreground]
            logging.info(f'Foreground images: {len(new_sims)} / {len(sims)}')
            indices = np.where(foreground_map)[0]
            sims = new_sims
            modified = True
        else:
            indices = range(len(sims))
        self.register_sims = sims
        self.register_indices = indices
        return sims, indices, modified

    def create_registration_method(self, sim0, params={}, method=''):
        registration_method = None
        pairwise_reg_func_kwargs = None

        if 'registration' in params:
            params = params['registration']
        if not method:
            method = params.get('method',
                                params.get('name', ''))
        method = method.lower()

        if 'cpd' in method:
            from src.muvis_align.registration_methods.RegistrationMethodCPD import RegistrationMethodCPD
            registration_method = RegistrationMethodCPD(sim0, params, self.debug)
            pairwise_reg_func = registration_method.registration
        elif 'feature' in method or 'orb' in method or 'sift' in method:
            if 'cv' in method:
                from src.muvis_align.registration_methods.RegistrationMethodCvFeatures import RegistrationMethodCvFeatures
                registration_method = RegistrationMethodCvFeatures(sim0, params, self.debug)
            else:
                from src.muvis_align.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures
                registration_method = RegistrationMethodSkFeatures(sim0, params, self.debug)
            pairwise_reg_func = registration_method.registration
        elif 'ant' in method:
            pairwise_reg_func = registration.registration_ANTsPy
            # args for ANTsPy registration: used internally by ANYsPy algorithm
            pairwise_reg_func_kwargs = {
                'transform_types': ['Rigid'],
                "aff_random_sampling_rate": 0.5,
                "aff_iterations": (2000, 2000, 1000, 1000),
                "aff_smoothing_sigmas": (4, 2, 1, 0),
                "aff_shrink_factors": (16, 8, 2, 1),
            }
        else:
            pairwise_reg_func = registration.phase_correlation_registration

        self.registration_method = registration_method

        return method, pairwise_reg_func, pairwise_reg_func_kwargs

    def create_fusion_method(self, fusion_method, sim0):
        if fusion_method is None:
            fusion_method = ''
        if 'compos' in fusion_method:
            fusion_method = None
            fuse_func = None
        elif 'exclus' in fusion_method:
            from src.muvis_align.fusion_methods.FusionMethodExclusive import FusionMethodExclusive
            fusion_method = FusionMethodExclusive(sim0, self.debug)
            fuse_func = fusion_method.fusion
        elif 'add' in fusion_method:
            from src.muvis_align.fusion_methods.FusionMethodAdditive import FusionMethodAdditive
            fusion_method = FusionMethodAdditive(sim0, self.debug)
            fuse_func = fusion_method.fusion
        else:
            fuse_func = fusion.simple_average_fusion

        self.fusion_method = fusion_method

        return fuse_func

    def register(self, sims, register_sims=None, register_indices=None, params=None):
        pair_results = self.register_pairs(sims, register_sims=register_sims, register_indices=register_indices, params=params)
        qualities = {key: metric[default_transform_key][default_quality_key]
                     for key, metric in pair_results['metrics']['pairs'].items()
                     if default_quality_key in metric[default_transform_key]}
        bboxes = {key: np.array(value.sel(t=0)).tolist() for key, value in nx.get_edge_attributes(self.pairs_graph, 'bbox').items()}
        self.save_pair_mappings(pair_results['pair_mappings'], qualities, bboxes)
        results = self.register_global(sims, self.msims, register_indices=register_indices, params=params)
        self.save_mappings(results['mappings'])
        self.save_metrics(results['metrics'])
        return results

    def register_full(self, sims, register_sims=None, register_indices=None, register_params=None):
        metrics = {}
        if register_params:
            params = register_params
        else:
            params = self.params
        sim0 = sims[0]
        ndims = si_utils.get_ndim_from_sim(sim0)

        operation = self.operation
        pairing = params.get('pairing', '')
        post_registration_quality_threshold = params.get('post_registration_quality_threshold')
        n_parallel_pairwise_regs = params.get('n_parallel_pairwise_regs')

        is_3d = (self.sources[0].get_size().get('z', 0) > 1)
        is_stack = ('stack' in operation)

        return_dict = True

        reg_channel = params.get('channel', 0)
        if isinstance(reg_channel, int):
            reg_channel_index = reg_channel
            reg_channel = None
        else:
            reg_channel_index = None

        groupwise_resolution_method = params.get('groupwise_resolution_method', 'global_optimization')
        groupwise_resolution_kwargs = None
        if groupwise_resolution_method == 'global_optimization' and 'transform_type' in params:
           groupwise_resolution_kwargs = {
                'transform': params['transform_type']  # options include 'translation', 'rigid', 'affine', 'similarity'
            }
        if groupwise_resolution_method == 'shortest_paths':
            return_dict = False

        if register_sims is None:
            register_sims = sims
        if is_stack and not is_3d:
            # register in 2d; pairwise consecutive views
            register_sims = [si_utils.max_project_sim(sim, dim='z') if 'z' in sim.dims else sim
                             for sim in register_sims]
            pairs = [(index, index + 1) for index in range(len(register_sims) - 1)]
        elif 'ortho' in pairing or 'overla' in pairing:
            origins = np.array([get_sim_position_final(sim, position, get_center=True)
                                for sim, position in zip(sims, self.positions)])
            sizes = [get_sim_physical_size(sim) for sim in sims]
            pairs, _ = get_pairs(origins, sizes, pairing)
            logging.info(f'#pairs: {len(pairs)}')
            #for pair in pairs:
            #    print(f'{self.file_labels[pair[0]]} - {self.file_labels[pair[1]]}')
        else:
            pairs = None

        reg_method, pairwise_reg_func, pairwise_reg_func_kwargs = self.create_registration_method(register_sims[0])

        logging.info(f'Registration method: {reg_method}')

        try:
            logging.info('Registering...')
            register_msims = [msi_utils.get_msim_from_sim(sim) for sim in register_sims]
            reg_result = registration.register(
                register_msims,
                reg_channel=reg_channel,
                reg_channel_index=reg_channel_index,
                transform_key=self.source_transform_key,
                new_transform_key=self.reg_transform_key,

                pairs=pairs,
                pre_registration_pruning_method=None,

                pairwise_reg_func=pairwise_reg_func,
                pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,

                groupwise_resolution_method=groupwise_resolution_method,
                groupwise_resolution_kwargs=groupwise_resolution_kwargs,

                post_registration_do_quality_filter=(post_registration_quality_threshold is not None),
                post_registration_quality_threshold=post_registration_quality_threshold,

                n_parallel_pairwise_regs=n_parallel_pairwise_regs,

                plot_summary=self.mpl_ui,
                return_dict=return_dict,
            )

            if register_indices is None:
                register_indices = range(len(register_msims))

            # copy transforms from register sims to unmodified sims
            for reg_msim, index in zip(register_msims, register_indices):
                si_utils.set_sim_affine(
                    sims[index],
                    msi_utils.get_transform_from_msim(reg_msim, transform_key=self.reg_transform_key),
                    transform_key=self.reg_transform_key)

            # set missing transforms
            for sim in sims:
                if self.reg_transform_key not in si_utils.get_tranform_keys_from_sim(sim):
                    si_utils.set_sim_affine(
                        sim,
                        param_utils.identity_transform(ndim=ndims, t_coords=[0]),
                        transform_key=self.reg_transform_key)

            if return_dict:
                mappings = reg_result['params']
                # re-index from subset of sims
                residual_error_dict = reg_result.get('groupwise_resolution', {}).get('metrics', {}).get('residuals', {})
                residual_error_dict = {(register_indices[key[0]], register_indices[key[1]]): value.item()
                                       for key, value in residual_error_dict.items()}
                registration_qualities_dict = reg_result.get('pairwise_registration', {}).get('metrics', {}).get('qualities', {})
                registration_qualities_dict = {(register_indices[key[0]], register_indices[key[1]]): value
                                               for key, value in registration_qualities_dict.items()}

                reg_channel = params.get('channel', 0)
                metrics = calc_global_metrics(register_msims, self.source_transform_key, self.reg_transform_key,
                                              params.get('metrics', []), reg_channel=reg_channel,
                                              reg_results=reg_result,
                                              n_parallel_pairs=n_parallel_pairwise_regs)
            else:
                mappings = reg_result
                reg_result = {}
                residual_error_dict = {}
                registration_qualities_dict = {}

        except NotEnoughOverlapError:
            logging.warning('Not enough overlap')
            reg_result = {}
            mappings = [param_utils.identity_transform(ndim=ndims, t_coords=[0])] * len(sims)
            residual_error_dict = {}
            registration_qualities_dict = {}

        # re-index from subset of sims
        mappings_dict = {index: mapping for index, mapping in zip(register_indices, mappings)}

        self.metrics = metrics
        self.state = RegState.GLOBAL_REG
        return {'reg_result': reg_result,
                'mappings': mappings_dict,
                'residual_errors': residual_error_dict,
                'registration_qualities': registration_qualities_dict,
                'metrics': metrics}

    def register_pairs(self, sims, register_sims=None, register_indices=None, params=None):
        if register_indices is None:
            if self.register_indices is not None:
                register_indices = self.register_indices
            else:
                register_indices = range(len(sims))

        operation = self.operation
        pairing = params.get('pairing',
                             params.get('registration', {}).get('pairing', '')).lower()
        n_parallel_pairwise_regs = params.get('n_parallel_pairwise_regs',
                                              params.get('registration', {}).get('n_parallel_pairwise_regs'))
        if n_parallel_pairwise_regs is not None and n_parallel_pairwise_regs == '0':
            n_parallel_pairwise_regs = None

        is_3d = (self.sources[0].get_size().get('z', 0) > 1)
        is_stack = ('stack' in operation)

        reg_channel = params.get('channel', 0)
        if isinstance(reg_channel, int):
            reg_channel_index = reg_channel
            reg_channel = None
        else:
            reg_channel_index = None

        if register_sims is None:
            register_sims = sims
        if is_stack and not is_3d:
            # register in 2d; pairwise consecutive views
            register_sims = [si_utils.max_project_sim(sim, dim='z') if 'z' in sim.dims else sim
                             for sim in register_sims]
            pairs = [(index, index + 1) for index in range(len(register_sims) - 1)]
        elif 'ortho' in pairing or 'overla' in pairing:
            origins = np.array([get_sim_position_final(sim, position, get_center=True)
                                for sim, position in zip(sims, self.positions)])
            sizes = [get_sim_physical_size(sim) for sim in sims]
            pairs, _ = get_pairs(origins, sizes, pairing)
            logging.info(f'#pairs: {len(pairs)}')
            #for pair in pairs:
            #    print(f'{self.file_labels[pair[0]]} - {self.file_labels[pair[1]]}')
        else:
            pairs = None

        reg_method, pairwise_reg_func, pairwise_reg_func_kwargs = self.create_registration_method(register_sims[0],
                                                                                                  params=params)
        logging.info(f'Registration method: {reg_method}')
        logging.info('Registering...')
        register_msims = [msi_utils.get_msim_from_sim(sim) for sim in register_sims]

        overlap_tolerance = 0

        # ******* start MVS registration functions

        if "c" in msi_utils.get_dims(register_msims[0]):
            if reg_channel is None:
                if reg_channel_index is None:
                    for msim in register_msims:
                        if "c" in msi_utils.get_dims(msim):
                            raise (
                                Exception("Please choose a registration channel.")
                            )
                else:
                    reg_channel = sims[0].coords["c"][reg_channel_index]

            msims_reg = [
                msi_utils.multiscale_sel_coords(msim, {"c": reg_channel})
                if "c" in msi_utils.get_dims(msim)
                else msim
                for imsim, msim in enumerate(register_msims)
            ]
        else:
            msims_reg = register_msims

        #print_sim_info(msims_reg[0])

        try:
            with dask.config.set(scheduler='threads'):
                g_reg = mv_graph.build_view_adjacency_graph_from_msims(
                    msims_reg,
                    transform_key=self.source_transform_key,
                    pairs=pairs,
                    overlap_tolerance=overlap_tolerance,
                )

                g_reg_computed = compute_pairwise_registrations(
                    msims_reg,
                    g_reg,
                    transform_key=self.source_transform_key,
                    overlap_tolerance=overlap_tolerance,
                    pairwise_reg_func=pairwise_reg_func,
                    pairwise_reg_func_kwargs=pairwise_reg_func_kwargs,
                    n_parallel_pairwise_regs=n_parallel_pairwise_regs,
                )

                # ******* end MVS registration functions

        except NotEnoughOverlapError:
            g_reg_computed = g_reg

        mappings = nx.get_edge_attributes(g_reg_computed, default_transform_key)
        mappings_dict = {(register_indices[indices[0]], register_indices[indices[1]]): mapping
                         for indices, mapping in mappings.items()}

        metrics = calc_pair_metrics(msims_reg, g_reg_computed, params.get('metrics', []), self.source_transform_key,
                                    reg_channel=reg_channel_index, n_parallel_pairs=n_parallel_pairwise_regs)

        self.pairs_graph = g_reg_computed
        self.msims = msims_reg
        self.pairs = pairs
        self.metrics = metrics
        self.state = RegState.PAIRS_REG
        return {
            'pairs_graph': self.pairs_graph,
            'msims': msims_reg,
            'pairs': pairs,
            'pair_mappings': mappings_dict,
            'metrics': metrics
        }

    def register_global(self, sims, msims, register_indices=None, params=None,
                        pairs_graph=None):
        if register_indices is None:
            if self.register_indices is not None:
                register_indices = self.register_indices
            else:
                register_indices = range(len(msims))

        if pairs_graph is not None:
            g_reg_computed = pairs_graph
        else:
            g_reg_computed = self.pairs_graph

        sim0 = sims[0]
        ndims = si_utils.get_ndim_from_sim(sim0)

        groupwise_resolution_method = params.get('groupwise_resolution_method',
                                                 params.get('registration', {}).get('groupwise_resolution_method', 'global_optimization'))
        groupwise_resolution_kwargs = {}
        if groupwise_resolution_method == 'global_optimization':
           groupwise_resolution_kwargs['transform'] = params.get('transform_type',
                                                                 params.get('registration', {}).get('transform_type'))
           # transform_type options include 'translation', 'rigid', 'affine', 'similarity'

        post_registration_quality_threshold = params.get('post_registration_quality_threshold',
                                                         params.get('registration', {}).get('post_registration_quality_threshold'))
        post_registration_do_quality_filter = (post_registration_quality_threshold is not None)

        n_parallel_pairwise_regs = params.get('n_parallel_pairwise_regs',
                                              params.get('registration', {}).get('n_parallel_pairwise_regs'))
        if n_parallel_pairwise_regs is not None and n_parallel_pairwise_regs == '0':
            n_parallel_pairwise_regs = None

        plot_summary = self.mpl_ui

        # ******* start MVS registration functions

        if post_registration_do_quality_filter:
            # filter edges by quality
            g_reg_computed = mv_graph.filter_edges(
                g_reg_computed,
                threshold=post_registration_quality_threshold,
                weight_key="quality",
            )

        with dask.config.set(scheduler='threads'):
            transforms_dict, groupwise_resolution_info_dict = groupwise_resolution(
                g_reg_computed,
                method=groupwise_resolution_method,
                **groupwise_resolution_kwargs,
            )

        transforms = [
            transforms_dict[iview] for iview in sorted(g_reg_computed.nodes())
        ]

        for imsim, msim in enumerate(msims):
            msi_utils.set_affine_transform(
                msim,
                transforms[imsim],
                transform_key=self.reg_transform_key,
                base_transform_key=self.source_transform_key,
            )

        if plot_summary:
            plot_info = _plot_registration_summaries(
                msims,
                self.source_transform_key,
                self.reg_transform_key,
                g_reg_computed,
                groupwise_resolution_info_dict,
                show_plot=plot_summary,
            )
        else:
            plot_info = {}

        reg_result = {
            "params": transforms,
            "pairwise_registration": {
                "graph": g_reg_computed,
                "metrics": {
                    "qualities": nx.get_edge_attributes(
                        g_reg_computed, "quality"
                    )
                },
                "summary_plot": None if plot_summary is False
                else (
                    plot_info['fig_pair_reg'],
                    plot_info['ax_pair_reg']
                )
            },
            "groupwise_resolution": {
                "metrics": groupwise_resolution_info_dict,
                "summary_plot": None if plot_summary is False
                else (
                    plot_info['fig_group_res'],
                    plot_info['ax_group_res']
                )
            },
        }

        # ******* end MVS registration functions

        # copy transforms from register sims to unmodified sims
        for reg_msim, index in zip(msims, register_indices):
            si_utils.set_sim_affine(
                sims[index],
                msi_utils.get_transform_from_msim(reg_msim, transform_key=self.reg_transform_key),
                transform_key=self.reg_transform_key)

        # set missing transforms
        for sim in sims:
            if self.reg_transform_key not in si_utils.get_tranform_keys_from_sim(sim):
                si_utils.set_sim_affine(
                    sim,
                    param_utils.identity_transform(ndim=ndims, t_coords=[0]),
                    transform_key=self.reg_transform_key)

        mappings = reg_result['params']
        # re-index from subset of sims
        residual_error_dict = reg_result.get('groupwise_resolution', {}).get('metrics', {}).get('residuals', {})
        residual_error_dict = {(register_indices[key[0]], register_indices[key[1]]): value.item()
                               for key, value in residual_error_dict.items()}
        registration_qualities_dict = reg_result.get('pairwise_registration', {}).get('metrics', {}).get('qualities', {})
        registration_qualities_dict = {(register_indices[key[0]], register_indices[key[1]]): value
                                       for key, value in registration_qualities_dict.items()}

        # re-index from subset of sims
        mappings_dict = {index: mapping for index, mapping in zip(register_indices, mappings)}

        reg_channel = params.get('channel', 0)
        metrics = calc_global_metrics(msims, self.source_transform_key, self.reg_transform_key,
                                      params.get('metrics', []), reg_channel=reg_channel, reg_results=reg_result,
                                      n_parallel_pairs=n_parallel_pairwise_regs)

        self.metrics = metrics
        self.state = RegState.GLOBAL_REG
        return {'reg_result': reg_result,
                'mappings': mappings_dict,
                'residual_errors': residual_error_dict,
                'registration_qualities': registration_qualities_dict,
                'metrics': metrics}

    def fuse(self, sims, fusion_method=None, output_spacing='mean', transform_key=None,
             output_filename=None, tile_size=None, ome_version=default_ome_zarr_version):
        if output_filename is not None:
            output_filename = self.output + output_filename
        sim0 = sims[0]
        if transform_key is None:
            transform_key = self.reg_transform_key
        if isinstance(self.extra_metadata, dict):
            z_scale = self.extra_metadata.get('scale', {}).get('z')
            channels = self.extra_metadata.get('channels', [])
        else:
            z_scale = None
            channels = []
        is_channel_overlay = (len(channels) > 1)

        if z_scale is None and self.scales is not None:
            z_scale0 = np.mean([scale.get('z', 0) for scale in self.scales])
            if z_scale0 > 0:
                z_scale = z_scale0
        if z_scale is None:
            if 'z' in sim0.dims:
                diffs = np.diff(sorted(set([si_utils.get_origin_from_sim(sim).get('z', 0) for sim in sims])))
                if len(diffs) > 0:
                    z_scale = min(diffs)

        z_positions = [position.get('z') for position in self.positions if 'z' in position]
        if len(set(z_positions)) > 1:
            sims = make_sims_3d(sims, z_scale=z_scale, positions=self.positions)

        output_stack_properties = calc_output_properties(sims, transform_key,
                                                         output_spacing_method=output_spacing, z_scale=z_scale)

        if self.verbose:
            logging.info(f'Output stack: {numpy_to_native(output_stack_properties)}')
        data_size = np.prod(list(output_stack_properties['shape'].values())) * sim0.dtype.itemsize
        logging.info(f'Fusing {print_hbytes(data_size)}')

        saving_zarr = False
        if is_channel_overlay:
            # convert to multichannel images
            channel_sims = [fusion.fuse(
                [sim],
                transform_key=transform_key,
                output_stack_properties=output_stack_properties
            ) for sim in sims]
            channel_sims = [sim.assign_coords({'c': [channels[simi]['label']]}) for simi, sim in enumerate(channel_sims)]
            fused_image = xr.combine_nested([sim.rename() for sim in channel_sims], concat_dim='c', combine_attrs='override')
        else:
            fuse_func = self.create_fusion_method(fusion_method, sim0)
            if fuse_func:
                saving_zarr = output_filename is not None
                output_chunksize = None
                if saving_zarr:
                    if not output_filename.lower().endswith('.zarr'):
                        output_filename += zarr_extension
                    zarr_options = {'ome_zarr': saving_zarr, 'ngff_version': ome_version}
                    if tile_size is not None:
                        if not isinstance(tile_size, (list, tuple)):
                            tile_size = [tile_size] * 2
                        output_chunksize = xyz_to_dict(tile_size)
                        if 'z' in output_stack_properties['shape'] and 'z' not in output_chunksize:
                            output_chunksize['z'] = 1
                else:
                    zarr_options = None
                with dask.config.set(scheduler='threads'):
                    fused_image = fusion.fuse(
                        sims,
                        fusion_func=fuse_func,
                        transform_key=transform_key,
                        output_stack_properties=output_stack_properties,
                        output_zarr_url=output_filename,
                        zarr_options=zarr_options,
                        output_chunksize=output_chunksize
                    )
            else:
                fused_image = sims
        return fused_image, saving_zarr

    def save_pair_mappings(self, mappings, qualities, bboxes):
        pair_mappings_filename = self.output + self.output_params.get('pair_mappings', default_pair_mappings_name)
        file_labels = self.file_labels
        output_mappings = {f'{file_labels[keys[0]]}-{file_labels[keys[1]]}':
                               {'mapping': np.array(mapping.sel(t=0)).tolist(),
                                default_quality_key: float(qualities[keys]),
                                'bbox': bboxes[keys]}
                           for keys, mapping in mappings.items()
                           if keys in qualities}
        export_json(pair_mappings_filename, output_mappings)

    def save_mappings(self, mappings):
        mappings_filename = self.output + self.output_params.get('mappings', default_mappings_name)
        output_mappings = {self.file_labels[key]: np.array(mapping.sel(t=0)).tolist()
                           for key, mapping in mappings.items()}
        export_json(mappings_filename, output_mappings)

    def save_metrics(self, metrics):
        metrics_filename = self.output + metrics_name
        output_metrics = {f'{self.file_labels[keys[0]]}-{self.file_labels[keys[1]]}':
                              {metric: float(value) for metric, value in metric_dict[self.reg_transform_key].items()}
                          for keys, metric_dict in metrics['pairs'].items() if metric_dict[self.reg_transform_key]}
        export_json(metrics_filename, output_metrics)

    def create_preview(self, output_filename=None, nom_sims=None, transform_key=None):
        output_params = self.params_general['output']
        preview_scale = output_params.get('preview_scale', 16)
        is_stack = ('stack' in self.operation)
        if isinstance(self.extra_metadata, dict):
            z_scale = self.extra_metadata.get('scale', {}).get('z')
        else:
            z_scale = None

        sims = self.init_sims(target_scale=preview_scale, store=False)
        if is_stack:
            sims = make_sims_3d(sims, z_scale, self.positions)

        if nom_sims is not None:
            if sims[0].sizes['x'] >= nom_sims[0].sizes['x']:
                logging.warning('Unable to generate scaled down preview due to lack of source pyramid sizes')
                return None

            if transform_key is not None and transform_key != self.source_transform_key:
                copy_transforms(nom_sims, sims, transform_key)
        fusion_method = self.fusion_params.get('method', '')
        fused_image, is_saved = self.fuse(sims, fusion_method=fusion_method, transform_key=transform_key,
                                          output_spacing='max', output_filename=output_filename)
        if output_filename and (not is_saved or 'tif' in output_params.get('preview')):
            self.save(output_filename, fused_image.squeeze(), transform_key=transform_key,
                      format=output_params.get('preview'), ome_version=output_params.get('ome_version'))
        return fused_image

    def save(self, output_filename, data, format=zarr_extension, transform_key=None, translations0=None,
             tile_size=None, compression=None, pyramid_downsample=2, npyramid_add=0, ome_version=default_ome_zarr_version):
        if output_filename is not None:
            output_filename = self.output + output_filename
        if isinstance(self.extra_metadata, dict):
            channels = self.extra_metadata.get('channels', [])
        else:
            channels = []
        save_image(output_filename, data, format,
                   transform_key=transform_key, channels=channels, translations0=translations0,
                   tile_size=tile_size, compression=compression,
                   pyramid_downsample=pyramid_downsample, npyramid_add=npyramid_add,
                   ome_version=ome_version,
                   verbose=self.verbose)

    def save_video(self, output, sims, fused_image):
        logging.info('Creating transition video...')
        pixel_size = [si_utils.get_spacing_from_sim(sims[0]).get(dim, 1) for dim in 'xy']
        params = self.params
        nframes = params.get('frames', 1)
        spacing = params.get('spacing', [1.1, 1])
        scale = params.get('scale', 1)
        transition_filename = output + 'transition'
        video = Video(transition_filename + '.mp4', fps=params.get('fps', 1))
        positions0 = np.array([si_utils.get_origin_from_sim(sim, asarray=True) for sim in sims])
        center = np.mean(positions0, 0)
        window = get_image_window(fused_image)

        max_size = None
        acum = 0
        for framei in range(nframes):
            c = (1 - np.cos(framei / (nframes - 1) * 2 * math.pi)) / 2
            acum += c / (nframes / 2)
            spacing1 = spacing[0] + (spacing[1] - spacing[0]) * acum
            for sim, position0 in zip(sims, positions0):
                transform = param_utils.identity_transform(ndim=2, t_coords=[0])
                transform[0][:2, 2] += (position0 - center) * spacing1
                si_utils.set_sim_affine(sim, transform, transform_key=self.transition_transform_key)
            frame = fusion.fuse(sims, transform_key=self.transition_transform_key).squeeze()
            frame = float2int_image(normalise_values(frame, window[0], window[1]))
            frame = cv.resize(np.asarray(frame), None, fx=scale, fy=scale)
            if max_size is None:
                max_size = frame.shape[1], frame.shape[0]
                video.size = max_size
            frame = image_reshape(frame, max_size)
            save_tiff(transition_filename + f'{framei:04d}.tiff', frame, None, pixel_size)
            video.write(frame)

        video.close()

    def get_metrics(self, metric=None, pair=None):
        metrics = self.metrics
        if pair is not None:
            if isinstance(pair, np.ndarray):
                pair = pair.tolist()
                pair = tuple(pair)
            metrics = metrics.get('pairs', {}).get(pair, {})
        else:
            if 'summary' in metrics:
                metrics = metrics['summary']
        transform_keys = metrics.keys()
        if len(transform_keys) > 0:
            transform_key = list(transform_keys)[-1]
            metrics = metrics.get(transform_key, {})
        if metric is not None:
            return metrics.get(metric)
        else:
            return metrics

    def output_exists(self, output_filename, output_format):
        if not output_format.startswith('.'):
            output_format = '.' + output_format
        output_filename = self.output + output_filename + output_format
        if output_format == zarr_extension:
            return (os.path.exists(os.path.join(output_filename, '.zattrs')) or
                    os.path.exists(os.path.join(output_filename, 'zarr.json')))
        else:
            return os.path.exists(output_filename)
