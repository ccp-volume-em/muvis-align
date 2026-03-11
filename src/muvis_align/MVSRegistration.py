# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

from contextlib import nullcontext
import dask
from dask.diagnostics import ProgressBar
import logging
from multiview_stitcher import registration, vis_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
from multiview_stitcher.registration import get_overlap_bboxes, sims_to_intrinsic_coord_system
import numpy as np
import os.path
import shutil
from skimage.transform import resize
import xarray as xr

from src.muvis_align.Timer import Timer
from src.muvis_align.constants import *
from src.muvis_align.image.Video import Video
from src.muvis_align.image.flatfield import flatfield_correction
from src.muvis_align.image.ome_helper import save_image, exists_output_image
from src.muvis_align.image.ome_tiff_helper import save_tiff
from src.muvis_align.image.source_helper import create_dask_source
from src.muvis_align.image.util import *
from src.muvis_align.metrics import calc_ncc, calc_ssim
from src.muvis_align.util import *


dask.config.set(scheduler='threads')


class MVSRegistration:
    def __init__(self, params_general):
        super().__init__()
        self.params_general = params_general

        params_logging = self.params_general.get('logging', {})
        self.verbose = params_logging.get('verbose', False)
        self.logging_dask = params_logging.get('dask', False)
        self.logging_time = params_logging.get('time', False)
        self.ui = self.params_general.get('ui', '')
        self.mpl_ui = ('mpl' in self.ui or 'plot' in self.ui)
        self.napari_ui = ('napari' in self.ui)
        self.source_transform_key = 'source_metadata'
        self.reg_transform_key = 'registered'
        self.transition_transform_key = 'transition'
        self.sources = None

    def init_operation(self, fileset_label, filenames, params, global_rotation=None, global_center=None):
        self.fileset_label = fileset_label
        self.filenames = filenames
        self.file_labels = get_unique_file_labels(filenames)
        self.params = params
        self.global_rotation = global_rotation
        self.global_center = global_center

        if filenames:
            input_dir = os.path.dirname(filenames[0])
            parts = split_numeric_dict(filenames[0])
            output_pattern = params['output'].format_map(parts)
        else:
            input_pattern = params['input']
            if isinstance(input_pattern, list):
                input_pattern = input_pattern[0]
            input_dir = os.path.dirname(input_pattern)
            output_pattern = params['output']
        self.output = os.path.join(input_dir, output_pattern)    # preserve trailing slash: do not use os.path.normpath()

    def run_operation(self, fileset_label, filenames, params, global_rotation=None, global_center=None):
        self.init_operation(fileset_label, filenames, params, global_rotation, global_center)
        with ProgressBar(minimum=60, dt=1) if self.logging_dask else nullcontext():
            return self._run_operation()

    def _run_operation(self):
        params = self.params
        filenames = self.filenames
        file_labels = self.file_labels
        output = self.output

        operation = params['operation']
        overlap_threshold = params.get('overlap_threshold', 0.5)
        source_metadata = import_metadata(params.get('source_metadata', {}), input_path=params['input'])
        save_images = params.get('save_images', True)
        target_scale = params.get('scale')
        extra_metadata = import_metadata(params.get('extra_metadata', {}), input_path=params['input'])
        z_scale = extra_metadata.get('scale', {}).get('z')
        channels = extra_metadata.get('channels', [])
        normalise_orientation = 'norm' in source_metadata

        show_original = self.params_general.get('show_original', False)
        output_params = self.params_general.get('output', {})
        clear = output_params.get('clear', False)
        overwrite = output_params.get('overwrite', True)

        is_stack = ('stack' in operation)
        is_3d = ('3d' in operation)
        is_simple_stack = is_stack and not is_3d
        is_transition = ('transition' in operation)
        is_channel_overlay = (len(channels) > 1)

        mappings_header = ['id','x_pixels', 'y_pixels', 'z_pixels', 'x', 'y', 'z', 'rotation']

        if len(filenames) == 0:
            logging.warning('Skipping (no images)')
            return False

        registered_fused_filename = output + registered_name
        mappings_filename = output + params.get('mappings', default_mappings_name)

        output_dir = os.path.dirname(output)
        if not overwrite and exists_output_image(registered_fused_filename):
            logging.warning(f'Skipping existing output {os.path.normpath(output_dir)}')
            return False
        if clear:
            shutil.rmtree(output_dir, ignore_errors=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with Timer('init sims', self.logging_time):
            sims = self.init_sims(target_scale=target_scale)

        if not z_scale:
            z_scale = self.scales[0].get('z', 1)

        with Timer('pre-process', self.logging_time):
            register_sims, indices = self.preprocess(sims, params)

        data = []
        for label, sim, scale in zip(file_labels, sims, self.scales):
            position, rotation = get_data_mapping(sim, transform_key=self.source_transform_key)
            position_pixels = {dim: position[dim] / float(scale[dim]) for dim in position.keys()}
            row = [label] + dict_to_xyz(position_pixels, add_zeros=True) + dict_to_xyz(position, add_zeros=True) + [rotation]
            data.append(row)
        export_csv(output + prereg_mappings_name, data, header=mappings_header)

        if show_original:
            # before registration:
            logging.info('Exporting original...')
            original_positions_filename = output + original_positions_name

            with Timer('plot positions', self.logging_time):
                vis_utils.plot_positions(sims, transform_key=self.source_transform_key,
                                         use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                         show_plot=self.mpl_ui, output_filename=original_positions_filename)
                plt_close()

            if self.napari_ui:
                shapes = [get_sim_shape_2d(sim, transform_key=self.source_transform_key) for sim in sims]
                self.update_napari_signal.emit(f'{self.fileset_label} original', shapes, file_labels)

            if save_images:
                if output_params.get('thumbnail'):
                    with Timer('create thumbnail', self.logging_time):
                        self.save_thumbnail(output + original_thumbnail_name,
                                            nom_sims=sims,
                                            transform_key=self.source_transform_key)

                sims2 = [sim.copy() for sim in sims]
                sims2 = make_sims_3d(sims2, z_scale, self.positions)

                original_fused_filename = output + original_name
                original_fused, is_saved = self.fuse(sims2, transform_key=self.source_transform_key)
                if not is_saved or 'tif' in output_params.get('format'):
                    self.save(original_fused_filename, original_fused, output_params.get('format'),
                              transform_key=self.source_transform_key)

        if len(filenames) == 1 and save_images:
            logging.warning('Skipping registration (single image)')
            self.save(registered_fused_filename, sims[0], output_params.get('format'), translations0=self.positions)
            return False

        _, has_overlaps = self.validate_overlap(sims, file_labels, is_simple_stack, is_simple_stack or is_channel_overlay)
        overall_overlap = np.mean(has_overlaps)
        if overall_overlap < overlap_threshold:
            raise ValueError(f'Not enough overlap: {overall_overlap * 100:.1f}%')

        if not overwrite and os.path.exists(mappings_filename):
            logging.info('Loading registration mappings...')
            # load registration mappings
            mappings = import_json(mappings_filename)
            # copy transforms to sims
            for sim, label in zip(sims, file_labels):
                mapping = param_utils.affine_to_xaffine(np.array(mappings[label]))
                if is_stack:
                    transform = param_utils.identity_transform(ndim=3)
                    transform.loc[{dim: mapping.coords[dim] for dim in mapping.dims}] = mapping
                else:
                    transform = mapping
                si_utils.set_sim_affine(sim, transform, transform_key=self.reg_transform_key)
            if is_stack:
                sims = make_sims_3d(sims, z_scale, self.positions)
        else:
            if 'register' in operation:
                with Timer('register', self.logging_time):
                    results = self.register(sims, register_sims, indices)

            if is_stack:
                results['sims'] = make_sims_3d(results['sims'], z_scale, self.positions)

            reg_result = results['reg_result']
            sims = results['sims']
            mappings = results['mappings']

            logging.info('Exporting registered...')
            metrics = self.calc_metrics(results, file_labels)
            logging.info(metrics['summary'])
            output_mappings = {file_labels[index]: np.array(mapping.sel(t=0)).tolist() for index, mapping in mappings.items()}
            export_json(mappings_filename, output_mappings)
            export_json(output + metrics_name, metrics)
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

        registered_positions_filename = output + registered_positions_name
        if self.reg_transform_key in sims[0].transforms:
            with Timer('plot positions', self.logging_time):
                vis_utils.plot_positions(sims, transform_key=self.reg_transform_key,
                                         use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                         show_plot=self.mpl_ui, output_filename=registered_positions_filename)
                plt_close()

            if self.napari_ui:
                shapes = [get_sim_shape_2d(sim, transform_key=self.reg_transform_key) for sim in sims]
                self.update_napari_signal.emit(f'{self.fileset_label} registered', shapes, file_labels)

            if save_images:
                if output_params.get('thumbnail'):
                    with Timer('create thumbnail', self.logging_time):
                        self.save_thumbnail(output + registered_thumbnail_name,
                                            nom_sims=sims,
                                            transform_key=self.reg_transform_key)

                with Timer('fuse image', self.logging_time):
                    #fused_image, is_saved = self.fuse(sims, output_filename=registered_fused_filename)
                    fused_image, is_saved = self.fuse(sims)

                if not is_saved or 'tif' in output_params.get('format'):
                    logging.info('Saving fused image...')
                    with Timer('save fused image', self.logging_time):
                        self.save(registered_fused_filename, fused_image, output_params.get('format'),
                                  transform_key=self.reg_transform_key, translations0=self.positions)

            if is_transition:
                self.save_video(output, sims, fused_image)

        return True

    def init_sources(self):
        source_metadata0 = import_metadata(self.params.get('source_metadata', 'source'), input_path=self.params['input'])
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

    def init_sims(self, target_scale=None):
        operation = self.params['operation']
        source_metadata = import_metadata(self.params.get('source_metadata', 'source'), input_path=self.params['input'])
        chunk_size = self.params_general.get('chunk_size', [1024, 1024])
        extra_metadata = import_metadata(self.params.get('extra_metadata', {}), input_path=self.params['input'])
        z_scale = extra_metadata.get('scale', {}).get('z')

        if len(self.filenames) == 0:
            raise ValueError('No input files')

        logging.info('Initialising sims...')
        if self.sources is None:
            self.init_sources()
        sources = self.sources
        source0 = sources[0]
        images = []
        sims = []
        scales = []
        translations = []
        rotations = []

        is_stack = ('stack' in operation)
        has_z_size = (source0.get_size().get('z', 0) > 0)

        output_order = 'zyx' if has_z_size else 'yx'
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
                level, rescale, scale = get_level_from_scale(source.scales, scale, target_scale)
            if 'invert' in source_metadata:
                translation['x'] = -translation['x']
                translation['y'] = -translation['y']
            if 'z' in translation:
                z_position = translation['z']
            else:
                z_position = 0
            if last_z_position is not None and z_position != last_z_position:
                different_z_positions = True
                delta_zs.append(z_position - last_z_position)
            if 'rotation' in source_metadata:
                rotation = source_metadata['rotation']
            if self.global_rotation is not None:
                rotation = self.global_rotation

            dask_data = source.get_data(level=level)
            if rescale != 1:
                new_shape = [int(size / rescale) if dim in 'xy' else 1
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
        scales2 = []
        translations2 = []
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
            translation2 = translation.copy()

            sim = si_utils.get_sim_from_array(
                image,
                dims=list(output_order),
                scale=scale,
                translation=translation2,
                affine=transform,
                transform_key=self.source_transform_key,
                c_coords=channel_labels
            )
            if len(sim.chunksizes.get('x')) == 1 and len(sim.chunksizes.get('y')) == 1:
                sim = sim.chunk(xyz_to_dict(chunk_size))
            sims.append(sim)
            scales2.append(scale)
            translations2.append(translation)

        self.scales = scales2
        self.positions = translations2
        self.rotations = rotations
        return sims

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

    def preprocess(self, sims, params):
        flatfield_quantiles = params.get('flatfield_quantiles')
        normalisation = params.get('normalisation', '')
        gaussian_sigma = params.get('gaussian_sigma')
        filter_foreground = params.get('filter_foreground', False)

        # normalise pixel size: take max pixel size
        max_scale = {dim: max(scale[dim] for scale in self.scales) for dim in 'xy'}
        #needs_reinit = False
        #for source in self.sources:
        #    if not np.all([np.isclose(source.get_pixel_size()[dim], max_scale[dim]) for dim in 'xy']):
        #        needs_reinit = True
        scales0 = self.scales
        #if needs_reinit:
        #    sims = self.init_sims(target_scale=max_scale)

        if filter_foreground:
            foreground_map = calc_foreground_map(sims)
        else:
            foreground_map = None
        if flatfield_quantiles is not None:
            logging.info('Flat-field correction...')
            new_sims = [None] * len(sims)
            for sim_indices in group_sims_by_z(sims, self.positions):
                sims_z_set = [sims[i] for i in sim_indices]
                foreground_map_z_set = [foreground_map[i] for i in sim_indices] if foreground_map is not None else None
                new_sims_z_set = flatfield_correction(sims_z_set, self.source_transform_key, flatfield_quantiles,
                                                      foreground_map=foreground_map_z_set)
                for sim_index, sim in zip(sim_indices, new_sims_z_set):
                    new_sims[sim_index] = sim
            sims = new_sims

        if gaussian_sigma:
            logging.info('Applying Gaussian filtering...')
            new_sims = []
            for sim, scale0 in zip(sims, scales0):
                # factor in original pixel size for gaussian sigma value
                scale = np.mean(list(scale0.values())) / np.mean(list(max_scale.values()))
                sigma = gaussian_sigma * (scale ** (1 / 3))
                new_sims.append(gaussian_filter_sim(sim, self.source_transform_key, sigma))
            sims = new_sims

        if normalisation:
            use_global = ('global' in normalisation)
            if use_global:
                logging.info('Normalising (global)...')
            else:
                logging.info('Normalising (individual)...')
            sims = normalise_sims(sims, self.source_transform_key, use_global=use_global)

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
        else:
            indices = range(len(sims))
        return sims, indices

    def create_registration_method(self, sim0):
        registration_method = None
        pairwise_reg_func_kwargs = None

        reg_params = self.params.get('method', {})
        if isinstance(reg_params, dict):
            reg_method = reg_params.get('name', '').lower()
        elif isinstance(reg_params, str):
            reg_method = reg_params.lower()
        else:
            reg_method = ''
        debug = self.params_general.get('debug', False)

        if 'cpd' in reg_method:
            from src.muvis_align.registration_methods.RegistrationMethodCPD import RegistrationMethodCPD
            registration_method = RegistrationMethodCPD(sim0, reg_params, debug)
            pairwise_reg_func = registration_method.registration
        elif 'feature' in reg_method or 'orb' in reg_method or 'sift' in reg_method:
            if 'cv' in reg_method:
                from src.muvis_align.registration_methods.RegistrationMethodCvFeatures import RegistrationMethodCvFeatures
                registration_method = RegistrationMethodCvFeatures(sim0, reg_params, debug)
            else:
                from src.muvis_align.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures
                registration_method = RegistrationMethodSkFeatures(sim0, reg_params, debug)
            pairwise_reg_func = registration_method.registration
        elif 'ant' in reg_method:
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

        return reg_method, pairwise_reg_func, pairwise_reg_func_kwargs

    def create_fusion_method(self, sim0):
        debug = self.params_general.get('debug', False)
        fusion_params = self.params.get('fusion', '')
        if isinstance(fusion_params, dict):
            fusion_method = fusion_params.get('method', fusion_params.get('name', '')).lower()
        else:
            fusion_method = fusion_params.lower()

        if 'compos' in fusion_method:
            fusion_method = None
            fuse_func = None
        elif 'exclus' in fusion_method:
            from src.muvis_align.fusion_methods.FusionMethodExclusive import FusionMethodExclusive
            fusion_method = FusionMethodExclusive(sim0, fusion_params, debug)
            fuse_func = fusion_method.fusion
        elif 'add' in fusion_method:
            from src.muvis_align.fusion_methods.FusionMethodAdditive import FusionMethodAdditive
            fusion_method = FusionMethodAdditive(sim0, fusion_params, debug)
            fuse_func = fusion_method.fusion
        else:
            fuse_func = fusion.simple_average_fusion

        self.fusion_method = fusion_method

        return fuse_func

    def register(self, sims, register_sims=None, indices=None):
        params = self.params
        sim0 = sims[0]
        ndims = si_utils.get_ndim_from_sim(sim0)

        operation = params['operation']
        pairing = params.get('pairing', '')
        post_registration_quality_threshold = params.get('post_registration_quality_threshold')
        n_parallel_pairwise_regs = params.get('n_parallel_pairwise_regs')

        is_stack = ('stack' in operation)
        is_3d = ('3d' in operation)

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

        # Pass registration through metrics method
        #from src.muvis_align.registration_methods.RegistrationMetrics import RegistrationMetrics
        #registration_metrics = RegistrationMetrics(sim0, pairwise_reg_function)
        #pairwise_reg_function = registration_metrics.registration
        # TODO: extract metrics from registration_metrics

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

            if indices is None:
                indices = range(len(register_msims))

            # copy transforms from register sims to unmodified sims
            for reg_msim, index in zip(register_msims, indices):
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
                residual_error_dict = {(indices[key[0]], indices[key[1]]): value.item()
                                       for key, value in residual_error_dict.items()}
                registration_qualities_dict = reg_result.get('pairwise_registration', {}).get('metrics', {}).get('qualities', {})
                registration_qualities_dict = {(indices[key[0]], indices[key[1]]): value
                                               for key, value in registration_qualities_dict.items()}
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
        mappings_dict = {index: mapping for index, mapping in zip(indices, mappings)}

        return {'reg_result': reg_result,
                'mappings': mappings_dict,
                'residual_errors': residual_error_dict,
                'registration_qualities': registration_qualities_dict,
                'sims': sims,
                'pairs': pairs}

    def fuse(self, sims, transform_key=None, output_filename=None, thumbnail=False):
        sim0 = sims[0]
        if transform_key is None:
            transform_key = self.reg_transform_key
        extra_metadata = import_metadata(self.params.get('extra_metadata', {}), input_path=self.params['input'])
        z_scale = extra_metadata.get('scale', {}).get('z')
        channels = extra_metadata.get('channels', [])
        is_channel_overlay = (len(channels) > 1)
        if thumbnail:
            output_spacing = 'max'
        else:
            output_spacing = self.params.get('output_spacing')

        if z_scale is None and self.scales is not None:
            z_scale0 = np.mean([scale.get('z', 0) for scale in self.scales])
            if z_scale0 > 0:
                z_scale = z_scale0
        if z_scale is None:
            if 'z' in sim0.dims:
                diffs = np.diff(sorted(set([si_utils.get_origin_from_sim(sim).get('z', 0) for sim in sims])))
                if len(diffs) > 0:
                    z_scale = min(diffs)

        output_stack_properties = calc_output_properties(sims, transform_key,
                                                         output_spacing=output_spacing, z_scale=z_scale)
        if 'z' in output_stack_properties['shape']:
            z_positions = sorted(set([si_utils.get_origin_from_sim(sim).get('z', 0) for sim in sims]))
            z_shape = len(z_positions)
            if z_shape <= 1:
                z_shape = len(sims)
            output_stack_properties['shape']['z'] = z_shape

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
            fuse_func = self.create_fusion_method(sim0)
            if fuse_func:
                saving_zarr = output_filename is not None
                if saving_zarr and not output_filename.lower().endswith('.zarr'):
                    output_filename += '.ome.zarr'
                    ome_version = str(self.params_general.get('output', {}).get('ome_version', '0.4'))
                    zarr_options = {'ome_zarr': saving_zarr, 'ngff_version': ome_version}
                else:
                    zarr_options = None
                fused_image = fusion.fuse(
                    sims,
                    fusion_func=fuse_func,
                    transform_key=transform_key,
                    output_stack_properties=output_stack_properties,
                    output_zarr_url=output_filename,
                    zarr_options=zarr_options
                )
                if saving_zarr:
                    open(output_filename.rstrip('.zarr').rstrip('.ome'), 'w')
            else:
                fused_image = sims
        return fused_image, saving_zarr

    def save_thumbnail(self, output_filename, nom_sims=None, transform_key=None):
        params = self.params
        output_params = self.params_general['output']
        thumbnail_scale = output_params.get('thumbnail_scale', 16)
        is_stack = ('stack' in params['operation'])
        extra_metadata = import_metadata(self.params.get('extra_metadata', {}), input_path=self.params['input'])
        z_scale = extra_metadata.get('scale', {}).get('z')

        sims = self.init_sims(target_scale=thumbnail_scale)
        if is_stack:
            sims = make_sims_3d(sims, z_scale, self.positions)

        if nom_sims is not None:
            if sims[0].sizes['x'] >= nom_sims[0].sizes['x']:
                logging.warning('Unable to generate scaled down thumbnail due to lack of source pyramid sizes')
                return

            if transform_key is not None and transform_key != self.source_transform_key:
                for nom_sim, sim in zip(nom_sims, sims):
                    si_utils.set_sim_affine(sim,
                                            si_utils.get_affine_from_sim(nom_sim, transform_key=transform_key),
                                            transform_key=transform_key)
        fused_image, is_saved = self.fuse(sims, transform_key=transform_key, thumbnail=True)
        if not is_saved or 'tif' in output_params.get('thumbnail'):
            self.save(output_filename, fused_image.squeeze(), output_params.get('thumbnail'), transform_key=transform_key)

    def save(self, output_filename, data, format=zarr_extension, transform_key=None, translations0=None):
        extra_metadata = import_metadata(self.params.get('extra_metadata', {}), input_path=self.params['input'])
        channels = extra_metadata.get('channels', [])
        general_output_params = self.params_general.get('output', {})
        save_image(output_filename, data, format, params=general_output_params,
                   transform_key=transform_key, channels=channels, translations0=translations0,
                   verbose=self.verbose)

    def calc_overlap_metrics(self, results):
        nccs = {}
        ssims = {}
        sims = results['sims']
        pairs = results['pairs']
        if pairs is None:
            origins = np.array([get_sim_position_final(sim) for sim in sims])
            sizes = [get_sim_physical_size(sim) for sim in sims]
            pairs, _ = get_pairs(origins, sizes)
        for pair in pairs:
            try:
                # experimental; in case fail to extract overlap images
                overlap_sims = self.get_overlap_images(sims[pair[0]], sims[pair[1]], self.reg_transform_key)
                nccs[pair] = calc_ncc(overlap_sims[0], overlap_sims[1])
                ssims[pair] = calc_ssim(overlap_sims[0], overlap_sims[1])
                #frcs[pair] = calc_frc(overlap_sims[0], overlap_sims[1])
            except Exception as e:
                logging.exception(e)
                #logging.warning(f'Failed to calculate resolution metric')
        return {'ncc': nccs, 'ssim': ssims}

    def get_overlap_images(self, sim1, sim2, transform_key, overlap_tolerance=0):
        sims = [sim1.squeeze(), sim2.squeeze()]
        # functionality copied from registration.register_pair_of_msims()
        spatial_dims = si_utils.get_spatial_dims_from_sim(sim1)
        lowers, uppers = get_overlap_bboxes(
            sims[0],
            sims[1],
            input_transform_key=transform_key,
            output_transform_key=None,
            overlap_tolerance=overlap_tolerance,
        )

        reg_sims_spacing = [
            si_utils.get_spacing_from_sim(sim) for sim in sims
        ]

        tol = 1e-6
        overlaps_sims = [
            sim.sel(
                {
                    # add spacing to include bounding pixels
                    dim: slice(
                        lowers[isim][idim] - tol - reg_sims_spacing[isim][dim],
                        uppers[isim][idim] + tol + reg_sims_spacing[isim][dim],
                    )
                    for idim, dim in enumerate(spatial_dims)
                },
            )
            for isim, sim in enumerate(sims)
        ]

        sims_pixel_space = sims_to_intrinsic_coord_system(
            overlaps_sims[0],
            overlaps_sims[1],
            transform_key=transform_key,
            overlap_bboxes=(lowers, uppers),
        )

        fixed_data = sims_pixel_space[0].data
        moving_data = sims_pixel_space[1].data

        fixed_data = xr.DataArray(fixed_data, dims=spatial_dims)
        moving_data = xr.DataArray(moving_data, dims=spatial_dims)

        return fixed_data, moving_data

    def calc_metrics(self, results, labels):
        var = None
        distances = [np.linalg.norm(param_utils.translation_from_affine(mapping.sel(t=0)))
                     for mapping in results['mappings'].values()]
        if len(distances) > 2:
            # Coefficient of variation
            mean_distance = np.mean(distances)
            if mean_distance > 0:
                cvar = np.std(distances) / mean_distance
                var = cvar
        if var is None:
            size = get_sim_physical_size(results['sims'][0])
            norm_distance = np.sum(distances) / np.linalg.norm(list(size.values()))
            var = norm_distance

        residual_errors = {labels[key[0]] + ' - ' + labels[key[1]]: value
                           for key, value in results['residual_errors'].items()}
        if len(residual_errors) > 0:
            residual_error = np.nanmean(list(residual_errors.values()))
        else:
            residual_error = 1

        registration_qualities = {labels[key[0]] + ' - ' + labels[key[1]]: value.item()
                                  for key, value in results['registration_qualities'].items()}
        if len(registration_qualities) > 0:
            registration_quality = np.nanmean(list(registration_qualities.values()))
        else:
            registration_quality = 0

        #overlap_metrics = self.calc_overlap_metrics(results)

        #nccs = {labels[key[0]] + ' - ' + labels[key[1]]: value
        #         for key, value in overlap_metrics['ncc'].items()}
        #ncc = np.nanmean(list(nccs.values()))

        #ssims = {labels[key[0]] + ' - ' + labels[key[1]]: value
        #         for key, value in overlap_metrics['ssim'].items()}
        #ssim = np.nanmean(list(ssims.values()))

        summary = (f'Residual error: {residual_error:.3f}'
                   f' Registration quality: {registration_quality:.3f}'
        #           f' NCC: {ncc:.3f}'
        #           f' SSIM: {ssim:.3f}'
                   f' Variation: {var:.3f}')

        return {'variation': var,
                'residual_error': residual_error,
                'residual_errors': residual_errors,
                'registration_quality': registration_quality,
                'registration_qualities': registration_qualities,
         #       'ncc': ncc,
         #       'nccs': nccs,
         #       'ssim': ssim,
         #       'ssims': ssims,
                'summary': summary}

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
