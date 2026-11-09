# https://stackoverflow.com/questions/62806175/xarray-combine-by-coords-return-the-monotonic-global-index-error
# https://github.com/pydata/xarray/issues/8828

from contextlib import contextmanager, nullcontext
import copy
import dask
import functools
import threading
import time
from dask.array.core import PerformanceWarning
from dask.diagnostics import ProgressBar
from enum import Enum, auto
import logging
from multiview_stitcher import param_utils, registration, vis_utils
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
from multiview_stitcher.param_resolution import groupwise_resolution
from multiview_stitcher.registration import compute_pairwise_registrations, _plot_registration_summaries
from datetime import datetime
from fair_metadata_integration import create_ro_crate, create_zarr_ro_crate
import importlib.metadata
import importlib.resources
import networkx as nx
import numpy as np
import os.path
from pathlib import Path
import shutil
from skimage.transform import resize
import warnings
import xarray as xr

from muvis_align.constants import *
from muvis_align.file.transforms import write_transforms, read_transforms
from muvis_align.GlobalOptProgress import GlobalOptProgress
from muvis_align.image.Video import Video
from muvis_align.image.flatfield import apply_msim_flatfield_model, calc_msims_flatfield_model
from muvis_align.image.ome_helper import save_image
from muvis_align.image.ome_zarr_helper import save_ome_multiscale_levels, save_ome_zarr_scene
from muvis_align.image.ome_tiff_helper import save_tiff
from muvis_align.image.source_helper import create_image_source, get_missing_metadata_summary
from muvis_align.image.util import *
from muvis_align.fusion_blocks import export_levels, fuse_levels_to_ome_zarr, level_blocks, written_voxels
from muvis_align.metrics import (calc_pair_metrics, calc_global_metrics, create_metric_methods, quality_to_scalar,
                                 remap_pair_metrics)
from muvis_align.robust_resolution import ROBUST_LINEAR, default_robust_rounds, find_reference_view
from muvis_align.split_registration import (SPLIT, channel_groups, group_label, register_groups, split_groups,
                                            within_group_pairs)
from muvis_align.Timer import Timer
from muvis_align.util import *


class RegState(Enum):
    UNINIT = auto()
    INIT = auto()
    SIMS_INIT = auto()
    PAIRS_REG = auto()
    GLOBAL_REG = auto()
    FUSED = auto()


def normalisation_enabled(value):
    # a project file carries this as text, where 'none'/'no'/'false'/empty means off
    if isinstance(value, str) and value.lower() in ['false', 'no', 'none', '']:
        return False
    return bool(value)


def registration_method_kind(method):
    """The pairwise registration a method name selects: 'cpd', 'feature', 'elastix', 'ants' or 'phase_correlation'."""
    method = (method or '').lower()
    if 'cpd' in method:
        return 'cpd'
    if 'feature' in method or 'orb' in method or 'sift' in method:
        return 'feature'
    if 'elastix' in method:
        return 'elastix'
    if 'ant' in method:
        return 'ants'
    return 'phase_correlation'


def metric_value_range(metric, method=None):
    """A metric's (low, high) value range. Quality is the registration method's own: multiview_stitcher scores phase
    correlation, Elastix and ANTs by Spearman correlation, the feature methods and CPD by a 0..1 match rate."""
    if metric == default_quality_key:
        is_correlation = registration_method_kind(method) in ('phase_correlation', 'elastix', 'ants')
        return (-1, 1) if is_correlation else (0, 1)
    return metric_value_ranges.get(metric, (0, 1))


class _DeferredQuality:
    """A link quality whose computation waits until it is known to be kept."""
    __slots__ = ('func', 'args')

    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def value(self):
        return self.func(*self.args)


@contextmanager
def deferred_link_quality(pairwise_reg_func):
    """Defer link qualities for phase correlation, which scores every candidate shift but keeps only the chosen
    one's, which resolve_deferred_quality() then computes. Other methods score at once: their images may be views of
    buffers freed once they return (Elastix's resampled image)."""
    if pairwise_reg_func is not registration.phase_correlation_registration:
        yield
        return
    original = registration.link_quality_metric_func
    registration.link_quality_metric_func = lambda *args: _DeferredQuality(original, *args)
    try:
        yield
    finally:
        registration.link_quality_metric_func = original


def resolve_deferred_quality(pairwise_reg_func):
    """`pairwise_reg_func` with a deferred quality in its result computed. Keeps its signature."""
    @functools.wraps(pairwise_reg_func)
    def resolved(*args, **kwargs):
        result = pairwise_reg_func(*args, **kwargs)
        if isinstance(result.get('quality'), _DeferredQuality):
            result['quality'] = result['quality'].value()
        return result
    return resolved


def elastix_registration_or_identity(fixed_data, moving_data, *, fixed_origin, moving_origin, fixed_spacing,
                                     moving_spacing, initial_affine, transform_types=None, **kwargs):
    """multiview_stitcher's Elastix registration, or where Elastix fails on a pair (e.g. an overlap too thin to
    sample) the identity with quality 0, as the feature methods give for a pair with too few points."""
    try:
        return registration.registration_ITKElastix(
            fixed_data, moving_data, fixed_origin=fixed_origin, moving_origin=moving_origin,
            fixed_spacing=fixed_spacing, moving_spacing=moving_spacing, initial_affine=initial_affine,
            transform_types=transform_types, **kwargs)
    except RuntimeError as error:
        logging.warning(f'Elastix: unable to register a pair: {str(error).strip().splitlines()[-1]}')
        return {'affine_matrix': param_utils.identity_transform(len(fixed_spacing)), 'quality': 0}


def register_pair_in_worker(msims, graph, pairwise_reg_func, kwargs):
    """compute_pairwise_registrations for one pair in a worker process, with its wall and CPU time."""
    start, cpu_start = time.perf_counter(), time.thread_time()
    with deferred_link_quality(pairwise_reg_func), \
            dask.config.set({'scheduler': 'synchronous', 'optimization.fuse.active': False}):
        registered = compute_pairwise_registrations(
            msims, graph, pairwise_reg_func=resolve_deferred_quality(pairwise_reg_func), **kwargs)
    return registered, time.perf_counter() - start, time.thread_time() - cpu_start


def estimated_pair_cost(graph, edge):
    """A pair's overlap in pixels at the coarser of its sources' pixel sizes: a proxy for its registration time."""
    spacings = [graph.nodes[node]['stack_props']['spacing'] for node in edge]
    pixel_size = np.prod([max(spacing[dim] for spacing in spacings) for dim in spacings[0]])
    return graph.edges[edge].get('overlap', 0) / pixel_size


def resolve_registration_channel(msim, channel, chosen=None):
    """The 'c' label to register on in this msim, from `channel` as an index or a label. A label
    msim lacks - files that name their channel differently, even within one project - falls back
    to its only channel if it has one, with a warning; with several it is an error naming them.
    `chosen`, a dict shared across a project's msims, keeps one answer (and warning) per set of labels."""
    labels = tuple(str(label) for label in get_msim_image0(msim).coords['c'].values)
    if chosen is not None and labels in chosen:
        return chosen[labels]
    if isinstance(channel, (int, np.integer)):
        label = labels[channel]
    elif channel in labels:
        label = channel
    elif len(labels) == 1:
        logging.warning(f'Registration channel {channel!r} not found - using the only channel, {labels[0]!r}')
        label = labels[0]
    else:
        raise ValueError(f'Registration channel {channel!r} not found; the channels are '
                         + ', '.join(repr(label) for label in labels))
    if chosen is not None:
        chosen[labels] = label
    return label


def check_operation_layout(params):
    """Raise for an operation in the CLI's former layout, which the project template's sections replaced."""
    old_keys = sorted({'operation', 'input', 'output', 'preprocessing', 'source_metadata'} & set(params))
    if old_keys:
        raise ValueError(f'Operation settings in the former layout ({", ".join(old_keys)}): give them in the '
                         f'sections input_output, pre_processing, registration and fusion, as a project file has them')


class MVSRegistration:
    def __init__(self, operation='register', label='', input_path=None, output_path=None,
                 source_metadata={}, extra_metadata={},
                 global_rotation=None, global_center=None,
                 overwrite=True, clear=False, ui='', verbose=False, debug=False):
        # set here as well as in init(): an instance may never have init() called on it
        self.verbose = verbose
        self.logging_dask = self.verbose
        self.logging_time = self.verbose
        self.reset()

        if input_path is not None:
            self.init(operation=operation, label=label, input_path=input_path, output_path=output_path,
                      source_metadata=source_metadata, extra_metadata=extra_metadata,
                      global_rotation=global_rotation, global_center=global_center,
                      overwrite=overwrite, clear=clear, ui=ui, verbose=verbose, debug=debug)

    @property
    def msims(self):
        # built lazily from the geometry init_data() resolved, only once pixel data is needed
        return self.ensure_msims()

    @msims.setter
    def msims(self, value):
        self._msims = value

    @staticmethod
    def has_eager_pre_processing(params):
        """Whether pre-processing will compute over the image data, rather than only rebuild the graph.

        Scaling and Gaussian filtering are lazy; flat-field, normalisation and foreground filtering read every source.
        """
        return bool(params.get('flatfield_quantiles') or params.get('filter_foreground')
                    or normalisation_enabled(params.get('normalisation')))

    def ensure_msims(self, progress_factory=None, target_scale=None, weight=1, ends_only=False):
        """The lazy build the msims property triggers, callable ahead of time to report its progress.

        `target_scale` builds each source's pyramid from the level that scale needs, cached per scale, never as
        self.msims, which stays the full-resolution pyramid. `ends_only` builds just each source's finest level at
        that scale and its coarsest (see source_levels): all pre-processing's consumers read.
        """
        # a factor as text ('2') or a pixel size ('10um'): one cache entry however it was written
        target_scale = parse_scale(target_scale, default=None)
        key = self._scaled_msims_key(target_scale, ends_only)
        if (target_scale or ends_only) and key not in self._scaled_msims:
            from_levels = [get_level_from_scale(source, target_scale)[0] if target_scale else 0
                           for source in self.sources]
            # nothing to skip: the shared full-resolution build serves, unless only the ends are wanted,
            # which the shared build must not be cut down to
            if any(from_levels) or ends_only:
                self._scaled_msims[key] = self._build_msims(progress_factory=progress_factory,
                                                            from_levels=from_levels, store=False,
                                                            weight=weight, ends_only=ends_only)
        if key in self._scaled_msims:
            return self._scaled_msims[key]
        if self._msims is None:
            self._build_msims(progress_factory=progress_factory, weight=weight)
        return self._msims

    @staticmethod
    def _scaled_msims_key(target_scale, ends_only=False):
        return f'{target_scale}:ends' if ends_only else str(target_scale)

    def msims_build_pending(self, target_scale=None, ends_only=False):
        """Whether ensure_msims() would actually build, and so report a progress phase of its own."""
        target_scale = parse_scale(target_scale, default=None)
        key = self._scaled_msims_key(target_scale, ends_only)
        if key in self._scaled_msims:
            return False
        if ends_only:
            return True
        if target_scale:
            from_levels = [get_level_from_scale(source, target_scale)[0] for source in self.sources]
            if any(from_levels):
                return True
        return self._msims is None

    def _build_source_msim(self, index, from_level=0, ends_only=False):
        return build_source_msim(self.sources[index], self._msim_output_order, self.positions[index],
                                 self._msim_transforms[index], self.source_transform_key,
                                 z_scale=self._msim_z_scale, from_level=from_level, ends_only=ends_only)

    def _build_msims(self, progress_factory=None, from_levels=None, store=True, weight=1, ends_only=False):
        # where each source's pixels are first opened: threads overlap network reads, each source's caches its own
        def build_msim(index):
            return self._build_source_msim(index, from_levels[index] if from_levels else 0, ends_only)

        msims = pooled_map(build_msim, len(self.sources), default_msim_build_workers, progress_factory,
                           desc='Building sources', weight=weight,
                           timing_label='Build msims' if self.logging_time else None)
        if store:
            self._apply_mappings(msims)
            self._msims = msims
        return msims

    def scaled_source_msims(self, indices, scale):
        """Sources `indices` as pre-processing starts from them, without building the others: from the level `scale`
        needs, binned down to it where the pyramid stops short."""
        built = self._scaled_msims.get(self._scaled_msims_key(scale, ends_only=True))
        if built is not None:
            msims = [built[index] for index in indices]
        else:
            def build_msim(index):
                from_level = get_level_from_scale(self.sources[index], scale)[0] if scale else 0
                return self._build_source_msim(index, from_level, ends_only=True)
            # kept per scale: flat-field models and normalisation stats each read a group's sources
            source_msims = self._scaled_source_msims.setdefault(scale, {})
            missing = [index for index in indices if index not in source_msims]
            source_msims.update(zip(missing, thread_map(build_msim, missing, default_msim_build_workers)))
            msims = [source_msims[index] for index in indices]
        if scale and scale != 1:
            msims = select_msim_subpyramid_at_scale(msims, [self.sources[index] for index in indices], scale,
                                                    bin_shortfall=True)
        return msims

    def _apply_mappings(self, msims):
        """The registration's corrections (self.mappings) chained onto each msim's source transform, as
        register_global writes them: for msims built after a saved registration was loaded."""
        for index, mapping in (self.mappings or {}).items():
            msi_utils.set_affine_transform(msims[index], mapping, transform_key=self.reg_transform_key,
                                           base_transform_key=self.source_transform_key)

    def source_geometry(self, transform_key, promote_z=False, z_scale=None, progress=None):
        """Each source's translation and level-0 stack properties under transform_key, from its metadata and
        self.mappings alone, building no msim: the registered transform is the mapping chained onto the source
        transform, as _apply_mappings writes it, identity for a source without one. `progress` is called a source."""
        translations, stack_props = [], []
        for index, (source, position, transform) in enumerate(zip(self.sources, self.positions,
                                                                   self._msim_transforms)):
            translation = dict(position)
            if promote_z and 'z' not in translation:
                translation['z'] = index * (z_scale or 1)
            props = build_source_stack_props(source, self._msim_output_order, translation, transform,
                                             self.source_transform_key, z_scale=self._msim_z_scale,
                                             promote_z=promote_z)
            if transform_key == self.reg_transform_key:
                matrix = np.asarray(props['transform'], dtype=float)
                mapping = (self.mappings or {}).get(index)
                if mapping is None:
                    matrix = np.eye(len(matrix))
                else:
                    mapping_matrix = widened_affine_matrix(mapping) if len(matrix) == 4 else np.asarray(
                        mapping.sel(t=0) if 't' in mapping.dims else mapping, dtype=float)
                    matrix = mapping_matrix @ matrix
                props['transform'] = props['transform'].copy(data=matrix)
            translations.append(translation)
            stack_props.append(props)
            if progress is not None:
                progress()
        return translations, stack_props

    def reset(self):
        self.state = RegState.UNINIT
        self.source_transform_key = 'source_metadata'
        self.reg_transform_key = 'registered'
        self.transition_transform_key = 'transition'
        self.source_metadata = {}
        self.extra_metadata = {}
        self.msims = []
        self.register_msims = None
        self.pair_msims = None
        self.pairs = []
        self.pairs_graph = None
        self._scaled_msims = {}
        self._scaled_source_msims = {}
        self.sources = []
        # set on each source read, cleared once shown
        self.missing_metadata_summary = ''
        self.metrics = {}
        self.group_pairs = {}
        self.register_indices = None
        # per source index, for fusion and the view as well as registration
        self._flatfield_key = None
        self._flatfield_models = {}
        self._flatfield_lock = threading.Lock()
        # what register_msims were made with, and global normalisation's stats per (settings, source group)
        self.preprocessed_settings = None
        self._global_normalise_stats = {}
        # per source index: the registration's correction on top of the source transform; absent means identity
        self.mappings = None
        self.input_output_params = {}

    def is_initialised(self):
        return self.state.value >= RegState.INIT.value

    def is_pairs_registered(self):
        return self.state.value >= RegState.PAIRS_REG.value

    def is_global_registered(self):
        return self.state.value >= RegState.GLOBAL_REG.value

    def is_fused(self):
        return self.state.value >= RegState.FUSED.value

    def init_params(self, params_general, params, label='', input_path=None, global_rotation=None, global_center=None):
        """A CLI operation: its settings in the project template's sections (input_output, pre_processing,
        registration, fusion), as a project file has them, with the run's general settings."""
        check_operation_layout(params)
        self.params_general = params_general
        self.params = params
        self.input_output_params = params.get('input_output', {})
        self.preprocess_params = params.get('pre_processing', {})
        self.register_params = params.get('registration', {})
        self.fusion_params = params.get('fusion', {})
        input_output = self.input_output_params
        if input_path is None:
            input_path = input_output.get('input_path')

        return self.init(operation=self.register_params.get('operation', ''), label=label, input_path=input_path,
                         pairing=self.register_params.get('pairing', ''),
                         input_labels=input_output.get('labels'),
                         output_path=input_output.get('output_path'),
                         source_metadata=source_metadata_from_params(input_output),
                         extra_metadata=extra_metadata_from_params(input_output),
                         global_rotation=global_rotation, global_center=global_center,
                         overwrite=input_output.get('overwrite', False), clear=input_output.get('clear', False),
                         ui=params_general.get('ui', ''),
                         verbose=params_general.get('verbose', False), debug=params_general.get('debug', False))

    @property
    def is_stack(self):
        """Stack registration: pair consecutive views, and treat the set as a z-stack rather than a tiled plane.
        Selected by `pairing: stack` only, as the plugin never sets an operation."""
        return 'stack' in self.pairing

    def init(self, operation='', label='', input_path=None, input_labels=None, output_path=None,
             source_metadata={}, extra_metadata={}, global_rotation=None, global_center=None,
             overwrite=True, clear=False, ui='', verbose=False, debug=False, pairing=''):
        self.overwrite = overwrite
        self.clear = clear
        self.ui = ui
        self.verbose = verbose
        self.debug = debug
        self.logging_dask = self.verbose
        self.logging_time = self.verbose
        self.mpl_ui = ('mpl' in self.ui or 'plot' in self.ui)
        self.operation = operation
        self.pairing = (pairing or '').lower()
        self.fileset_label = label
        self.global_rotation = global_rotation
        self.global_center = global_center
        self.source_transform_key = 'source_metadata'
        self.reg_transform_key = 'registered'
        self.transition_transform_key = 'transition'
        self.msims = []
        self.sources = []
        self.state = RegState.INIT
        self.start_time = datetime.now()

        self.input_path = input_path
        # a folder stands for its files, but a zarr folder is one image
        patterns = [os.path.join(path, '*') if os.path.isdir(path) and '.zar' not in path.lower() else path
                    for path in ensure_list(input_path)]
        self.input_dir = pattern_base_dir(patterns[0])
        self.filenames = [Path(path).as_posix() for path in dir_regex(patterns)]
        if not self.filenames:
            return False
        item_files = [{Path(path).as_posix() for path in dir_regex(pattern)} for pattern in patterns]
        # the input path item each file came from: a channel each, in a channel registration
        self.input_groups = [next(item for item, files in enumerate(item_files) if filename in files)
                             for filename in self.filenames]

        if input_labels:
            self.file_labels = input_labels
        else:
            self.file_labels = get_unique_file_labels(self.filenames)

        self.source_metadata = source_metadata
        self.extra_metadata = extra_metadata

        try:
            output_path = output_path.format_map(split_numeric_dict(self.filenames[0]))
            self.output = os.path.join(self.input_dir, output_path)    # preserve trailing slash: do not use os.path.normpath()
            output_dir = os.path.dirname(self.output)
            if self.clear:
                shutil.rmtree(output_dir, ignore_errors=True)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Error initializing output directory: {e}")
            return False

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
        z_scale = get_metadata_z_scale(extra_metadata)
        channels = extra_metadata.get('channels', []) if isinstance(extra_metadata, dict) else []
        normalise_orientation = 'norm' in source_metadata
        fusion_params = self.fusion_params
        overlap_threshold = self.register_params.get('overlap_threshold', 0.5)
        save_images = fusion_params.get('save_images', True)

        output_format = fusion_params.get('format', zarr_extension)
        output_block_size = parse_block_size(fusion_params.get('block_size'))
        output_compression = fusion_params.get('compression')
        output_pyramid_downsample = fusion_params.get('pyramid_downsample', 2)
        output_npyramid_add = fusion_params.get('npyramid_add', 0)
        output_ome_version = fusion_params.get('ome_version') or default_ome_zarr_version

        if len(filenames) == 0:
            logging.warning('Skipping (no images)')
            return False

        output_filename = operation_to_past_participle(operation)

        self.check_progress(output_filename, output_format)

        if self.is_fused() and not self.overwrite:
            logging.warning(f'Skipping existing output {output_filename}')
            return False

        with Timer('init sims', verbose=self.logging_time):
            self.init_data()
            msims = self.msims

        is_3d = (self.sources[0].get_size().get('z', 0) > 1)
        is_stack = self.is_stack
        is_simple_stack = is_stack and not is_3d
        is_transition = ('transition' in operation)
        is_channel_overlay = (len(channels) > 1)
        if not z_scale:
            z_scale = self.scales[0].get('z', 1)

        with Timer('pre-process', verbose=self.logging_time):
            self.preprocess(msims, **self.preprocess_params)
            register_msims, register_indices = self.register_msims, self.register_indices

        self.init_progress(output_filename, output_format)

        data = []
        mappings_header = ['id', 'filename', 'x_pixels', 'y_pixels', 'z_pixels', 'x', 'y', 'z', 'rotation']
        for label, filename, position, rotation, scale in zip(file_labels, self.filenames, self.positions, self.rotations, self.scales):
            position_pixels = {dim: position[dim] / float(scale.get(dim, 1)) for dim in position.keys()}
            row = [label] + [filename] + dict_to_xyz(position_pixels, add_zeros=True) + dict_to_xyz(position, add_zeros=True) + [rotation]
            data.append(row)
        export_csv(output + prereg_mappings_name, data, header=mappings_header)

        if len(filenames) == 1 and save_images and not 'register' in operation:
            logging.warning('Skipping operation (single image)')
            sim = msi_utils.get_sim_from_msim(msims[0], scale='scale0')
            self.save(output_filename, sim, translations0=self.positions,
                      format=output_format,
                      block_size=output_block_size,
                      pyramid_downsample=output_pyramid_downsample,
                      npyramid_add=output_npyramid_add,
                      ome_version=output_ome_version)
            return False

        _, has_overlaps = self.validate_overlap(msims, file_labels, is_stack=is_simple_stack,
                                                expect_large_overlap=is_simple_stack or is_channel_overlay)
        overall_overlap = np.mean(has_overlaps)
        if overall_overlap < overlap_threshold:
            raise ValueError(f'Not enough overlap: {overall_overlap * 100:.1f}%')

        if not self.is_global_registered() or self.overwrite:
            if 'register' in operation:
                with Timer('register', verbose=self.logging_time):
                    results = self.register(register_msims, register_indices, self.register_params)
                reg_result = results['reg_result']
                mappings = results['mappings']
                metrics = results['metrics']

            if 'register' in operation:
                logging.info(metrics['summary'])
                self.save_mappings_csv(mappings, normalise_orientation=normalise_orientation)
                self.save_residuals(results['residual_errors'])
                self.write_ro_crates([], {'general': self.params_general, **self.params})

                for reg_label, reg_item in reg_result.items():
                    if isinstance(reg_item, dict):
                        summary_plot = reg_item.get('summary_plot')
                        if summary_plot is not None:
                            figure, axes = summary_plot
                            summary_plot_filename = output + f'{reg_label}.pdf'
                            figure.savefig(summary_plot_filename)

        self.msims = msims
        registered_positions_filename = output + registered_positions_name
        if self.reg_transform_key in get_msim_transform_keys(msims[0]):
            transform_key = self.reg_transform_key
            with Timer('plot positions', verbose=self.logging_time):
                # plot_positions needs sims; a stack is stored 2D, so promoted for this plot to space its slices in z
                plot_msims = make_msims_3d(msims, z_scale, self.positions) if is_stack else msims
                plot_sims = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in plot_msims]
                vis_utils.plot_positions(plot_sims, transform_key=transform_key,
                                         use_positional_colors=False, view_labels=file_labels, view_labels_size=3,
                                         show_plot=self.mpl_ui, output_filename=registered_positions_filename)
                plt_close()
        else:
            transform_key = self.source_transform_key

        logging.info('Exporting...')

        image_paths = []
        if save_images:
            if fusion_params.get('preview'):
                with Timer('create preview', verbose=self.logging_time):
                    self.create_preview('preview_' + output_filename,
                                        nom_msims=msims,
                                        transform_key=transform_key)

            if ('register' in operation or 'fuse' in operation or 'merge' in operation):
                with Timer('fuse image', verbose=self.logging_time):
                    fusion_method, output_spacing = self.fusion_settings()
                    zarr_output_filename = output_filename if 'zar' in output_format else None
                    # msims as stored (2D for a stack): fuse() promotes what it needs to
                    fuse_msims = self.flatfield_corrected_msims(msims, self.preprocess_params)
                    fused_msim, is_saved = self.fuse(fuse_msims, fusion_method=fusion_method, output_spacing=output_spacing,
                                                     transform_key=transform_key, output_filename=zarr_output_filename,
                                                     block_size=output_block_size, ome_version=output_ome_version)
                    self.state = RegState.FUSED
            else:
                fused_msim = msims
                is_saved = False

            # save_image() and save_video() only accept sims
            fused_sims = extract_sims_from_fused(fused_msim)

            if not is_saved or 'tif' in output_format:
                extra_output_format = output_format
                if is_saved:
                    extra_output_format = extra_output_format.replace('ome.zarr', '').replace('zar', '')
                logging.info('Saving fused image...')
                with Timer('save fused image', verbose=self.logging_time):
                    self.save(output_filename, fused_sims,
                              transform_key=transform_key, translations0=self.positions,
                              format = extra_output_format,
                              block_size = output_block_size,
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

        self.write_ro_crates(image_paths, {'general': self.params_general, **self.params})

        if is_transition:
            self.save_video(output, msims, fused_msim)

        return True

    def write_ro_crates(self, image_filenames, params):
        """A crate in each fused zarr for where it came from, and a workflow run crate over all output with the
        run's parameter values. Written once the registration is saved, and again over the fused images."""
        output_dir = os.path.dirname(self.output)
        end_time = datetime.now()
        try:
            software_version = importlib.metadata.version('muvis-align')
        except importlib.metadata.PackageNotFoundError:
            software_version = None
        software = {'name': 'muvis-align', 'url': MUVIS_ALIGN_URL, 'version': software_version}
        measurements = {name: {'value': value, 'description': f'Registration {name}, over the overlaps of all '
                                                              f'registered pairs'}
                        for name, value in self.summary_metrics().items()}
        for filename in image_filenames:
            if filename.endswith(zarr_extension):
                # the first and last sources stand for the acquisition: reading every source's metadata is too slow
                create_zarr_ro_crate(self.output + filename, source_paths=self.filenames,
                                     acquisition_metadata=[self.sources[0].metadata, self.sources[-1].metadata],
                                     measurements=measurements,
                                     description=f'OME-Zarr image fused by muvis-align from {len(self.filenames)} '
                                                 f'sources',
                                     derivation={'name': 'Fusion to OME-Zarr', 'software': software,
                                                 'end_time': end_time})
        result_paths = [self.output + filename for filename in image_filenames] + sorted(self.saved_progress_paths())
        template = importlib.resources.files('muvis_align').joinpath(NAPARI_PROJECT_TEMPLATE)
        with importlib.resources.as_file(template) as template_path:
            create_ro_crate(output_dir, str(template_path), self.filenames, result_paths, parameters=params,
                            name=f'muvis-align run: {get_filetitle(os.path.normpath(output_dir))}',
                            run_name='muvis-align run', start_time=self.start_time, end_time=end_time,
                            software=software)

    def middle_section_indices(self):
        """The files of the middle section, by section number (a label as S000 or SBEMimage's s00538), else by folder;
        None if neither tells sections apart."""
        # tile and slice indices look alike in a name: only the label tells them apart
        labelled = [find_labelled_numbers(Path(filename).name) for filename in self.filenames]
        candidates = [[numbers.get(label) for numbers in labelled] for label in section_number_labels]
        candidates.append([Path(filename).parent for filename in self.filenames])
        for keys in candidates:
            sections = sorted(set(keys)) if None not in keys else []
            if len(sections) > 1:
                # the step napari's dims start on: int((nsteps - 1) / 2)
                middle = sections[(len(sections) - 1) // 2]
                return [index for index, key in enumerate(keys) if key == middle]
        return None

    def section_registration(self, indices):
        """A registration of only the files at `indices`, labelled as here: to show them before the rest are read."""
        section = MVSRegistration()
        section.init(operation=self.operation, label=self.fileset_label,
                     input_path=[self.filenames[index] for index in indices],
                     input_labels=[self.file_labels[index] for index in indices], output_path=self.output,
                     overwrite=False, pairing=self.pairing, verbose=self.verbose)
        return section

    def init_sources(self, progress_factory=None):
        def build_source(index):
            return create_image_source(self.filenames[index], self.source_metadata, file_label=self.file_labels[index],
                                       index=index, transform_key=self.source_transform_key)

        self.sources = pooled_map(build_source, len(self.filenames), default_source_init_workers, progress_factory,
                                  desc='Initialising sources', timing_label='Init sources' if self.logging_time else None)
        self.missing_metadata_summary = get_missing_metadata_summary(self.sources)
        if self.missing_metadata_summary:
            logging.warning(self.missing_metadata_summary)

    def init_data(self, source_metadata={}, extra_metadata={}, target_scale=None, store=True,
                  progress_factory=None):
        if not source_metadata:
            source_metadata = self.source_metadata
        if not extra_metadata:
            extra_metadata = self.extra_metadata
        source_metadata = import_metadata(source_metadata, input_path=self.input_path)
        extra_metadata = import_metadata(extra_metadata, input_path=self.input_path)
        source_metadata_changed = (source_metadata != self.source_metadata)
        self.source_metadata = source_metadata
        self.extra_metadata = extra_metadata
        z_scale = get_metadata_z_scale(source_metadata) or get_metadata_z_scale(extra_metadata)

        if len(self.filenames) == 0:
            raise ValueError('No input files')

        logging.info('Initialising sources...')
        if not self.sources or source_metadata_changed:
            self.init_sources(progress_factory=progress_factory)
        sources = self.sources
        source0 = sources[0]
        scales = []
        translations = []
        rotations = []
        levels = []
        rescales = []

        is_3d = (source0.get_size().get('z', 0) > 1)
        is_stack = self.is_stack
        output_order = 'zyx' if is_3d else 'yx'

        ndims = len(output_order)
        if source0.get_nchannels() > 1:
            if source0.is_rgb:
                output_order = output_order + 'c'
            else:
                output_order = 'c' + output_order

        last_z_position = None
        delta_zs = []
        # timed apart from init_sources(): these serial per-source passes are not in its pooled timing
        geometry_timer = Timer('init_data: resolve per-source geometry', verbose=self.logging_time)
        geometry_timer.start()
        for filename, source in zip(self.filenames, sources):
            # metadata ImageSource already resolved (fix_metadata); no pixel data is read
            # copies: the z and x/y filled in below are this run's, not the source's
            scale = dict(source.get_pixel_size())
            translation = dict(source.get_position())
            rotation = source.get_rotation()

            level = 0
            rescale = {}
            if target_scale:
                # Only downscaling
                level, rescale, scale = get_level_from_scale(source, target_scale)
            if 'z' in translation:
                z_position = translation['z']
            else:
                z_position = 0
            if last_z_position is not None and z_position != last_z_position:
                delta_zs.append(z_position - last_z_position)
            if self.global_rotation is not None:
                rotation = self.global_rotation

            scales.append(scale)
            translations.append(translation)
            rotations.append(rotation)
            levels.append(level)
            rescales.append(rescale)
            last_z_position = z_position
        geometry_timer.record()
        geometry_timer.get_total_time()

        dimension = getattr(self, 'registration_dimension', None) or self.input_output_params.get('registration_dimension')
        if not is_3d and self.is_channel_overlay(dimension, extra_metadata):
            # a z per channel is a focus offset, not stack planes: all channels on the first one's plane
            channel_zs = {}
            for group, translation in zip(self.channel_groups(), translations):
                channel_zs.setdefault(group, set()).add(translation.get('z'))
            if all(len(zs) == 1 for zs in channel_zs.values()) and 'z' in translations[0]:
                for translation in translations:
                    translation['z'] = translations[0]['z']

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

        # applies only to 3D sources: make_msims_3d() spaces 2D slices but skips a source already 3D
        increase_z_positions = is_stack

        z_position = 0
        final_scales = []
        final_translations = []
        transforms = []
        msims = [] if not store else None
        transform_timer = Timer(f'init_data: build per-source transforms'
                                f'{"" if store else " and msims"}', verbose=self.logging_time)
        transform_timer.start()
        for source, level, rescale, scale, translation, rotation, file_label in zip(
                sources, levels, rescales, scales, translations, rotations, self.file_labels):
            # transform #dimensions need to match
            if 'z' in output_order:
                if len(scale) > 0 and 'z' not in scale:
                    scale['z'] = abs(z_scale)
                if (len(translation) > 0 and 'z' not in translation) or increase_z_positions:
                    translation['z'] = z_position
                if increase_z_positions:
                    z_position += z_scale
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

            translation = complete_translation(translation)
            transforms.append(transform)
            if not store:
                # from the source's cached msim, adding only the run-level deltas no single source knows
                msim = build_source_msim(source, output_order, translation, transform, self.source_transform_key,
                                         z_scale=z_scale)
                msims.append(msim)
            final_scales.append(scale)
            final_translations.append(translation)
        transform_timer.record()
        transform_timer.get_total_time()

        if store:
            self.scales = final_scales
            self.positions = final_translations
            self.rotations = rotations
            self.state = RegState.SIMS_INIT
            # the msim build itself waits until self.msims is read
            self._msim_output_order = output_order
            self._msim_z_scale = z_scale
            self._msim_transforms = transforms
            self._msims = None
            # the geometry was re-resolved, so the per-scale builds are stale too
            self._scaled_msims = {}
            self._scaled_source_msims = {}
            return None

        return msims

    def has_saved_progress(self, output_filename=None, output_format=None):
        """Whether there is a saved registration to resume: the files check_progress() reads, without changing
        the state. Without an output filename only the mappings are checked."""
        pair_mappings_filename = self.output + self.input_output_params.get('pair_mappings', default_pair_mappings_name)
        mappings_filename = self.output + self.input_output_params.get('mappings', default_mappings_name)
        return bool((output_filename and self.output_exists(output_filename, output_format))
                    or os.path.exists(mappings_filename)
                    or os.path.exists(pair_mappings_filename))

    def saved_progress_paths(self, output=None):
        """The saved registration's existing files, in this run's output or another (e.g. before init() has run)."""
        output = os.path.join(output, '') if output else self.output
        paths = [output + self.input_output_params.get('pair_mappings', default_pair_mappings_name),
                 output + self.input_output_params.get('mappings', default_mappings_name),
                 output + self.input_output_params.get('mappings', default_mappings_tabular_name),
                 output + metrics_name,
                 output + residuals_name]
        return [path for path in set(paths) if os.path.exists(path)]

    def discard_saved_progress(self, output=None):
        """Removes the saved registration, which no longer matches the sources."""
        for path in self.saved_progress_paths(output):
            os.remove(path)
        logging.info(f'Saved registration in {output or self.output} discarded')

    def check_progress(self, output_filename, output_format):
        pair_mappings_filename = self.output + self.input_output_params.get('pair_mappings', default_pair_mappings_name)
        mappings_filename = self.output + self.input_output_params.get('mappings', default_mappings_name)
        if self.output_exists(output_filename, output_format):
            self.state = RegState.FUSED
        elif os.path.exists(mappings_filename):
            self.state = RegState.GLOBAL_REG
        elif os.path.exists(pair_mappings_filename):
            self.state = RegState.PAIRS_REG

    def init_progress(self, output_filename, output_format, progress_factory=None):
        pair_mappings_filename = self.output + self.input_output_params.get('pair_mappings', default_pair_mappings_name)
        mappings_filename = self.output + self.input_output_params.get('mappings', default_mappings_name)
        metrics_filename = self.output + metrics_name
        self.check_progress(output_filename, output_format)

        load_pairs = self.is_pairs_registered() and os.path.exists(pair_mappings_filename)
        self.group_pairs = {}
        # builds no msims: the graph comes from source geometry, mappings are applied as msims are built
        file_index = file_list_index_finder(self.filenames)

        if load_pairs:
            logging.info(f'Loading pair mapping from {pair_mappings_filename}')
            pairs = import_json(pair_mappings_filename)
            indexed_pair_transforms = {}
            indexed_qualities = {}
            indexed_bboxes = {}
            for key, value in pairs.items():
                key1, key2 = json.loads(key)
                if value.get('kind') == split_group_pair_kind:
                    # a label such as 'S000' would also match a file under S000/ as a tile pair
                    self.group_pairs[(key1, key2)] = {item: data for item, data in value.items() if item != 'kind'}
                else:
                    index1, index2 = file_index(key1), file_index(key2)
                    if index1 is not None and index2 is not None:
                        indexed_key = index1, index2
                        indexed_pair_transforms[indexed_key] = (
                            xaffine_from_matrix(value['mapping'], with_t=True))
                        indexed_qualities[indexed_key] = np.array(value.get(default_quality_key, 0))
                        if 'bbox' in value:
                            indexed_bboxes[indexed_key] = data_array_from_values(value['bbox'])
            # not per-source-divisible, but long enough to report as two steps
            progress_context = progress_phase(progress_factory, total=2, desc='Loading pair registration')
            with progress_context as pbar:
                self.pair_msims = self._msims
                self.pairs = list(indexed_pair_transforms.keys())
                self.metrics = {
                    'summary': {default_transform_key: {default_quality_key: np.mean(list(indexed_qualities.values()))}},
                    'pairs': {key: {default_transform_key: {default_quality_key: value.item()}}
                              for key, value in indexed_qualities.items()}
                }
                if pbar is not None:
                    pbar.update(1)
                # built directly from the saved pairs, rather than multiview_stitcher rediscovering each overlap;
                # the saved bboxes give the overlap weight
                overlaps = {key: float(np.prod([abs(high - low) for low, high
                                                in zip(*np.array(value).reshape(2, -1))]))
                            for key, value in indexed_bboxes.items()}
                self.pairs_graph = build_pairs_graph(self.source_geometry(self.source_transform_key)[1], self.pairs,
                                                     overlaps=overlaps)
                if pbar is not None:
                    pbar.update(1)
            nx.set_edge_attributes(self.pairs_graph, indexed_pair_transforms, default_transform_key)
            nx.set_edge_attributes(self.pairs_graph, indexed_qualities, default_quality_key)
            nx.set_edge_attributes(self.pairs_graph, indexed_bboxes, 'bbox')

        if self.is_global_registered():
            logging.info(f'Loading global mapping from {mappings_filename}')

            mappings = read_transforms(mappings_filename)
            missing = [filename for filename in self.filenames if find_file_dict_item(mappings, filename) is None]
            if missing:
                # mappings for a different fileset (e.g. a shared output dir) can't be trusted as this one's
                logging.warning(f'{mappings_filename} has no mapping for {len(missing)} of '
                                f'{len(self.filenames)} files (e.g. {missing[0]}) - ignoring it')
                self.state = RegState.INIT
            else:
                progress_context = progress_phase(progress_factory, total=len(self.filenames),
                                                  desc='Loading global registration')
                self.mappings = {}
                with progress_context as pbar:
                    for index, filename in enumerate(self.filenames):
                        # stored as registration produced it - 2D for a stack, whose msims are stored 2D too
                        self.mappings[index] = xaffine_from_matrix(find_file_dict_item(mappings, filename))
                        if pbar is not None:
                            pbar.update(1)
                if self._msims is not None:
                    self._apply_mappings(self._msims)
                self.pair_msims = self._msims
                # a project saved before metrics.csv reopens without metrics
                rows = import_csv(metrics_filename) if os.path.exists(metrics_filename) else []
                indexed_metrics = {}
                for row in rows:
                    # an empty cell: a metric this pair has no value for
                    values = {metric: float(value) for metric, value in row.items()
                              if metric not in pair_table_columns and value not in ('', None)}
                    # a split group pair's row has no filenames, and an empty name would match any file
                    if not row.get('filename1'):
                        labels = (row['id1'], row['id2'])
                        if labels in self.group_pairs:
                            # its quality is the pair mappings file's
                            self.group_pairs[labels]['metrics'] = {metric: value for metric, value in values.items()
                                                                   if metric != default_quality_key}
                    else:
                        index1, index2 = file_index(row['filename1']), file_index(row['filename2'])
                        if index1 is not None and index2 is not None:
                            indexed_metrics[index1, index2] = values
                # each metric's mean under the registered key, as a global registration's own summary
                metric_keys = list(dict.fromkeys(metric for value in indexed_metrics.values() for metric in value))
                summary = {metric: float(np.mean([value[metric] for value in indexed_metrics.values()
                                                  if value.get(metric) is not None]))
                           for metric in metric_keys}
                self.metrics = {
                    'summary': {self.reg_transform_key: summary},
                    'pairs': {key: {self.reg_transform_key: value} for key, value in indexed_metrics.items()}
                }
        if self.group_pairs and self.metrics:
            self.metrics['group_pairs'] = self.group_pair_metrics()

    def validate_overlap(self, items, labels, is_stack=False, expect_large_overlap=False):
        # sims or msims: only position/size metadata is read
        sims = sims_from_sims_or_msims(items)
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

    def preprocess(self, msims, scale=None,
                   flatfield_quantiles=None, normalisation=None, gaussian_sigma=None, filter_foreground=False,
                   progress_factory=None,
                   **kwargs):
        if kwargs:
            # not fatal, but visible: a mistyped option otherwise silently leaves pre-processing at its defaults
            logging.warning('Ignoring unknown pre-processing option(s):'
                            f' {", ".join(sorted(str(key) for key in kwargs))}')

        do_normalisation = normalisation_enabled(normalisation)
        # a factor, as a number or the text of a text field, or a pixel size with its unit ('10um')
        scale = parse_scale(scale, default=None)
        params = {'scale': scale, 'flatfield_quantiles': flatfield_quantiles, 'normalisation': normalisation,
                  'gaussian_sigma': gaussian_sigma, 'filter_foreground': filter_foreground}
        settings = self.preprocess_settings(params)

        def count_progress_steps():
            n_steps = 0
            if scale and scale != 1:
                n_steps += 1
            if filter_foreground:
                # foreground map + final foreground filtering
                n_steps += 2
            if flatfield_quantiles:
                n_steps += 1
            if gaussian_sigma:
                n_steps += 1
            if do_normalisation:
                n_steps += 1
            return max(n_steps, 1)

        def update_progress():
            if pbar is not None:
                pbar.update(1)

        progress_context = progress_phase(progress_factory, total=count_progress_steps(), desc='Pre-processing')

        with progress_context as pbar:
            modified = False
            source_indices = range(len(msims))

            if scale and scale != 1:
                # native levels where the pyramid reaches the scale, binned down to it where it does not
                msims = select_msim_subpyramid_at_scale(msims, self.sources, scale, bin_shortfall=True)
                modified = True
                update_progress()

            if filter_foreground:
                foreground_map = calc_foreground_map(msims)
                modified = True
                update_progress()
            else:
                foreground_map = None

            if flatfield_quantiles:
                logging.info('Flat-field correction...')
                flatfield_params = {'flatfield_quantiles': flatfield_quantiles, 'scale': scale,
                                    'filter_foreground': filter_foreground}
                self.flatfield_models(flatfield_params, msims=msims, foreground_map=foreground_map)
                msims = self.flatfield_corrected_msims(msims, flatfield_params)
                modified = True
                update_progress()

            if gaussian_sigma:
                logging.info('Applying Gaussian filtering...')
                msims = self._gaussian_filtered_msims(msims, gaussian_sigma)
                modified = True
                update_progress()

            if do_normalisation:
                use_global = ('global' in str(normalisation).lower())
                if use_global:
                    logging.info('Normalising (global)...')
                else:
                    logging.info('Normalising (individual)...')
                if use_global:
                    groups = self.source_groups()
                    msims_of = dict(zip(source_indices, msims))
                    group_stats = {group: self._global_normalise_stat(group, params, msims_of)
                                   for group in set(groups)}
                    norm_stats, norm_dtypes = zip(*(group_stats[groups[index]] for index in source_indices))
                else:
                    norm_stats, norm_dtype = calc_normalise_stats(msims, use_global=False)
                    norm_dtypes = [norm_dtype] * len(msims)
                msims = self._normalised_msims(msims, norm_stats, norm_dtypes)
                modified = True
                update_progress()

            if filter_foreground:
                logging.info('Filtering foreground images...')
                # tile_vars = np.array([np.asarray(np.std(sim)).item() for sim in sims])
                # threshold1 = np.mean(tile_vars)
                # threshold2 = np.median(tile_vars)
                # threshold3, _ = cv.threshold(np.array(tile_vars).astype(np.uint16), 0, 1, cv.THRESH_OTSU)
                # threshold = min(threshold1, threshold2, threshold3)
                # foregrounds = (tile_vars >= threshold)
                new_msims = [msim for msim, is_foreground in zip(msims, foreground_map) if is_foreground]
                logging.info(f'Foreground images: {len(new_msims)} / {len(msims)}')
                indices = np.where(foreground_map)[0]
                msims = new_msims
                modified = True
                update_progress()
            else:
                indices = range(len(msims))

        self.register_msims = msims
        self.register_indices = indices
        self.preprocessed_settings = settings
        return msims, indices, modified

    @staticmethod
    def preprocess_settings(params):
        """Pre-processing params, parsed: equal for settings that give the same pixels, however they were written."""
        quantiles, scale, filter_foreground = MVSRegistration.flatfield_settings(params)
        normalisation = params.get('normalisation')
        if normalisation_enabled(normalisation):
            normalisation = 'global' if 'global' in str(normalisation).lower() else 'individual'
        else:
            normalisation = None
        gaussian_sigma = params.get('gaussian_sigma')
        return (quantiles, scale if scale != 1 else None, normalisation,
                float(gaussian_sigma) if gaussian_sigma else None, filter_foreground)

    def register_msims_current(self, params):
        """Whether register_msims were made with these pre-processing settings."""
        return self.register_msims is not None and self.preprocessed_settings == self.preprocess_settings(params)

    def needs_full_preprocess(self, params):
        """Whether preprocess_sources() cannot give these settings' result: it needs every source."""
        return self.preprocess_settings(params)[4]

    def _global_normalise_stat(self, group, params, msims_of=None):
        """Group `group`'s global normalisation (mean, stddev) and dtype at these settings, over its sources as
        pre-processed up to normalisation: from `msims_of` ({source index: msim}) or pre-processed for it."""
        key = (self.preprocess_settings(params), group)
        if key not in self._global_normalise_stats:
            members = [index for index, member_group in enumerate(self.source_groups()) if member_group == group]
            if msims_of is not None:
                group_msims = [msims_of[index] for index in members]
            else:
                group_msims = self.preprocess_sources(members, {**params, 'normalisation': 'none'})
            stats, dtype = calc_normalise_stats(group_msims, use_global=True)
            self._global_normalise_stats[key] = (stats[0], dtype)
        return self._global_normalise_stats[key]

    def preprocess_sources(self, indices, params):
        """Sources `indices` pre-processed as preprocess() makes them, reading no other source's pixels; only where
        needs_full_preprocess() is False."""
        settings = self.preprocess_settings(params)
        quantiles, scale, normalisation, gaussian_sigma, _ = settings
        msims = self.scaled_source_msims(indices, scale)
        if quantiles:
            models = [self.flatfield_model(index, params) for index in indices]
            msims = [apply_msim_flatfield_model(msim, model) if model is not None else msim
                     for msim, model in zip(msims, models)]
        if gaussian_sigma:
            msims = self._gaussian_filtered_msims(msims, gaussian_sigma)
        if normalisation == 'global':
            groups = self.source_groups()
            norm_stats, norm_dtypes = zip(*(self._global_normalise_stat(groups[index], params) for index in indices))
            msims = self._normalised_msims(msims, norm_stats, norm_dtypes)
        elif normalisation:
            norm_stats, norm_dtype = calc_normalise_stats(msims, use_global=False)
            msims = self._normalised_msims(msims, norm_stats, [norm_dtype] * len(msims))
        return msims

    def _gaussian_filtered_msims(self, msims, gaussian_sigma):
        # sigma is in pixels at the finest native pixel size: every image and level gets the same physical blur
        finest = {dim: min(scale[dim] for scale in self.scales if dim in scale)
                  for dim in 'xy' if any(dim in scale for scale in self.scales)}

        def level_func(level_sim, scale_key):
            level_scale = si_utils.get_spacing_from_sim(level_sim)
            ratios = [finest[dim] / level_scale[dim] for dim in finest if dim in level_scale]
            return gaussian_filter_sim(level_sim, self.source_transform_key,
                                       gaussian_sigma * (np.mean(ratios) if ratios else 1))
        return [map_msim_levels(msim, level_func) for msim in msims]

    def _normalised_msims(self, msims, norm_stats, norm_dtypes):
        normalised = []
        for msim, (norm_min, norm_range), norm_dtype in zip(msims, norm_stats, norm_dtypes):
            # the scale0 min/range at every level, so the whole pyramid is normalised alike
            def level_func(level_sim, scale_key, norm_min=norm_min, norm_range=norm_range, norm_dtype=norm_dtype):
                return normalise_sim(level_sim, self.source_transform_key, norm_min, norm_range, norm_dtype)
            normalised.append(map_msim_levels(msim, level_func))
        return normalised

    @staticmethod
    def flatfield_settings(params):
        """(quantiles, scale, filter_foreground) of pre-processing params, parsed: what a flat-field model depends on."""
        quantiles = params.get('flatfield_quantiles')
        if isinstance(quantiles, str):
            quantiles = [quantile for quantile in quantiles.split(',') if quantile.strip()]
        if not isinstance(quantiles, (list, tuple)):
            quantiles = [quantiles] if quantiles else []
        return (tuple(float(quantile) for quantile in quantiles), parse_scale(params.get('scale'), default=None),
                bool(params.get('filter_foreground')))

    def source_groups(self):
        """Each source's pre-processing group: the input path item it came from (one item: a single group)."""
        input_groups = getattr(self, 'input_groups', None)
        return list(input_groups) if input_groups is not None else [0] * len(self.sources)

    def flatfield_groups(self):
        """Source indices per flat-field model: a section's sources of one group that cover the same field (pixel size
        x image size), as images of a different field share no shading."""
        section_of = {index: section for section, indices in enumerate(group_sims_by_z(self.positions, self.positions))
                      for index in indices}
        groups = {}
        for index, (group, source, scale) in enumerate(zip(self.source_groups(), self.sources, self.scales)):
            size = source.get_size()
            field = tuple(float(f'{size[dim] * scale[dim]:.4g}') for dim in 'yx' if dim in size and dim in scale)
            groups.setdefault((section_of.get(index), group, field), []).append(index)
        return list(groups.values())

    def _flatfield_group_of(self, settings):
        """{source index: flat-field group} the models are kept by, emptied when the settings or sources change."""
        key = (settings, id(self.positions))
        if self._flatfield_key != key:
            self._flatfield_key, self._flatfield_models = key, {}
            self._flatfield_groups = self.flatfield_groups()
            self._flatfield_group_index = {index: group for group, indices in enumerate(self._flatfield_groups)
                                           for index in indices}
        return self._flatfield_group_index

    def _calc_flatfield_groups(self, groups, settings, msims=None, foreground_map=None):
        """Models of flat-field `groups`, from `msims` (every source at the pre-processing scale) or their own tiles
        built at that scale as pre-processing builds them; None for a group of too few images."""
        quantiles, scale, filter_foreground = settings
        indices = [index for group in groups for index in self._flatfield_groups[group]]
        if msims is None:
            msims = dict(zip(indices, self.scaled_source_msims(indices, scale)))
        if filter_foreground and foreground_map is None:
            # the tiles kept out of the quantiles are chosen over every source: groups are then computed all at once
            foreground_map = calc_foreground_map([msims[index] for index in range(len(self.sources))])
        for group in groups:
            group_indices = self._flatfield_groups[group]
            if len(group_indices) < flatfield_min_images:
                logging.warning(f'No flat-field correction for {len(group_indices)} image(s) of one field'
                                f' (at least {flatfield_min_images} needed):'
                                f' {", ".join(self.file_labels[index] for index in group_indices[:5])}')
                self._flatfield_models[group] = None
            else:
                self._flatfield_models[group] = calc_msims_flatfield_model(
                    [msims[index] for index in group_indices], quantiles,
                    foreground_map=[foreground_map[index] for index in group_indices] if foreground_map is not None
                    else None)

    def flatfield_model(self, index, params):
        """Source `index`'s flat-field model, its group's computed from that group's tiles alone when first asked:
        None without quantiles or for too few images, and with foreground filtering until flatfield_models() has
        computed every group."""
        settings = self.flatfield_settings(params)
        if not settings[0]:
            return None
        with self._flatfield_lock:
            group = self._flatfield_group_of(settings)[index]
            if group not in self._flatfield_models and not settings[2]:
                self._calc_flatfield_groups([group], settings)
            return self._flatfield_models.get(group)

    def flatfield_models(self, params, msims=None, foreground_map=None):
        """Every source's flat-field model (None without quantiles), the sections not computed yet computed together:
        from `msims` (every source at the pre-processing scale) if given, else from their tiles."""
        settings = self.flatfield_settings(params)
        if not settings[0]:
            return None
        with self._flatfield_lock:
            group_of = self._flatfield_group_of(settings)
            missing = [group for group in range(len(self._flatfield_groups))
                       if group not in self._flatfield_models]
            if missing and settings[2]:
                missing = list(range(len(self._flatfield_groups)))
            if missing:
                logging.info(f'Flat-field model of {len(missing)} group(s)...')
                if msims is not None:
                    msims = dict(enumerate(msims))
                self._calc_flatfield_groups(missing, settings, msims, foreground_map)
            return [self._flatfield_models[group_of[index]] for index in range(len(self.sources))]

    def flatfield_corrected_msims(self, msims, params):
        """msims (every source, in source order) flat-field corrected by flatfield_models(); unchanged without one."""
        models = self.flatfield_models(params)
        if models is None:
            return msims
        return [apply_msim_flatfield_model(msim, model) if model is not None else msim
                for msim, model in zip(msims, models)]

    def create_registration_method(self, sim0, params={}, method=''):
        registration_method = None
        pairwise_reg_func_kwargs = None

        if not method:
            method = params.get('method',
                                params.get('name', ''))
        method = method.lower()
        method_kind = registration_method_kind(method)

        if method_kind == 'cpd':
            try:
                from muvis_align.registration_methods.RegistrationMethodCPD import RegistrationMethodCPD
            except ImportError:
                raise ImportError('probreg is required for CPD registration.')
            registration_method = RegistrationMethodCPD(sim0, params, self.debug)
            pairwise_reg_func = registration_method.registration
        elif method_kind == 'feature':
            if 'cv' in method:
                from muvis_align.registration_methods.RegistrationMethodCvFeatures import RegistrationMethodCvFeatures
                registration_method = RegistrationMethodCvFeatures(sim0, params, self.debug)
            else:
                from muvis_align.registration_methods.RegistrationMethodSkFeatures import RegistrationMethodSkFeatures
                registration_method = RegistrationMethodSkFeatures(sim0, params, self.debug)
            pairwise_reg_func = registration_method.registration
        elif method_kind == 'elastix':
            try:
                from itk import ElastixRegistrationMethod
            except ImportError:
                raise ImportError('ITK-Elastix is required for ITK-Elastix registration.')
            pairwise_reg_func = elastix_registration_or_identity
            # a translation to get close, then the project's transform type (multiview_stitcher's default ends rigid)
            transform_type = params.get('transform_type', 'rigid')
            pairwise_reg_func_kwargs = {'transform_types': ['translation'] if transform_type == 'translation'
                                        else ['translation', transform_type]}
        elif method_kind == 'ants':
            try:
                import ants
            except ImportError:
                raise ImportError('ANTsPy is required for ANTsPy registration.')
            pairwise_reg_func = registration.registration_ANTsPy
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

    def select_pair_overlap(self, msim1, msim2, params=None):
        """The pair's overlap images at the resolution multiview_stitcher would auto-select for it.

        Depends only on the source data and the registration channel, not the method or its parameters, so it can
        be cached across parameter changes. See register_overlap() to register on it.
        """
        params = params or {}
        # the binning and overlap helpers below do not handle a 'c' dim themselves
        if 'c' in get_msim_dims(msim1):
            chosen = {}
            msim1 = msi_utils.multiscale_sel_coords(
                msim1, {'c': resolve_registration_channel(msim1, params.get('channel', 0), chosen)})
            msim2 = msi_utils.multiscale_sel_coords(
                msim2, {'c': resolve_registration_channel(msim2, params.get('channel', 0), chosen)})

        sim1_0 = msi_utils.get_sim_from_msim(msim1, scale='scale0')
        sim2_0 = msi_utils.get_sim_from_msim(msim2, scale='scale0')
        registration_binning = registration.get_optimal_registration_binning(sim1_0, sim2_0)
        scale_key, remaining_binning = msi_utils.get_res_level_from_binning_factors(msim1, registration_binning)
        sim1 = msi_utils.get_sim_from_msim(msim1, scale=scale_key)
        sim2 = msi_utils.get_sim_from_msim(msim2, scale=scale_key)
        if max(remaining_binning.values()) > 1:
            binned = []
            for sim in (sim1, sim2):
                sim_b = sim.coarsen(remaining_binning, boundary='trim').mean().astype(sim.dtype)
                sim_b.attrs.update(copy.deepcopy(sim.attrs))
                binned.append(sim_b)
            sim1, sim2 = binned

        return get_overlap_images(sim1, sim2, self.source_transform_key)

    def register_overlap(self, overlap1, overlap2, sims_pixel_space, params=None):
        """Register an overlap crop from select_pair_overlap().

        Returns the physical-space transform, quality, and pairwise_reg_func's raw result (e.g. feature matches).
        """
        params = params or {}
        _, pairwise_reg_func, pairwise_reg_func_kwargs = self.create_registration_method(overlap1, params=params)
        pairwise_reg_func_kwargs = dict(pairwise_reg_func_kwargs or {})

        fixed_data = overlap1.compute() if hasattr(overlap1, 'compute') else overlap1
        moving_data = overlap2.compute() if hasattr(overlap2, 'compute') else overlap2
        result = pairwise_reg_func(fixed_data, moving_data, **pairwise_reg_func_kwargs)

        try:
            transform = affine_from_intrinsic_affine(result['affine_matrix'], sims_pixel_space, self.source_transform_key)
        except NotImplementedError:
            transform = result['affine_matrix']

        return transform, result.get('quality'), result

    def fusion_settings(self):
        """(method, output spacing) from the fusion section, the spacing defaulting to native as in the plugin."""
        return self.fusion_params.get('method', ''), self.fusion_params.get('spacing') or 'native'

    def create_fusion_method(self, fusion_method, sim0):
        if fusion_method is None:
            fusion_method = ''
        if 'compos' in fusion_method:
            fusion_method = None
            fuse_func = None
        elif 'exclus' in fusion_method:
            from muvis_align.fusion_methods.FusionMethodExclusive import FusionMethodExclusive
            fusion_method = FusionMethodExclusive(sim0, self.debug)
            fuse_func = fusion_method.fusion
        elif 'add' in fusion_method:
            from muvis_align.fusion_methods.FusionMethodAdditive import FusionMethodAdditive
            fusion_method = FusionMethodAdditive(sim0, self.debug)
            fuse_func = fusion_method.fusion
        else:
            from muvis_align.fusion_methods.FusionMethodAverage import FusionMethodAverage
            fusion_method = FusionMethodAverage(sim0, self.debug)
            fuse_func = fusion_method.fusion

        self.fusion_method = fusion_method

        return fuse_func

    def register(self, register_msims=None, register_indices=None, params=None):
        pair_results = self.register_pairs(register_msims=register_msims, register_indices=register_indices, params=params)
        self.save_pair_results(pair_results['pair_mappings'], metrics=pair_results['metrics'])
        results = self.register_global(self.pair_msims, register_indices=register_indices, params=params)
        self.save_group_pair_mappings()
        self.save_mappings(results['mappings'])
        self.save_metrics(results['metrics'])
        return results

    def register_pairs(self, register_msims=None, register_indices=None, params=None,
                       progress_factory=None):
        logging.info('Pair registration...')
        if register_indices is None:
            if self.register_indices is not None:
                register_indices = self.register_indices
            else:
                register_indices = range(len(self.msims))

        pairing = params.get('pairing', '').lower()
        n_parallel_pairwise_regs = params.get('n_parallel_pairwise_regs')
        # None or 0 is automatic; a yaml value may be a float
        n_parallel_pairwise_regs = int(n_parallel_pairwise_regs) if n_parallel_pairwise_regs else None
        logging.info(f"Pair registration settings: pairing {pairing or 'default'}, method {params.get('method')},"
                     f" transform {params.get('transform_type')}, channel {params.get('channel')},"
                     f" workers {n_parallel_pairwise_regs or f'automatic ({default_pair_workers})'},"
                     f" registration dimension {getattr(self, 'registration_dimension', None) or 'space'}")

        # the caller's params win over init()'s: the plugin's dropdown can change in between
        self.pairing = pairing
        is_3d = (self.sources[0].get_size().get('z', 0) > 1)
        is_stack = self.is_stack

        reg_channel = params.get('channel', 0)
        if isinstance(reg_channel, int):
            reg_channel_index = reg_channel
            reg_channel = None
        else:
            reg_channel_index = None

        if register_msims is None:
            register_msims = self.register_msims

        if is_stack and not is_3d:
            # register in 2D, pairing consecutive views: max-project every level that has z
            def level_func(level_sim, scale_key):
                return si_utils.max_project_sim(level_sim, dim='z') if 'z' in level_sim.dims else level_sim
            register_msims = [map_msim_levels(msim, level_func) for msim in register_msims]
            pairs = [(index, index + 1) for index in range(len(register_msims) - 1)]
        elif 'ortho' in pairing or SPLIT in pairing:
            # every source, never the filtered register_msims, so it matches self.positions; from metadata,
            # as building self.msims just to read their geometry is expensive
            with Timer('pair registration: source geometry', verbose=self.logging_time):
                shape_sims = [build_source_shape_sim(source, self._msim_output_order, position, transform,
                                                     self.source_transform_key, z_scale=self._msim_z_scale)
                              for source, position, transform in zip(self.sources, self.positions,
                                                                     self._msim_transforms)]
                origins = np.array([get_sim_position_final(sim, position, get_center=True)
                                    for sim, position in zip(shape_sims, self.positions)])
                sizes = [get_sim_physical_size(sim) for sim in shape_sims]
            with Timer('pair registration: get_pairs', verbose=self.logging_time):
                pairs, _ = get_pairs(origins, sizes)
            if SPLIT in pairing:
                # stage 1: each z-plane's (or channel's) tiles only; the groups are registered in register_global
                pairs = within_group_pairs(pairs, self.split_groups())
            # found over every source, but registered on the retained ones, numbered from 0
            retained_index = {int(index): position for position, index in enumerate(register_indices)}
            pairs = [(retained_index[index1], retained_index[index2]) for index1, index2 in pairs
                     if index1 in retained_index and index2 in retained_index]
            logging.info(f'#pairs: {len(pairs)}')
            #for pair in pairs:
            #    print(f'{self.file_labels[pair[0]]} - {self.file_labels[pair[1]]}')
        else:
            pairs = None

        # feature-based/CPD methods read pixel data from one concrete sim
        reg_method, pairwise_reg_func, pairwise_reg_func_kwargs = self.create_registration_method(
            msi_utils.get_sim_from_msim(register_msims[0], scale='scale0'), params=params)
        logging.info(f'Registration method: {reg_method}')
        logging.info('Registering pairs...')
        # a full pyramid per source lets compute_pairwise_registrations select each pair's resolution

        overlap_tolerance = 0

        # ******* start MVS registration functions

        with Timer('pair registration: select channel', verbose=self.logging_time):
            has_channel = ["c" in get_msim_dims(msim) for msim in register_msims]
            if has_channel[0]:
                if reg_channel is None and reg_channel_index is None:
                    raise Exception("Please choose a registration channel.")
                requested = reg_channel_index if reg_channel is None else reg_channel
                # each source by its own labels: one project's files need not all name it the same
                chosen = {}
                msims_reg = [
                    msi_utils.multiscale_sel_coords(
                        msim, {"c": resolve_registration_channel(msim, requested, chosen)})
                    if has_channel[imsim]
                    else msim
                    for imsim, msim in enumerate(register_msims)
                ]
            else:
                msims_reg = register_msims

        # match each level's transform to its image dimensions: a multiscale image can mix 3D/4D transforms
        for msim in msims_reg:
            for scale_node in msim.ds.values():
                if 'source_metadata' in scale_node.ds.data_vars:
                    img_data = scale_node.ds['image']
                    if hasattr(img_data, 'data'):
                        img_data = img_data.data
                    spatial_dims = [d for d in img_data.dims if d not in ('t', 'c')]
                    current_transform = scale_node.ds.data_vars[self.source_transform_key]
                    transform_spatial_dims = [d for d in current_transform.coords['x_in'].values if d != '1']
                    if len(transform_spatial_dims) != len(spatial_dims):
                        relevant_dim_names = spatial_dims + ['1']
                        adapted = current_transform.sel(x_in=relevant_dim_names, x_out=relevant_dim_names)
                        scale_node.ds[self.source_transform_key] = adapted

        try:
            # fusion off: dask's fused key names drop the unique token, so per-pair crop chains can collide
            # and a pair silently registers another pair's crop
            with dask.config.set({'scheduler': 'threads', 'optimization.fuse.active': False}):
                # multiview_stitcher would test every source within the largest one's diameter: with overview
                # images among tiles, nearly every pair
                graph_pairs = (pairs if pairs is not None
                               else find_candidate_overlap_pairs(msims_reg, self.source_transform_key))
                if pairs is None:
                    logging.info(f'#candidate pairs: {len(graph_pairs)}')
                raise_if_cancelled()
                with Timer('pair registration: view adjacency graph', verbose=self.logging_time):
                    g_reg = build_view_adjacency_graph(msims_reg, self.source_transform_key, graph_pairs,
                                                       overlap_tolerance=overlap_tolerance)
                raise_if_cancelled()

                g_reg_computed = g_reg.copy()
                workers = n_parallel_pairwise_regs or default_pair_workers
                log_every = 2 * workers
                pair_times, pair_cpu_times = [], []
                timed_reg_func = timed_calls(resolve_deferred_quality(pairwise_reg_func), pair_times,
                                             pair_cpu_times)

                pair_kwargs = dict(transform_key=self.source_transform_key, overlap_tolerance=overlap_tolerance,
                                   pairwise_reg_func_kwargs=pairwise_reg_func_kwargs, n_parallel_pairwise_regs=1)
                # largest first, so workers are not left idle at the end and the bar's estimate holds
                edges = sorted(g_reg.edges, key=lambda edge: estimated_pair_cost(g_reg, edge), reverse=True)
                pool = None
                # worker processes when a pair can be sent to one (its sources' arrays, the method)
                if (workers > 1 and len(edges) > workers
                        and picklable(([msims_reg[node] for node in edges[0]], pairwise_reg_func))):
                    pool = worker_process_pool(workers, max_tasks_per_child=default_pair_worker_tasks)

                def register_edge(edge):
                    graph = g_reg.edge_subgraph([edge]).copy()
                    if pool is None:
                        return compute_pairwise_registrations(msims_reg, graph, pairwise_reg_func=timed_reg_func,
                                                              **pair_kwargs)
                    registered, wall_time, cpu_time = result_unless_cancelled(pool.submit(
                        register_pair_in_worker, {node: msims_reg[node] for node in graph.nodes}, graph,
                        pairwise_reg_func, pair_kwargs))
                    pair_times.append(wall_time)
                    pair_cpu_times.append(cpu_time)
                    return registered

                phase_start, phase_cpu_start = time.time(), time.process_time()
                # one pair per thread, each its own compute, so a slow pair holds up no batch; a single worker
                # keeps the threads scheduler so one (e.g. 3D) pair still runs its tasks in parallel
                with progress_phase(progress_factory, total=g_reg.number_of_edges(),
                                         desc='Registering pairs') as pbar, \
                        Timer(f'register {g_reg.number_of_edges()} pairs on {workers}'
                              f' {"processes" if pool is not None else "threads"}', verbose=self.logging_time), \
                        (pool if pool is not None else nullcontext()), \
                        deferred_link_quality(pairwise_reg_func), \
                        dask.config.set(scheduler='synchronous' if workers > 1 else 'threads'):
                    log_start = time.time()
                    for count, (edge, pair_computed) in enumerate(
                            rolling_map(register_edge, edges, workers), start=1):
                        g_reg_computed.edges[edge].update(pair_computed.edges[edge])
                        del pair_computed
                        if count % log_every == 0 or count == g_reg.number_of_edges():
                            release_memory(generation=1)
                            if self.logging_time:
                                # cpu per pair rising means harder pairs; wall alone rising means idle workers or I/O
                                batch_wall = time.time() - log_start
                                batch_times, batch_cpu_times = pair_times[-log_every:], pair_cpu_times[-log_every:]
                                pair_timing = ''
                                if batch_times and batch_cpu_times:
                                    pair_timing = (f', a pair {np.mean(batch_times):.2f}s wall'
                                                   f' {np.mean(batch_cpu_times):.2f}s cpu,'
                                                   f' {sum(batch_times) / max(batch_wall, 1e-9):.1f} busy')
                                logging.info(f'Pairs {count}/{g_reg.number_of_edges()}:'
                                             f' last {log_every} in {batch_wall:.1f}s{pair_timing}'
                                             f'{print_memory_usage()}')
                            log_start = time.time()
                        if pbar is not None:
                            pbar.update(1)
                if self.logging_time and pair_times:
                    logging.info(f'Register pairs: {len(pair_times)} pairs'
                                 f' {format_phase_timing(time.time() - phase_start, pair_times, pair_cpu_times,
                                                         workers, time.process_time() - phase_cpu_start,
                                                         processes=pool is not None)}')

                # ******* end MVS registration functions

        except NotEnoughOverlapError:
            g_reg_computed = g_reg

        release_memory('Pair registration')
        mappings = nx.get_edge_attributes(g_reg_computed, default_transform_key)
        mappings_dict = {(register_indices[indices[0]], register_indices[indices[1]]): mapping
                         for indices, mapping in mappings.items()}

        with Timer(f'pair metrics ({g_reg_computed.number_of_edges()} pairs)', verbose=self.logging_time):
            metrics = calc_pair_metrics(msims_reg, g_reg_computed, params.get('metrics', []),
                                        self.source_transform_key, reg_channel=reg_channel_index,
                                        n_parallel_pairs=n_parallel_pairwise_regs,
                                        progress_factory=progress_factory)
            remap_pair_metrics(metrics, register_indices)
        release_memory('Pair metrics')

        self.pairs_graph = g_reg_computed
        self.pair_msims = msims_reg
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

    def split_groups(self, indices=None):
        """Group index per source (of `indices`, renumbered from 0): z-planes, or channels for dimension 'c'."""
        dimension = (getattr(self, 'registration_dimension', None)
                     or getattr(self, 'register_params', {}).get('registration_dimension'))
        groups = split_groups(self.positions, self.sources, dimension, getattr(self, 'input_groups', None))
        if indices is not None:
            groups = [groups[index] for index in indices]
            order = sorted(set(groups))
            groups = [order.index(group) for group in groups]
        return groups

    def channel_groups(self):
        """Each source's output channel in a channel overlay: see split_registration.channel_groups."""
        return channel_groups(self.sources, getattr(self, 'input_groups', None))

    def channel_labels(self, channels=None):
        """A label per channel of channel_groups(): as set in `channels` (extra_metadata's), else its sources' shared
        metadata channel label if no other channel has it, else their common label prefix, else 'channel <index>'."""
        groups = self.channel_groups()
        members = [[index for index, member_group in enumerate(groups) if member_group == group]
                   for group in range(max(groups, default=-1) + 1)]
        metadata_labels = []
        for group_members in members:
            group_labels = {(self.sources[index].get_channels() or [{}])[0].get('label') or ''
                            for index in group_members}
            metadata_labels.append(group_labels.pop() if len(group_labels) == 1 else '')
        labels = []
        for group, group_members in enumerate(members):
            label = metadata_labels[group] if metadata_labels.count(metadata_labels[group]) == 1 else ''
            if channels and group < len(channels):
                label = channels[group].get('label') or label
            labels.append(label or group_label([self.file_labels[index] for index in group_members])
                          or f'channel {group}')
        return labels

    def split_group_labels(self, indices):
        """A label per group of split_groups(indices): its sources' common label prefix (e.g. 'S000'), else its
        z-position or channel."""
        groups = self.split_groups(indices)
        labels = []
        for group in range(max(groups) + 1):
            members = [index for index, member_group in zip(indices, groups) if member_group == group]
            label = group_label([self.file_labels[index] for index in members])
            if not label:
                channel = (self.sources[members[0]].get_channels() or [{}])[0].get('label')
                is_channel = (getattr(self, 'registration_dimension', None) == 'c')
                label = str(channel) if is_channel else f"z {self.positions[members[0]].get('z', 0)}"
            labels.append(label)
        return labels

    def _register_split_groups(self, pair_msims, transforms, register_indices, params, progress_factory=None):
        method, pairwise_reg_func, pairwise_reg_func_kwargs = self.create_registration_method(
            msi_utils.get_sim_from_msim(pair_msims[0], scale='scale0'), params=params)
        transform_type = params.get('transform_type', 'rigid')
        resolution_method = ROBUST_LINEAR if transform_type in ('translation', 'rigid') else 'global_optimization'
        binning = params.get('split_binning')
        binning = max(int(float(binning)), 1) if is_valid_value(binning) else default_split_binning
        with progress_phase(progress_factory, total=1, desc='Registering z-planes / channels'), \
                Timer('split: register groups', verbose=self.logging_time):
            transforms, group_graph = register_groups(
                pair_msims, transforms, self.split_groups(list(register_indices)), self.source_transform_key,
                pairwise_reg_func, pairwise_reg_func_kwargs, resolution_method=resolution_method,
                resolution_kwargs={'transform': transform_type}, binning=binning,
                metric_funcs=create_metric_methods(params['metrics'], pair_msims[0]) if params.get('metrics') else None)
        labels = self.split_group_labels(list(register_indices))
        self.group_pairs = {}
        for group1, group2 in group_graph.edges:
            data = group_graph.edges[group1, group2]
            entry = {'mapping': np.array(data[default_transform_key].sel(t=0)).tolist(),
                     default_quality_key: float(quality_to_scalar(data[default_quality_key]))}
            if 'bbox' in data:
                entry['bbox'] = np.array(data['bbox'].sel(t=0)).tolist()
            if data.get('metrics'):
                entry['metrics'] = data['metrics']
            self.group_pairs[(labels[group1], labels[group2])] = entry
        return transforms

    def register_global(self, pair_msims, register_indices=None, params=None,
                        pairs_graph=None, progress_factory=None):
        logging.info('Global registration...')
        if register_indices is None:
            if self.register_indices is not None:
                register_indices = self.register_indices
            else:
                register_indices = range(len(pair_msims))

        if pairs_graph is not None:
            g_reg_computed = pairs_graph
        else:
            g_reg_computed = self.pairs_graph

        ndims = si_utils.get_ndim_from_sim(get_msim_image0(pair_msims[0]))

        groupwise_resolution_method = params.get('groupwise_resolution_method', 'global_optimization')
        transform_type = params.get('transform_type', 'affine')
        if groupwise_resolution_method in (ROBUST_LINEAR, 'linear_two_pass') and transform_type not in ('translation', 'rigid'):
            logging.warning(f'{groupwise_resolution_method} resolves translation or rigid only,'
                            f' not {transform_type}: using global_optimization')
            groupwise_resolution_method = 'global_optimization'
        split_settings = ''
        if SPLIT in params.get('pairing', '').lower():
            split_settings = f", split binning {params.get('split_binning') or default_split_binning}"
        logging.info(f'Global registration settings: resolution {groupwise_resolution_method},'
                     f' transform {transform_type}{split_settings}')
        groupwise_resolution_kwargs = {}
        if groupwise_resolution_method in ('global_optimization', ROBUST_LINEAR, 'linear_two_pass'):
            # transform_type options include 'translation', 'rigid', 'affine', 'similarity'
            groupwise_resolution_kwargs['transform'] = transform_type

        quality_threshold = params.get('post_registration_quality_threshold')
        quality_threshold = float(quality_threshold) if is_valid_value(quality_threshold) else 0.0

        n_parallel_pairwise_regs = params.get('n_parallel_pairwise_regs')
        # None or 0 is automatic; a yaml value may be a float
        n_parallel_pairwise_regs = int(n_parallel_pairwise_regs) if n_parallel_pairwise_regs else None

        plot_summary = self.mpl_ui

        # ******* start MVS registration functions

        # a pair scored at the threshold or below (a failed method gives the identity with 0) is left out of the solve
        # only, so a tile without other pairs keeps its metadata position rather than an unconstrained one's
        failed_pairs = [edge for edge, quality in nx.get_edge_attributes(g_reg_computed, 'quality').items()
                        if not np.nan_to_num(np.asarray(quality, dtype=float), nan=0.0).max() > quality_threshold]
        resolve_graph = g_reg_computed
        if failed_pairs:
            resolve_graph = g_reg_computed.copy()
            resolve_graph.remove_edges_from(failed_pairs)
            logging.warning(f'Global registration: {len(failed_pairs)} of {g_reg_computed.number_of_edges()} pairs'
                            f' left out, scored {quality_threshold:g} or below')
            unregistered = sorted(node for node in resolve_graph.nodes
                                  if resolve_graph.degree(node) == 0 and g_reg_computed.degree(node) > 0)
            if unregistered:
                logging.warning('Global registration: no registered pair, so kept at their metadata position: '
                                + ', '.join(str(self.file_labels[register_indices[node]]) for node in unregistered))

        # multiview_stitcher's own search is O(nodes x edges); a reference outside a component would send that
        # component back to it, so only a connected graph gets one
        if resolve_graph.number_of_nodes() > 2 and nx.is_connected(resolve_graph):
            groupwise_resolution_kwargs['reference_view'] = find_reference_view(resolve_graph)

        if groupwise_resolution_method == ROBUST_LINEAR:
            resolution_progress = progress_phase(progress_factory, total=default_robust_rounds,
                                                      desc='Global registration', weight=4)
        else:
            # global_optimization has nothing to report into a phase: GlobalOptProgress follows its log instead
            resolution_progress = GlobalOptProgress(progress_factory, desc='Global registration',
                                                    max_passes=g_reg_computed.number_of_edges(), weight=4)
        with resolution_progress as pbar, dask.config.set(scheduler='threads'):
            if groupwise_resolution_method == ROBUST_LINEAR and pbar is not None:
                groupwise_resolution_kwargs['progress'] = lambda: pbar.update(1)
            transforms_dict, groupwise_resolution_info_dict = groupwise_resolution(
                resolve_graph,
                method=groupwise_resolution_method,
                **groupwise_resolution_kwargs,
            )

        transforms = [
            transforms_dict[iview] for iview in sorted(g_reg_computed.nodes())
        ]
        if SPLIT in params.get('pairing', '').lower():
            transforms = self._register_split_groups(pair_msims, transforms, register_indices, params,
                                                     progress_factory=progress_factory)

        # the stages below take long enough on a large project to need progress of their own
        with progress_phase(progress_factory, total=len(pair_msims),
                                 desc='Applying transforms') as pbar, \
                Timer('apply registered transforms', verbose=self.logging_time):
            for imsim, msim in enumerate(pair_msims):
                set_msim_affine(
                    msim,
                    transforms[imsim],
                    transform_key=self.reg_transform_key,
                    base_transform_key=self.source_transform_key,
                )
                if pbar is not None:
                    pbar.update(1)

        if plot_summary:
            with progress_phase(progress_factory, total=1, desc='Registration summary'), \
                    Timer('plot registration summaries', verbose=self.logging_time):
                plot_info = _plot_registration_summaries(
                    pair_msims,
                    self.source_transform_key,
                    self.reg_transform_key,
                    resolve_graph,
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

        # onto self.msims, the persistent full per-source pyramid
        with progress_phase(progress_factory, total=len(pair_msims),
                                 desc='Storing transforms') as pbar, \
                Timer('store transforms on sources', verbose=self.logging_time):
            for reg_msim, index in zip(pair_msims, register_indices):
                reg_transform = msi_utils.get_transform_from_msim(reg_msim, transform_key=self.reg_transform_key)
                set_msim_affine(self.msims[index], reg_transform, transform_key=self.reg_transform_key)
                if pbar is not None:
                    pbar.update(1)

        # set missing transforms - sources that never took part in registration (e.g. filtered out)
        for msim in self.msims:
            if self.reg_transform_key not in get_msim_transform_keys(msim):
                set_msim_affine(
                    msim,
                    param_utils.identity_transform(ndim=ndims, t_coords=[0]),
                    transform_key=self.reg_transform_key)

        mappings = reg_result['params']
        # re-index from subset of sims
        # per timepoint index; a registration resolves its first timepoint
        edge_residuals = (groupwise_resolution_info_dict.get('edge_residuals') or {}).get(0, {})
        residual_error_dict = {(int(register_indices[key[0]]), int(register_indices[key[1]])): float(value)
                               for key, value in edge_residuals.items()}
        self.log_residuals(residual_error_dict)
        registration_qualities_dict = reg_result.get('pairwise_registration', {}).get('metrics', {}).get('qualities', {})
        registration_qualities_dict = {(register_indices[key[0]], register_indices[key[1]]): value
                                       for key, value in registration_qualities_dict.items()}

        # re-index from subset of sims
        mappings_dict = {index: mapping for index, mapping in zip(register_indices, mappings)}

        with Timer('global registration metrics', verbose=self.logging_time):
            metrics = calc_global_metrics(pair_msims, self.reg_transform_key, params.get('metrics', []), reg_result,
                                          n_parallel_pairs=n_parallel_pairwise_regs,
                                          progress_factory=progress_factory)
            remap_pair_metrics(metrics, register_indices)

        self.mappings = mappings_dict
        if self.group_pairs:
            metrics['group_pairs'] = self.group_pair_metrics()
        self.metrics = metrics
        self.state = RegState.GLOBAL_REG
        return {'reg_result': reg_result,
                'mappings': mappings_dict,
                'residual_errors': residual_error_dict,
                'registration_qualities': registration_qualities_dict,
                'metrics': metrics}

    @staticmethod
    def _fusion_batch_options(saving_zarr, max_workers=None):
        """Options to fuse a zarr export's blocks on every worker, as multiview_stitcher otherwise fuses them
        sequentially: each block writes its own chunk, and reserves its memory from the shared pool first."""
        if not saving_zarr:
            return None
        return {'workers': max(1, max_workers or default_fusion_workers), 'memory_pool': default_fusion_memory_pool}

    def is_channel_overlay(self, dimension=None, extra_metadata=None):
        if extra_metadata is None:
            extra_metadata = self.extra_metadata
        channels = extra_metadata.get('channels', []) if isinstance(extra_metadata, dict) else []
        return bool((dimension and dimension == 'c') or len(channels) > 1)

    def fuses_to_zarr(self, fusion_method=None, dimension=None, extra_metadata=None):
        """Whether fuse() given an output filename writes the export itself; otherwise its caller saves the lazy
        result it returns, which is where the work is then done."""
        return not self.is_channel_overlay(dimension, extra_metadata) and 'compos' not in (fusion_method or '')

    def _fusion_z_geometry(self, extra_metadata):
        """(z scale, number of distinct z positions, whether 2D sources become a 3D stack) as fusion takes them: a
        stack's slices have no z position of their own, and make_msims_3d() spaces them by index."""
        z_scale = get_metadata_z_scale(self.source_metadata) or get_metadata_z_scale(extra_metadata)
        if z_scale is None:
            z_scale = extract_z_scale(self.positions, self.scales)
        num_z_positions = len({position.get('z') for position in self.positions if 'z' in position})
        return z_scale, num_z_positions, num_z_positions > 1 or self.is_stack

    @staticmethod
    def _output_spacing_method(output_spacing, exports):
        """(native, the spacing method for calc_output_properties): 'native' only for an export of fused sources."""
        is_native = (output_spacing is None or str(output_spacing).lower() == 'native') and exports
        if is_native:
            return True, 'min'
        if output_spacing is None or str(output_spacing).lower() == 'native':
            return False, 'mean'
        return False, output_spacing

    @staticmethod
    def _export_chunksize(dtype, output_stack_properties, sources, num_z_positions, block_size=None):
        """The block an export fuses and writes: a configured block_size, else the memory budget's, for `sources`'
        (msims' or stack properties') extents."""
        if not block_size:
            return get_export_chunk_sizes(dtype, output_stack_properties, sources, num_z_positions=num_z_positions)
        if not isinstance(block_size, (list, tuple)):
            block_size = [block_size] * 2
        chunksize = xyz_to_dict(block_size)
        if 'z' in output_stack_properties['shape'] and 'z' not in chunksize:
            # block_size says nothing about z, so the budget decides it
            chunksize['z'] = get_chunk_sizes(dtype, list(output_stack_properties['shape']), num_sources=len(sources),
                                             num_z_positions=num_z_positions).get('z', 1)
        return chunksize

    def estimate_fusion_size(self, fusion_method=None, output_spacing=None, transform_key=None, dimension=None,
                             block_size=None, extra_metadata=None):
        """Uncompressed bytes per level of the OME-Zarr fuse() would write, from source geometry alone (no msims):
        {'levels': [{'spacing', 'shape', 'bytes'}], 'bytes', 'native'}, or None for an export fuse() does not write.
        Only the blocks some source reaches are written - exactly these."""
        if extra_metadata is None:
            extra_metadata = self.extra_metadata
        if not self.sources or not self.fuses_to_zarr(fusion_method, dimension, extra_metadata):
            return None
        if transform_key is None:
            transform_key = self.reg_transform_key
        z_scale, num_z_positions, promote_z = self._fusion_z_geometry(extra_metadata)
        is_native, spacing_method = self._output_spacing_method(output_spacing, True)
        dtype = np.dtype(self.sources[0].dtype)
        nonspatial = int(np.prod([size for dim, size in self.sources[0].get_size().items() if dim not in 'zyx']))
        voxel_bytes = nonspatial * dtype.itemsize

        translations, source_props = self.source_geometry(transform_key, promote_z=promote_z, z_scale=z_scale)
        matrices = [np.asarray(props['transform'], dtype=float) for props in source_props]

        stacks = {}

        def stack(index, level):
            if (index, level) not in stacks:
                stacks[index, level] = build_source_stack_props(
                    self.sources[index], self._msim_output_order, translations[index], None, None,
                    z_scale=self._msim_z_scale, level=level, promote_z=promote_z)
            return stacks[index, level]

        views0 = [(stack(index, 0), matrix) for index, matrix in enumerate(matrices)]
        level0 = calc_output_properties_from_views(views0, output_spacing_method=spacing_method, z_scale=z_scale)
        levels = []
        source_spacings = [min(view[0]['spacing'][dim] for dim in 'yx') for view in views0]
        scaled_dims = ['y', 'x'] + (['z'] if views0[0][0]['shape'].get('z', 1) > 1 else [])
        chunksize = self._export_chunksize(dtype, level0, [view[0] for view in views0], num_z_positions, block_size)
        for level_spacing, properties, selected in export_levels(source_spacings, level0, scaled_dims,
                                                                 native=is_native):
            output_properties = {key: properties[key] for key in ('spacing', 'origin', 'shape')}
            views = []
            for index in selected:
                # as fusion picks it (fuse_levels_to_ome_zarr)
                level = coarsest_level_within([stack(index, level)['spacing'] for level in range(len(
                    self.sources[index].shapes))], output_properties['spacing'],
                    dims=list(output_properties['spacing']))
                views.append((stack(index, level), matrices[index]))
            chunk, groups = level_blocks(views, output_properties, chunksize)
            levels.append({'spacing': float(level_spacing), 'shape': output_properties['shape'],
                           'bytes': written_voxels(groups, output_properties, chunk) * voxel_bytes})
        return {'levels': levels, 'bytes': sum(level['bytes'] for level in levels), 'native': is_native}

    def fuse(self, msims, fusion_method=None, output_spacing=None, transform_key=None,
             dimension=None, output_filename=None,
             block_size=None, ome_version=default_ome_zarr_version, extra_metadata=None,
             output_chunksize=None, workers=None, level_bytes=None, source_indices=None):
        """Fuse the sources' msims into one multiscale output msim; returns (fused, whether it was written).
        Sims are wrapped first with util.wrap_sims_as_msims. `source_indices`: each msim's source, when not all are
        given in order, for its channel in a channel overlay.

        output_chunksize, a chunk size per spatial dim, is used as-is; otherwise it is sized against memory.
        Written to a file, every level is fused from the sources (fusion_blocks.export_levels): with output_spacing
        None or 'native' each at its own pixel size from the sources at least that fine, else doubling from the set
        spacing. In memory, 'native' stays 'mean'. `level_bytes` (estimate_fusion_size's, a level) shares the progress
        bar out over the levels.
        """
        logging.info('Fusion...')
        if output_filename is not None:
            output_filename = self.output + output_filename

        # only its dtype is read, which needs no full sim
        sim0 = get_msim_image0(msims[0])

        if extra_metadata is None:
            extra_metadata = self.extra_metadata

        channels = extra_metadata.get('channels', []) if isinstance(extra_metadata, dict) else []
        is_channel_overlay = self.is_channel_overlay(dimension, extra_metadata)

        if transform_key is None:
            transform_key = self.reg_transform_key

        z_scale, num_z_positions, promote_z = self._fusion_z_geometry(extra_metadata)
        # the one place a stack becomes 3D: msims are stored 2D, and each consumer that needs the stacked geometry
        # asks for it, rather than every level of every source being rebuilt up front
        if promote_z:
            with Timer('fusion: make_msims_3d', verbose=self.logging_time):
                msims = make_msims_3d(msims, z_scale=z_scale, positions=self.positions)
        if not is_channel_overlay:
            with Timer('fusion: unify channels', verbose=self.logging_time):
                msims = unify_msim_channels(msims)

        is_native, output_spacing = self._output_spacing_method(
            output_spacing, output_filename is not None and not is_channel_overlay)
        with Timer('fusion: output properties', verbose=self.logging_time):
            output_stack_properties = calc_output_properties(msims, transform_key,
                                                             output_spacing_method=output_spacing, z_scale=z_scale)

        if self.verbose:
            logging.info(f'Output stack: {numpy_to_native(output_stack_properties)}')
        data_size = np.prod(list(output_stack_properties['shape'].values())) * sim0.dtype.itemsize
        # the bounding box, not what is written: see estimate_fusion_size()
        logging.info(f'Fusion output bounding box at full resolution: {print_hbytes(data_size)}')

        # peak memory follows the sources in one output chunk, each transformed to a full float32 chunk;
        # the input's own chunk grid would put every source in one chunk of a coarse output
        default_output_chunksize = get_chunk_sizes(sim0.dtype, list(output_stack_properties['shape']),
                                                   num_sources=len(msims),
                                                   num_z_positions=num_z_positions)
        saving_zarr = False
        if is_channel_overlay:
            # each channel's sources fused together, onto the same grid
            source_groups = self.channel_groups()
            groups = [source_groups[index] for index in (source_indices or range(len(msims)))]
            labels = self.channel_labels(channels)
            labels = [labels[group] for group in sorted(set(groups))]
            fuse_func = self.create_fusion_method(fusion_method, sim0)
            channel_results = []
            for group in sorted(set(groups)):
                group_msims = unify_msim_channels([msim for msim, msim_group in zip(msims, groups)
                                                   if msim_group == group])
                channel_results.append(fusion.fuse(
                    group_msims,
                    # compose (no fusion function) still fuses, as a channel needs one image
                    fusion_func=fuse_func or fusion.weighted_average_fusion,
                    transform_key=transform_key,
                    output_stack_properties=output_stack_properties,
                    output_chunksize=output_chunksize or get_chunk_sizes(
                        sim0.dtype, list(output_stack_properties['shape']), num_sources=len(group_msims),
                        num_z_positions=num_z_positions)))
            fused_image = combine_msims_as_channels(channel_results, labels)
        else:
            if fusion_method:
                logging.info(f'Fusion method: {fusion_method}')
            else:
                logging.info('Fusion method: [default method]')
            fuse_func = self.create_fusion_method(fusion_method, sim0)
            if fuse_func:
                saving_zarr = output_filename is not None
                if output_chunksize is None and saving_zarr:
                    with Timer('fusion: export chunk sizes', verbose=self.logging_time):
                        output_chunksize = self._export_chunksize(sim0.dtype, output_stack_properties, msims,
                                                                  num_z_positions, block_size)
                if output_chunksize is None:
                    output_chunksize = dict(default_output_chunksize)
                # a chunk past the output's extent also fails dask's zarr write-alignment check, with a warning
                output_chunksize = {dim: min(int(size), output_stack_properties['shape'].get(dim, int(size)))
                                    for dim, size in output_chunksize.items()}
                if self.verbose:
                    # logged once settled by the branches above
                    logging.info(f'Fusion output_chunksize: {numpy_to_native(output_chunksize)}'
                                 f' ({len(msims)} sources over'
                                 f' {max(1, num_z_positions)} z position(s))')
                if saving_zarr:
                    if not output_filename.lower().endswith('.zarr'):
                        output_filename += zarr_extension
                    zarr_options = {'ngff_version': ome_version}
                else:
                    zarr_options = None
                with dask.config.set(scheduler='threads'), warnings.catch_warnings():
                    # fused blocks are whole zarr chunks, one writer each: dask flags every edge block, whose chunk
                    # the array ends in. Set here, as a napari worker puts 'always' ahead of any module-level filter
                    warnings.filterwarnings('ignore', message='The input Dask array will be rechunked',
                                            category=PerformanceWarning)
                    if saving_zarr:
                        source_spacings = [min(si_utils.get_spacing_from_sim(get_msim_image0(msim))[dim]
                                               for dim in 'yx') for msim in msims]
                        # a section stack keeps every section at every level; a volume is scaled in z too
                        scaled_dims = ['y', 'x'] + (['z'] if get_msim_image0(msims[0]).sizes.get('z', 1) > 1 else [])
                        levels = export_levels(source_spacings, output_stack_properties, scaled_dims, native=is_native)
                        # native: tiles over the overview, where a finer source covers a pixel the coarser one under it
                        # is left out, each level from the sources it has; a set spacing averages them as one, and
                        # every level has all of them, so averaging the level above gives it
                        fused_image = fuse_levels_to_ome_zarr(
                            msims, levels, output_filename, transform_key, output_stack_properties, output_chunksize,
                            fusion_func=fuse_func, zarr_options=zarr_options,
                            batch_options=self._fusion_batch_options(saving_zarr, max_workers=workers),
                            ranks=source_spacings if is_native else None, average_levels=not is_native,
                            level_weights=level_bytes)
                    else:
                        fused_image = fusion.fuse(msims, fusion_func=fuse_func, transform_key=transform_key,
                                                  output_stack_properties=output_stack_properties,
                                                  output_chunksize=output_chunksize)
            else:
                # 'compose' mode: no fusion, the sources as they are
                fused_image = msims
                if output_filename is not None:
                    if str(ome_version) != '0.6':
                        logging.warning(f'Compose writes an OME-Zarr scene, which only 0.6 has: written as 0.6, '
                                        f'not {ome_version}')
                    if not output_filename.lower().endswith('.zarr'):
                        output_filename += zarr_extension
                    self.save_composed(output_filename, msims, channels)
                    saving_zarr = True
        return fused_image, saving_zarr

    def save_pair_results(self, pair_mappings, metrics=None, qualities=None):
        """The pair mappings file, each pair with its quality (from `metrics` unless given) and its pairs graph bbox."""
        if qualities is None:
            qualities = {key: metric[default_transform_key][default_quality_key]
                         for key, metric in metrics['pairs'].items() if default_quality_key in metric[default_transform_key]}
        bboxes = {key: np.array(value.sel(t=0) if 't' in value.dims else value).tolist()
                  for key, value in nx.get_edge_attributes(self.pairs_graph, 'bbox').items()}
        self.save_pair_mappings(pair_mappings, qualities, bboxes)

    def save_pair_mappings(self, mappings, qualities, bboxes):
        pair_mappings_filename = self.output + self.input_output_params.get('pair_mappings', default_pair_mappings_name)
        output_mappings = {}
        for keys, mapping in mappings.items():
            label_key = json.dumps([self.filenames[keys[0]], self.filenames[keys[1]]])
            output_mappings[label_key] = {'mapping': np.array(mapping.sel(t=0)).tolist()}
            if keys in qualities:
                quality = quality_to_scalar(qualities[keys])
                output_mappings[label_key][default_quality_key] = float(quality)
            if keys in bboxes:
                output_mappings[label_key]['bbox'] = bboxes[keys]
        export_json(pair_mappings_filename, output_mappings)

    def group_pair_metrics(self):
        return {labels: {self.reg_transform_key: entry.get('metrics', {})
                         | {default_quality_key: entry[default_quality_key]}}
                for labels, entry in self.group_pairs.items()}

    def save_group_pair_mappings(self):
        """Split pairing's group (e.g. section) pairs added to the pair mappings file, keyed by their labels and marked
        by kind, replacing those of an earlier run."""
        pair_mappings_filename = self.output + self.input_output_params.get('pair_mappings', default_pair_mappings_name)
        output_mappings = import_json(pair_mappings_filename) if os.path.exists(pair_mappings_filename) else {}
        output_mappings = {key: value for key, value in output_mappings.items()
                           if value.get('kind') != split_group_pair_kind}
        for labels, entry in self.group_pairs.items():
            # its image metrics go in the metrics file, as every pair's do
            output_mappings[json.dumps(list(labels))] = {'kind': split_group_pair_kind} | {
                key: value for key, value in entry.items() if key != 'metrics'}
        export_json(pair_mappings_filename, output_mappings)

    def save_mappings(self, mappings):
        mappings_filename = self.output + self.input_output_params.get('mappings', default_mappings_name)
        output_mappings = {self.filenames[int(key)]: np.array(mapping.sel(t=0)).tolist()
                           for key, mapping in mappings.items()}
        write_transforms(mappings_filename, output_mappings, self.source_transform_key, self.reg_transform_key)

    def save_mappings_csv(self, mappings, normalise_orientation=False):
        data = []
        mappings_header = ['id',' filename', 'x_pixels', 'y_pixels', 'z_pixels', 'x', 'y', 'z', 'rotation']
        mappings_filename = self.output + self.input_output_params.get('mappings', default_mappings_tabular_name)
        for label, filename, msim, mapping, scale, position, rotation \
                in zip(self.file_labels, self.filenames, self.msims, mappings.values(), self.scales, self.positions, self.rotations):
            if not normalise_orientation:
                # rotation already in msim affine transform
                rotation = None
            position, rotation = get_data_mapping(msim,
                                                  transform_key=self.reg_transform_key,
                                                  transform=mapping,
                                                  translation0=position,
                                                  rotation=rotation)
            position_pixels = {dim: position[dim] / float(scale.get(dim, 1)) for dim in position.keys()}
            row = ([label] + [filename] + dict_to_xyz(position_pixels, add_zeros=True)
                   + dict_to_xyz(position, add_zeros=True) + [rotation])
            data.append(row)
        export_csv(mappings_filename, data, header=mappings_header)

    def save_metrics(self, metrics):
        """The registered pairs' metrics, then split pairing's group pairs' (without filenames), also read back on
        resuming."""
        pair_metrics = {keys: {metric: float(value) for metric, value in metric_dict[self.reg_transform_key].items()}
                        for keys, metric_dict in metrics['pairs'].items() if metric_dict[self.reg_transform_key]}
        group_metrics = {labels: metric_dict[self.reg_transform_key]
                         for labels, metric_dict in metrics.get('group_pairs', {}).items()}
        metric_names = list(dict.fromkeys(metric for values in list(pair_metrics.values()) + list(group_metrics.values())
                                          for metric in values))
        rows = [[self.file_labels[keys[0]], self.file_labels[keys[1]], self.filenames[keys[0]], self.filenames[keys[1]]]
                + [values.get(metric) for metric in metric_names]
                for keys, values in sorted(pair_metrics.items())]
        rows += [list(labels) + ['', ''] + [values.get(metric) for metric in metric_names]
                 for labels, values in group_metrics.items()]
        export_csv(self.output + metrics_name, rows, header=pair_table_columns + metric_names)

    def log_residuals(self, residuals):
        """Each pair's residual from the global solution (RMS distance, physical units), the largest first."""
        if residuals:
            values = list(residuals.values())
            pair_lines = [f'  {self.file_labels[keys[0]]} - {self.file_labels[keys[1]]}: {value:.4g}'
                          for keys, value in sorted(residuals.items(), key=lambda item: -item[1])]
            logging.info(f'Global registration residuals over {len(values)} pairs: mean {np.mean(values):.4g},'
                         f' median {np.median(values):.4g}, max {np.max(values):.4g}\n' + '\n'.join(pair_lines))

    def save_residuals(self, residuals):
        rows = [[self.file_labels[keys[0]], self.file_labels[keys[1]], self.filenames[keys[0]], self.filenames[keys[1]],
                 value] for keys, value in sorted(residuals.items())]
        export_csv(self.output + residuals_name, rows, header=pair_table_columns + ['residual'])

    def create_preview(self, output_filename=None, nom_msims=None, transform_key=None):
        preview_scale = parse_scale(self.input_output_params.get('preview_scale'), default=16)
        is_stack = self.is_stack
        z_scale = get_metadata_z_scale(self.extra_metadata)

        msims = select_msim_subpyramid_at_scale(self.msims, self.sources, preview_scale)
        msims = self.flatfield_corrected_msims(msims, self.preprocess_params)
        if is_stack:
            msims = make_msims_3d(msims, z_scale, self.positions)

        if nom_msims is not None:
            preview_image0 = get_msim_image0(msims[0])
            nom_image0 = get_msim_image0(nom_msims[0])
            if preview_image0.sizes['x'] >= nom_image0.sizes['x']:
                logging.warning('Unable to generate scaled down preview due to lack of source pyramid sizes')
                return None

            if transform_key is not None and transform_key != self.source_transform_key:
                copy_transforms_to_msims(nom_msims, msims, transform_key)
        fusion_method, _ = self.fusion_settings()
        fused_msim, is_saved = self.fuse(msims, fusion_method=fusion_method, transform_key=transform_key,
                                         output_spacing='max', output_filename=output_filename)
        fused_sim = extract_sims_from_fused(fused_msim)
        preview_format = self.fusion_params.get('preview')
        if output_filename and (not is_saved or 'tif' in preview_format):
            self.save(output_filename, fused_sim.squeeze(), transform_key=transform_key, format=preview_format,
                      ome_version=self.fusion_params.get('ome_version') or default_ome_zarr_version)
        return fused_sim

    def save(self, output_filename, data, format=zarr_extension, transform_key=None, translations0=None, channels=[],
             block_size=None, compression=None, pyramid_downsample=2, npyramid_add=4, ome_version=default_ome_zarr_version):
        if output_filename is not None:
            output_filename = self.output + output_filename
        if isinstance(self.extra_metadata, dict) and self.extra_metadata:
            channels = self.extra_metadata.get('channels', [])
        save_image(output_filename, data, format,
                   transform_key=transform_key, channels=channels, translations0=translations0,
                   block_size=block_size, compression=compression,
                   pyramid_downsample=pyramid_downsample, npyramid_add=npyramid_add,
                   ome_version=ome_version,
                   verbose=self.verbose)

    def save_native_levels(self, output_filename, msim, position=None, channels=None, min_length=128,
                           compression=None, ome_version=default_ome_zarr_version):
        """Write msim's own native pyramid levels as-is, without resampling (see save_ome_multiscale_levels()).

        position: this source's position; a source without a 'z' dim is promoted to keep its z position.
        """
        if output_filename is not None:
            output_filename = self.output + output_filename
        if channels is None:
            channels = self.extra_metadata.get('channels', []) if isinstance(self.extra_metadata, dict) else []

        if position is not None and 'z' not in get_msim_dims(msim):
            msim = make_msims_3d([msim], z_scale=self._msim_z_scale, positions=[position])[0]

        levels, dim_order, translation = msim_native_levels(msim)
        save_ome_multiscale_levels(str(output_filename) + zarr_extension, levels, dim_order, channels,
                                   translation, min_length=min_length,
                                   compression=compression, ome_version=ome_version)

    def save_composed(self, output_filename, msims, channels, min_length=128, compression=None):
        """Each source as its own image at its native levels, in one OME-Zarr 0.6 scene with a coordinate system per
        transform key; the registered one sorts first, so it is the default view."""
        transform_keys = [key for key in sorted([self.reg_transform_key, self.source_transform_key])
                          if key in get_msim_transform_keys(msims[0])]
        labels = self.file_labels if len(self.file_labels) == len(msims) else [str(index) for index in range(len(msims))]
        sources = {}
        for label, msim in zip(labels, msims):
            levels, dim_order, translation = msim_native_levels(msim)
            affines = {}
            for transform_key in transform_keys:
                affine = msi_utils.get_transform_from_msim(msim, transform_key)
                affines[transform_key] = affine.sel(t=0) if 't' in affine.dims else affine
            spatial_dims = [str(dim) for dim in affines[transform_keys[0]].coords['x_in'].values[:-1]]
            sources[label] = (levels, dim_order, translation, spatial_dims, affines)
        save_ome_zarr_scene(output_filename, sources, transform_keys, channels, min_length=min_length,
                            compression=compression)

    def save_video(self, output, msims, fused_msim):
        logging.info('Creating transition video...')
        sims = [msi_utils.get_sim_from_msim(msim, scale='scale0') for msim in msims]
        fused_image = extract_sims_from_fused(fused_msim)
        pixel_size = [si_utils.get_spacing_from_sim(sims[0]).get(dim, 1) for dim in 'xy']
        params = self.fusion_params.get('transition', {})
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

    def summary_metrics(self):
        """The registration's metrics over all its pairs, by name; empty without a registration."""
        summary = self.metrics.get('summary', {})
        if self.reg_transform_key in summary:
            # the pairs' registration quality is kept apart from the metrics of their registered overlaps
            values = summary[self.reg_transform_key] | summary.get(default_transform_key, {})
        else:
            # loaded from the metrics file, without the overlap areas the summary weighs its pairs by
            pair_metrics = [pair.get(self.reg_transform_key, {}) for pair in self.metrics.get('pairs', {}).values()]
            names = {name for metrics in pair_metrics for name in metrics}
            values = {name: np.mean([metrics[name] for metrics in pair_metrics if name in metrics]) for name in names}
        return {name: float(value) for name, value in values.items() if value is not None and np.isfinite(value)}

    def get_metrics(self, metric_key=None, pair=None):
        metrics = self.metrics
        if pair is not None:
            if isinstance(pair, np.ndarray):
                pair = pair.tolist()
                pair = tuple(pair)
            metrics = metrics.get('pairs', {}).get(pair, {})
        else:
            if 'summary' in metrics:
                metrics = metrics['summary']
        reg_transform_key = self.reg_transform_key
        if reg_transform_key in metrics:
            transform_key = reg_transform_key
        elif default_transform_key in metrics:
            transform_key = default_transform_key
        elif metrics:
            transform_key = list(metrics)[-1]
        else:
            transform_key = None
        metrics = metrics.get(transform_key, {})
        if metric_key is not None:
            return metrics.get(metric_key)
        else:
            return metrics

    def current_fused_output(self, output_filename):
        """The fused OME-Zarr's path if it was written after every saved registration file (it shows that
        registration), else None."""
        path = self.output + output_filename + zarr_extension
        metadata = [os.path.join(path, name) for name in ('zarr.json', '.zattrs')
                    if os.path.exists(os.path.join(path, name))]
        if not metadata:
            return None
        registration_times = [os.path.getmtime(saved) for saved in self.saved_progress_paths()]
        if registration_times and max(registration_times) > os.path.getmtime(metadata[0]):
            return None
        return path

    def output_exists(self, output_filename, output_format):
        if not output_format.startswith('.'):
            output_format = '.' + output_format
        output_filename = self.output + output_filename + output_format
        if output_format == zarr_extension:
            return (os.path.exists(os.path.join(output_filename, '.zattrs')) or
                    os.path.exists(os.path.join(output_filename, 'zarr.json')))
        else:
            return os.path.exists(output_filename)
