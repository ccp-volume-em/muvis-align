from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import logging
from enum import Enum, auto
from magicclass.ext.napari import ViewerWidget
from multiview_stitcher import spatial_image_utils as si_utils, param_utils
from napari.qt.threading import create_worker
from napari.utils import progress
from napari.utils.notifications import show_warning
import networkx as nx
import numpy as np
import os.path
from xarray import DataTree
from qtpy.QtCore import QEventLoop, QObject, QTimer, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication, QMessageBox

from muvis_align.constants import zarr_extension, default_transform_key, default_quality_key, \
    default_interactive_preview_scale, default_preview_workers, default_chunk_size
from muvis_align.file.project_yaml import read_params, get_template_params, write_params, update_params
from muvis_align.MVSRegistration import MVSRegistration, RegState
from muvis_align.image.util import get_sim_physical_size, get_sim_position_final, \
    create_image_shapes, create_overlap_shapes, build_source_stack_props, \
    draw_keypoints_matches_napari, get_transforms, copy_transforms_to_msims, \
    make_msims_3d, metric_to_rgb, get_msim_level_data, get_contrast_limits, \
    get_msim_image0, wrap_sims_as_msims, extract_sims_from_fused, extract_sims_from_msims, \
    select_msim_subpyramid_at_scale, reduce_msims_to_fused_size, composite_msims_overview
from muvis_align.file.resources import get_project_template
from muvis_align.logging import init_logging
from muvis_align.metrics import calc_msims_metrics
from muvis_align.Timer import Timer
from muvis_align.ui.NapariDaskProgress import NapariDaskProgress
from muvis_align.ui.MagicColorPicker import MagicColorPicker
from muvis_align.ui.NapariMVSProgress import NapariMVSProgress
from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress
from muvis_align.ui.ParamWidget import create_dict_of_lists, update_dict_value
from muvis_align.ui._utils import TemporarilyDisabledWidgets, VisibleActivityDock, catch_run_errors
from muvis_align.ui.bilayers_util import get_section_dict
from muvis_align.util import print_dict_simple, set_dict_value, is_valid_value, \
    calculate_rigid_difference, operation_to_past_participle, eval_path, path_param_to_text, \
    resolve_to_project_dir, relativize_to_project_dir


class _ProgressBridge(QObject):
    """Carries a worker thread's progress positions back to the Qt thread, where the bar lives."""

    moved = Signal(int)


class ViewMode(Enum):
    OVERVIEW = auto()
    PAIRS = auto()
    FEATURES = auto()
    FUSED = auto()


def position_sort_key(position):
    return position.get('z', 0), position.get('y', 0), position.get('x', 0)


def parse_channel_color(color):
    if isinstance(color, str):
        try:
            return tuple(eval(color))
        except Exception:
            return None
    return color


class Interface:
    def __init__(self, viewer, overview, enable_tabs=None, select_tab=None, is_tab_enabled=None,
                 enable_tab=None, enable_plugin_widget=None, verbose=False, initialize=True):
        self.viewer = viewer
        self.overview = overview
        self.enable_tabs = enable_tabs
        self.select_tab = select_tab
        self.is_tab_enabled = is_tab_enabled
        self.enable_tab = enable_tab
        self.enable_plugin_widget = enable_plugin_widget
        self.verbose = verbose
        self.raw_template = get_project_template()
        if not self.raw_template:
            raise FileNotFoundError('Project template not found')
        self.template = get_section_dict(self.raw_template, ['inputs', 'parameters', 'display_only', 'outputs'])
        self.param_widgets = {}
        self.params = {}
        self.pre_processing_performed = False
        self.metrics_methods = ['ncc', 'ssim', 'onmi']
        self.transform_key = 'source_metadata'
        self.need_source_reinit = False

        self.pair_metrics_timer = QTimer()
        self.pair_metrics_timer.setSingleShot(True)
        self.pair_metrics_timer.setInterval(1000)
        self.pair_metrics_timer.timeout.connect(self.update_pair_metrics)

        self.reg = MVSRegistration()
        if initialize:
            self.reset()

    def reset(self):
        self.source_metadata = {}
        self.extra_metadata = {}
        self.output_channels = []
        self.view_mode = None
        self._preview_overlap_cache = None
        self._view_msims = None
        self.reg.reset()
        self._clear_napari_view(self.overview)
        self._clear_napari_view(self.viewer)
        if self.enable_tabs:
            self.enable_tabs(False, 2)
        if self.select_tab:
            self.select_tab(1)

    def get_all_widgets(self):
        # excludes widgets on a currently disabled tab - their .enabled always reads False
        # (inherited from the disabled tab page), so snapshotting and restoring it as an explicit
        # per-widget state (see modify_pair_registration) would leave them disabled even after
        # their tab is enabled again
        return {name: param_widget.widget for name, param_widget in self.param_widgets.items()
               if self.is_tab_enabled is None or self.is_tab_enabled(name.split('.', 1)[0])}

    def get_function(self, function_label):
        if hasattr(self, function_label):
            return eval(f'self.{function_label}')
        else:
            return None

    def tab_changed(self, tab_label):
        if tab_label != 'registration' and self.view_mode == ViewMode.FEATURES:
            self._clear_napari_view(self.viewer)
            self.view_mode = None
        self.pair_metrics_timer.stop()

    def project_path(self, path):
        self.reset()
        self.params_path = path
        self.params = get_template_params(self.template)
        if os.path.exists(path):
            self.params = update_params(self.params, read_params(path))
            self.update_widgets()
        else:
            self.write_params()
        self.update_input_output_path()

    def get_project_dir(self):
        # input/output path params are stored relative to this directory, so a project stays
        # portable when the project file and its data are moved or shared together
        params_path = getattr(self, 'params_path', None)
        return os.path.dirname(os.path.abspath(params_path)) if params_path else None

    def update_widgets(self):
        for param_name, param_widget in self.param_widgets.items():
            # input/output path widgets are handled separately by update_input_output_path(),
            # which resolves them relative to the project directory before display
            if param_name not in ('input_output.input_path', 'input_output.output_path'):
                keys = param_name.split('.')
                value = self.params.get(keys[0], {}).get(keys[1])
                if value is not None:
                    param_widget.set_value(value)
                    if param_name == 'input_output.channels_table':
                        # a project loaded from disk already has channels, so
                        # update_output_channels() (and its populate_channels_table() call)
                        # never runs for it - attach the color pickers here instead
                        self.populate_channels_table_color_pickers()

    def write_params(self):
        write_params(self.params_path, self.params)

    def change_param(self, param_name, value):
        keys = param_name.split('.')
        if keys[0] not in self.params:
            self.params[keys[0]] = {}
        if isinstance(value, str):
            value = value.replace('\\', '/')
            if param_name in ('input_output.input_path', 'input_output.output_path'):
                # the file dialog (and the FileEdit widget itself) always reports an
                # absolute path - convert it back to relative-to-project-dir before storing,
                # so the project file keeps portable relative paths
                value = relativize_to_project_dir(value, self.get_project_dir())
        self.params[keys[0]][keys[1]] = value
        self.write_params()

    def update_input_output_path(self):
        # display the path exactly as stored (relative-to-project-dir when the project file
        # keeps it relative) - FileEdit.set_value() would force it absolute, so the inner
        # line edit's text is set directly instead, bypassing that conversion
        params = self.params['input_output']
        widget = self.param_widgets.get('input_output.input_path')
        # several comma-separated globs are one valid path value - show them as they are stored
        # rather than leaving the widget empty (the widget holds plain text, not a single file)
        input_path = path_param_to_text(params.get('input_path', ''))
        if widget is not None:
            self._set_path_widget_text(widget, input_path)
        widget = self.param_widgets.get('input_output.output_path')
        output_path = path_param_to_text(params.get('output_path', ''))
        if widget is not None:
            self._set_path_widget_text(widget, output_path)
        resolved_output_path = resolve_to_project_dir(output_path, self.get_project_dir())
        init_logging(log_filename=os.path.join(resolved_output_path, 'muvis-align.log'), verbose=self.verbose)

    def _set_path_widget_text(self, param_widget, value):
        line_edit = getattr(param_widget.widget, 'line_edit', None)
        if line_edit is not None:
            line_edit.value = value
        else:
            param_widget.set_value(value)

    def input_path(self, value):
        self.need_source_reinit = True

    def source_position_z(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['position', 'z'], value)
            self.need_source_reinit = True

    def source_position_y(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['position', 'y'], value)
            self.need_source_reinit = True

    def source_position_x(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['position', 'x'], value)
            self.need_source_reinit = True

    def source_scale_z(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['scale', 'z'], value)
            self.need_source_reinit = True

    def source_scale_y(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['scale', 'y'], value)
            self.need_source_reinit = True

    def source_scale_x(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['scale', 'x'], value)
            self.need_source_reinit = True

    def source_rotation(self, value):
        if is_valid_value(value):
            set_dict_value(self.source_metadata, ['rotation'], value)
            self.need_source_reinit = True

    def registration_dimension(self, value):
        # Force reinitialization of extra metadata / channels when registration dimension changes
        self.extra_metadata.pop('channels', None)

    def channels_table(self, value):
        old_value = self.param_widgets.get('input_output.channels_table').get_value()
        channels_dict = update_dict_value(old_value, value)
        channels = [{'label': label} for label in channels_dict['label']]
        for channeli, channel in enumerate(channels):
            if channeli < len(channels_dict['color']):
                color = parse_channel_color(channels_dict['color'][channeli])
                if color is not None:
                    channel['color'] = color
        self.extra_metadata['channels'] = channels

    def input_output_process(self):
        # re-sync path widget display to the normalised value now, not live while typing
        self.update_input_output_path()
        params = self.params['input_output']
        project_dir = self.get_project_dir()
        output = resolve_to_project_dir(path_param_to_text(params['output_path']), project_dir)
        if not self.reg.is_initialised() or self.need_source_reinit:
            self.need_source_reinit = False
            if not output.endswith('/'):
                output += '/'
            input_path = resolve_to_project_dir(path_param_to_text(params['input_path']), project_dir)
            # one bar for opening the project - reading the sources and loading any saved
            # registration. It finishes before the view work starts, which shows the one bar
            # after it (see _show_loaded_project(), _operation_progress())
            with self._operation_progress('Initialising sources', phases=1) as factory:
                ok = self.reg.init(input_path=eval_path(input_path),
                                   output_path=output,
                                   overwrite=params['overwrite'],
                                   pairing=self.params['registration'].get('pairing', ''),
                                   verbose=self.verbose)
                if ok:
                    # resuming a saved registration does far more here than a fresh open (see
                    # _load_saved_progress) - reserve that room before reading the sources, the
                    # one phase a fresh open has, sizes itself against the whole bar
                    operation = self.params['registration'].get('operation', '')
                    fused_name = operation_to_past_participle(operation) if operation else None
                    if self.reg.has_saved_progress(fused_name, zarr_extension):
                        factory.ensure_phases(4)
                    # _show_loaded_project() below always ends by drawing the view, so drawing
                    # here would only draw once with the not-yet-registered transform
                    ok = self.update_metadata_source(skip_view_update=True, progress_factory=factory)
                    if ok:
                        self.populate_image_selection()
                        self._load_saved_progress(factory)
            if ok:
                self._show_loaded_project()
            else:
                show_warning('Invalid input or output')
                self.reg.state = RegState.UNINIT
        elif self.reg.is_global_registered():
            self.update_registered(view_transform_key=self.reg.reg_transform_key)
        elif self.reg.is_pairs_registered():
            self.update_registered(view_transform_key=self.reg.source_transform_key)
        else:
            self.update_metadata_source()

    def _run_off_thread(self, work, progress_factory):
        """Run work(progress_factory) on a worker thread, and wait for it here.

        The heavy calls (registration, fusion, pre-processing, source and view building) are
        pure computation that never touches the viewer, but they used to run on the Qt thread,
        which is what froze the window for as long as they took - global registration being the
        worst of it, one blocking call in multiview_stitcher with nothing to report from inside.
        Running them on a worker while a nested event loop keeps Qt going leaves the window
        alive: the bar animates, the viewer still pans and zooms, and the plugin's own widgets
        stay disabled for the duration as they already did.

        `work` is handed a headless progress factory of its own (see NapariPhaseProgress's
        `emit`), whose positions cross back to the bar here as a queued signal - a worker must
        never touch a Qt widget itself. Anything a phase needs patched for it (dask callbacks,
        multiview_stitcher's tqdm) belongs inside `work`, so that it too reports to that factory
        rather than to the bar directly.

        Without a Qt application - tests, and any headless caller - the work simply runs here.
        """
        app = QApplication.instance() if QApplication is not None else None
        if app is None or progress_factory is None:
            return work(progress_factory)

        bridge = _ProgressBridge()
        bridge.moved.connect(progress_factory.set_position, Qt.QueuedConnection)
        # started where the bar has got to, not at zero (NapariPhaseProgress.worker_twin()): an
        # operation runs several of these in turn, and a twin re-planning from empty each time
        # reported positions the bar was already past, freezing it partway
        worker_factory = progress_factory.worker_twin(bridge.moved.emit)

        def run():
            with worker_factory as factory:
                return work(factory)

        outcome = {}
        loop = QEventLoop()
        # the handlers go in at creation: given none for 'errored', create_worker() adds one of
        # its own that re-raises inside the Qt event loop, where nothing can catch it (the
        # failure belongs to the caller here, and to @catch_run_errors above it)
        worker = create_worker(
            run,
            _start_thread=False,
            _connect={
                'returned': lambda value: outcome.__setitem__('value', value),
                'errored': lambda error: outcome.__setitem__('error', error),
                'finished': loop.quit,
            },
        )
        worker.start()
        loop.exec_()
        # whatever the twin accounted for is now the bar's own state, so the next off-thread
        # call of this operation continues from here instead of dividing it up again
        progress_factory.continue_from(worker_factory)
        if 'error' in outcome:
            raise outcome['error']
        return outcome.get('value')

    @contextmanager
    def _operation_progress(self, desc, progress_factory=None, phases=1):
        """Progress reporting for one user-facing operation: the activity dock, disabled widgets,
        and a single NapariPhaseProgress bar that every phase of the operation reports into (see
        NapariPhaseProgress - phases name themselves in its description rather than each opening
        a bar of their own).

        There is never more than one bar: an operation that runs inside another (the caller
        passed its factory, or one is simply already running) reports into that one and leaves
        the dock and the widget state to it.

        Computing and then showing the result are deliberately kept as two operations, in that
        order - one bar for the work (pre-processing, registration, fusion, opening a project),
        which finishes and disappears, and then one for building and drawing the view.

        `phases` is how many phases the operation expects to report - it sizes their slices of
        the one bar, which fills once from empty to full (see NapariPhaseProgress); being out
        by one only makes the bar move unevenly, never wrong.
        """
        factory = progress_factory or getattr(self, '_running_operation', None)
        if factory is not None:
            # the bar is someone else's, but this operation's phase count still has to reach it,
            # or its phases each take most of whatever the outer operation had left
            factory.ensure_phases(phases)
            yield factory
            return
        with NapariPhaseProgress(progress_class=progress, desc=desc, phases=phases,
                                 min_duration=0.1) as factory, \
             TemporarilyDisabledWidgets(self.enable_plugin_widget), \
             VisibleActivityDock(self.viewer):
            self._running_operation = factory
            try:
                yield factory
            finally:
                self._running_operation = None

    def init_progress(self):
        # the two halves of opening a project, each its own bar, one after the other
        with self._operation_progress('Initialising sources', phases=3) as progress_factory:
            self._load_saved_progress(progress_factory)
        self._show_loaded_project()

    def _load_saved_progress(self, progress_factory=None):
        output_filename = operation_to_past_participle(self.params['registration']['operation'])
        # Resuming a saved project does real work: reg.init_progress() forces the per-source
        # msim build (~15s for 328 sources) and loads/redimensions the saved registration, and
        # a reloaded pair registration has to be re-preprocessed.
        with self._operation_progress('Initialising sources', progress_factory,
                                      phases=3) as progress_factory:
            self._run_off_thread(
                lambda worker_factory: self.reg.init_progress(output_filename, zarr_extension,
                                                              progress_factory=worker_factory),
                progress_factory)
            if self.reg.is_pairs_registered() and self.reg.register_msims is None:
                # loading a saved pair registration sets pair_msims straight from the raw full
                # pyramid, bypassing the scale reduction a live run applies first - after which a
                # global registration's metrics can select full resolution and run out of memory.
                # Run the same preprocessing now, matching what a fresh run would have produced.
                self.run_pre_processing(progress_factory=progress_factory)
                self.reg.pair_msims = self.reg.register_msims

    def _show_loaded_project(self):
        # building the view data and drawing it - the second bar of opening a project, shown
        # once the load above has finished
        if self.reg.is_fused():
            with self._operation_progress('Refreshing view', phases=2) as view_factory:
                self._copy_transforms_to_view_msims(self.reg.reg_transform_key,
                                                    progress_factory=view_factory)
                self.preview_fusion(progress_factory=view_factory)
            self.enable_tabs(True, 4)
            self.select_tab(4)
        elif self.reg.is_global_registered():
            with self._operation_progress('Refreshing view', phases=2) as view_factory:
                self._copy_transforms_to_view_msims(self.reg.reg_transform_key,
                                                    progress_factory=view_factory)
                self.update_registered(view_transform_key=self.reg.reg_transform_key,
                                       progress_factory=view_factory)
            self.enable_tabs(True, 4)
            self.select_tab(4)
        elif self.reg.is_pairs_registered():
            self.update_registered(view_transform_key=self.reg.source_transform_key)
            self.enable_tabs(True, 3)
            self.select_tab(3)
        else:
            # No prior registration to view with a specific transform - this is the one
            # draw update_metadata_source()'s own (skipped, see input_output_process())
            # would otherwise have done for a brand-new project. No pre-processing has run
            # yet, so only shapes are shown - see update_views()'s show_images param.
            self.update_views(show_images=False)
            self.enable_tabs(True, 2)

    def update_metadata_source(self, skip_view_update=False, progress_factory=None):
        if not self.reg.is_pairs_registered():
            try:
                with self._operation_progress('Initialising sources',
                                              progress_factory) as factory:
                    self._run_off_thread(
                        lambda worker_factory: self.reg.init_data(
                            source_metadata=self.source_metadata,
                            progress_factory=worker_factory,
                        ),
                        factory)
            except ValueError as e:
                show_warning('Unable to read source data\n' + str(e))
                logging.exception('Unable to read source data')
                return False

            # building each source's multiscale msim is deferred to the lazy view_msims
            # property, forced only once something needs pixel-shaped data. The z-index
            # remapping is cheap and stays eager, so reg.positions is immediately correct for
            # the shapes, which are drawn instantly either way.
            z_positions = sorted(set([position.get('z', 0) for position in self.reg.positions]))
            if len(z_positions) > 1:
                for position in self.reg.positions:
                    position['z'] = z_positions.index(position.get('z', 0))
            self._view_msims = None
        # this function only ever runs pre-registration (both call sites guard on
        # `not self.reg.is_pairs_registered()`), so self.reg.source_transform_key is the only
        # transform key that can exist yet - no need to force self.reg.msims just to read it
        # off a msim that would say the same thing
        coord_systems = [self.reg.source_transform_key]
        self.populate_channels()
        self.populate_coordinate_systems(coord_systems)
        if self.update_output_channels():
            self.populate_channels_table()
        if self.reg.is_initialised():
            # populate_metadata_table() never reads its first arg when transform_keys is None
            # (it reads self.reg.positions/scales directly) - passing None instead of
            # self.reg.msims avoids forcing the expensive msim build just for this call
            self.populate_metadata_table(None)
            self.check_3d_view()
            if not skip_view_update:
                # no factory: by here the source initialisation above has finished, and drawing
                # the view is the next operation (or a phase of the caller's, if one is still
                # running - see _operation_progress())
                self.update_views(show_images=False)

        return True

    @property
    def view_msims(self):
        # per-source msim for the napari image data layer - built lazily, only once something
        # (the fused image preview) actually needs it; shapes no longer depend on this at all
        # (see _create_napari_shapes(), which builds its own cheap per-source sims directly)
        return self.ensure_view_msims()

    def ensure_view_msims(self, progress_factory=None):
        # the same lazy build view_msims triggers, callable ahead of time with a progress_factory
        # so a caller forcing it can report per source - mirrors MVSRegistration.ensure_msims()
        if self._view_msims is None:
            self._view_msims = self._build_view_msims(progress_factory=progress_factory)
            if len(set(position.get('z', 0) for position in self.reg.positions)) > 1:
                self._view_msims = make_msims_3d(self._view_msims, positions=self.reg.positions)
        return self._view_msims

    def _copy_transforms_to_view_msims(self, transform_key, progress_factory=None):
        # forcing view_msims builds one msim per source (and, for a multi-z set, wraps the whole
        # lot into a volume) - seconds to tens of seconds for a few hundred sources, so report it
        # rather than letting it run silently as a side effect of an argument expression
        with self._operation_progress('Building views', progress_factory) as factory:
            view_msims = self._run_off_thread(
                lambda worker_factory: self.ensure_view_msims(progress_factory=worker_factory),
                factory)
        copy_transforms_to_msims(self.reg.msims, view_msims, transform_key)

    @view_msims.setter
    def view_msims(self, value):
        self._view_msims = value

    def _build_view_msims(self, progress_factory=None):
        # per-source msim for the napari image data layer: a source with a native multi-
        # resolution pyramid is used as-is; a single-resolution source is downscaled by one
        # constant factor when its largest spatial dimension exceeds 1000px
        view_msims = []
        progress_context = (
            progress_factory(total=len(self.reg.sources), desc='Building views')
            if progress_factory is not None
            else nullcontext(None)
        )
        with progress_context as pbar:
            for source, msim in zip(self.reg.sources, self.reg.msims):
                if len(source.shapes) == 1:
                    image0 = get_msim_image0(msim)
                    spatial_dims = si_utils.get_spatial_dims_from_sim(image0)
                    largest_dim = max(image0.sizes[dim] for dim in spatial_dims)
                    if largest_dim > 1000:
                        scale_factor = largest_dim / 1000
                        sim = extract_sims_from_msims(
                            [msim], [source], self.reg.source_transform_key, target_scale=scale_factor
                        )[0]
                        msim = wrap_sims_as_msims([sim])[0]
                view_msims.append(msim)
                if pbar is not None:
                    pbar.update(1)
        return view_msims

    @catch_run_errors
    def run_pre_processing(self, progress_factory=None):
        params_features = self.params['pre_processing']
        # both phases below (per-source msim build, then pre-processing itself) report into the
        # one bar this opens - or into the caller's, when pre-processing is a phase of a larger
        # operation such as loading a saved project. The build only reports when it actually
        # runs: reserving its half for msims already built left the bar (and its time estimate)
        # finishing at 50%, the end of the one phase that did run.
        # It also takes the bar in proportion to what it costs: opening every source is 15.6 of
        # the 16.2 seconds of a 328-source run whose only step is scaling, and splitting the bar
        # evenly put the end of the work at the halfway mark just the same. A step that computes
        # over the data (flat-field, normalisation, foreground) makes the rest real work again.
        build_weight = 2 if self.reg.has_eager_pre_processing(params_features) else 8
        build_pending = self.reg.msims_build_pending(params_features.get('scale'))
        phases = build_weight + 1 if build_pending else 1
        with self._operation_progress('Pre-processing', progress_factory, phases=phases) as progress_factory, \
             Timer('pre_processing_process', verbose=self._timing_verbose()):
            # self.reg.msims is built lazily (see MVSRegistration.msims) - building it here
            # explicitly, through ensure_msims(), gives that per-source construction its own
            # progress reporting instead of it happening silently (no progress feedback) as a
            # side effect of evaluating `self.reg.msims` as a plain argument below
            def preprocess(worker_factory):
                with Timer('run_pre_processing: build msims (load image data)',
                           verbose=self._timing_verbose()):
                    # at pre-processing's own scale: building the finer levels only to have
                    # select_msim_subpyramid_at_scale() drop them is most of this phase
                    msims = self.reg.ensure_msims(progress_factory=worker_factory,
                                                  target_scale=params_features.get('scale'),
                                                  weight=build_weight)
                with Timer('run_pre_processing: preprocess', verbose=self._timing_verbose()):
                    return self.reg.preprocess(msims, progress_factory=worker_factory,
                                               **params_features)

            _, _, modified = self._run_off_thread(preprocess, progress_factory)
        self.pre_processing_performed = modified
        return True

    def pre_processing_process(self):
        # pre-processing reports its own bar, then the view it leaves on screen reports a second
        # one of its own (update_views()) - the work and showing the result are two operations
        if not self.run_pre_processing():
            return
        self.update_views(show_preprocessed=True)
        self.enable_tabs(True, 3)
        self.enable_modify_pair_registration(False)
        if self.reg.is_pairs_registered():
            # register_msims just changed - prior pair/global registration is stale
            self.reg.state = RegState.SIMS_INIT
        self.select_tab(3)

    def populate_channels(self):
        channel_labels = list({channel.get('label', '') for source in self.reg.sources for channel in source.get_channels()})
        choices = {channel: channel for channel in channel_labels}
        param_widget = self.param_widgets.get('registration.channel')
        param_widget.set_choices(choices)

    def populate_coordinate_systems(self, coord_systems):
        choices = {coord_system: coord_system.replace('_', ' ').capitalize() for coord_system in coord_systems}
        param_widget = self.param_widgets.get('input_output.coordinate_system')
        param_widget.set_choices(choices)

    def coordinate_system(self, transform_key):
        self.transform_key = transform_key
        if self.reg.is_initialised():
            self.populate_metadata_table(self.reg.msims, [transform_key])

    def populate_metadata_table(self, sims, transform_keys=None):
        # https://pyapp-kit.github.io/magicgui/api/widgets/Table/
        # https://pyapp-kit.github.io/magicgui/generated_examples/demo_widgets/table/
        table_widget = self.param_widgets.get('input_output.metadata_table')
        properties = ['position', 'size']
        if transform_keys is None:
            positions = self.reg.positions
            scales = self.reg.scales
        else:
            positions = [get_sim_position_final(sim, transform_keys=transform_keys) for sim in sims]
            scales = [get_sim_physical_size(sim) for sim in sims]
        order = sorted(range(len(positions)), key=lambda i: position_sort_key(positions[i]))
        data = [[print_dict_simple(positions[i]), print_dict_simple(scales[i])] for i in order]
        row_headers = [self.reg.file_labels[i] for i in order]
        # Table: tuple-of-values : ([values], [row_headers], [column_headers])
        table_widget.set_value((data, row_headers, properties))
        table_widget.set_table_column_resize_mode()

    def update_output_channels(self):
        channels = self.extra_metadata.get('channels')
        # a blank-labelled channel (e.g. an old save) is treated as unconfigured, so it re-derives
        if not channels or not any(channel.get('label') for channel in channels):
            # get channels from source
            source0 = self.reg.sources[0]
            channels = source0.get_channels()

            dimension = self.params['input_output']['registration_dimension']
            while dimension.lower() == 'c' and len(channels) < len(self.reg.sources):
                channel = {'label': f'channel {len(channels)}'}
                channels.append(channel)

            self.extra_metadata['channels'] = channels

            # convert to list dict - dict.get()'s default only applies when the key is missing,
            # so an explicit but blank label (e.g. an unnamed OME channel) needs `or`, not `get`
            data = [[channel.get('label') or f'channel {index}', channel.get('color', (1, 1, 1))]
                     for index, channel in enumerate(channels)]
            self.output_channels = create_dict_of_lists(data, ['label', 'color'])
            return True

        return False

    def populate_channels_table(self):
        param_widget = self.param_widgets.get('input_output.channels_table')
        param_widget.set_value(self.output_channels)
        self.populate_channels_table_color_pickers()

    def populate_channels_table_color_pickers(self):
        # replace the plain-text 'color' cells with a MagicColorPicker per channel row, so
        # clicking a channel's color opens a color picker instead of typing a raw tuple
        table = self.param_widgets.get('input_output.channels_table').widget
        if 'color' not in table.column_headers:
            # an empty table has no headers at all - nothing to put a picker on
            return
        color_coli = table.column_headers.index('color')
        for rowi in range(table.shape[0]):
            color = parse_channel_color(table.data[rowi, color_coli]) or (1, 1, 1)
            color_picker = MagicColorPicker(value=color)
            color_picker.changed.connect(
                lambda _=None, picker=color_picker, rowi=rowi: self.channel_color_changed(rowi, picker.value))
            table.native.setCellWidget(rowi, color_coli, color_picker.native)

    def channel_color_changed(self, rowi, color):
        table = self.param_widgets.get('input_output.channels_table').widget
        color_coli = table.column_headers.index('color')
        # writing back into the table's own data model reuses the existing 'changed' wiring
        # (param persistence + self.extra_metadata update), same as a manual text edit would
        table.data[rowi, color_coli] = str(tuple(color))

    def populate_image_selection(self):
        labels = self.reg.file_labels
        widget1 = self.param_widgets.get('registration.reg_preview_image1')
        widget1.set_value(labels[0], choices=labels)

        widget2 = self.param_widgets.get('registration.reg_preview_image2')
        index = 1 if len(labels) > 1 else 0
        widget2.set_value(labels[index], choices=labels)

    def select_pair_preview(self, ref):
        # a plain image shape's ref is a single index (e.g. '0'); only an overlap shape's
        # ref ('0 1') identifies a pair, so single-image clicks are ignored here
        indices = ref.split()
        if len(indices) != 2:
            return
        labels = self.reg.file_labels
        label1, label2 = labels[int(indices[0])], labels[int(indices[1])]
        self.param_widgets.get('registration.reg_preview_image1').set_value(label1)
        self.param_widgets.get('registration.reg_preview_image2').set_value(label2)

    def get_best_transform_key(self):
        if not self.reg.is_pairs_registered():
            # only self.reg.source_transform_key can exist yet - avoid forcing the expensive
            # msim build (get_transforms() needs self.reg.msims) just to learn that
            return self.reg.source_transform_key
        transforms = get_transforms(self.reg.msims)
        if self.reg.reg_transform_key in transforms:
            transform_key = self.reg.reg_transform_key
        elif default_transform_key in transforms:
            transform_key = default_transform_key
        elif self.reg.source_transform_key in transforms:
            transform_key = self.reg.source_transform_key
        else:
            transform_key = None
        return transform_key

    def check_3d_view(self):
        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        ndisplay = 3 if is_3d else 2
        self.viewer.dims.ndisplay = ndisplay
        #self.overview._qtwidget._viewer_model.dims.ndisplay = ndisplay

    def _timing_verbose(self):
        # a few view-update methods below are unit-tested against a bare Interface with no
        # self.reg at all (e.g. _napari_view_add_fused_data) - fall back to no timing logging
        # rather than requiring self.reg just to gate these diagnostic Timer() calls
        return getattr(getattr(self, 'reg', None), 'logging_time', False)

    def update_views(self, transform_key=None, show_preprocessed=False, show_images=True,
                     progress_factory=None):
        if transform_key is None:
            transform_key = self.get_best_transform_key()

        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        is_multi_z_shapes = (len(set(position.get('z', 0) for position in self.reg.positions)) > 1)
        force_2d = is_multi_z_shapes and not is_3d

        # each step is a phase weighted by what it costs, not counted equally: building the view
        # data is 10 of a 4733-source refresh's 11.6 minutes, and as one of five equal steps it
        # left the bar at 18% for ten minutes. It also reports from the inside
        # (_create_napari_data), so the long step moves rather than only bracketing itself.
        view_data_weight = 12
        # building the shapes is the other step that grows with the source count (a geometry per
        # source, then every overlapping pair), and it reports per source from the inside too
        shapes_weight = 3
        phases = shapes_weight + 2 + (view_data_weight + 1 if show_images else 0)

        with self._operation_progress('Refreshing view', progress_factory, phases=phases) as factory:
            with Timer('update_views: create shapes', verbose=self._timing_verbose()):
                # pure geometry, touching no viewer, so it runs off the Qt thread like the
                # fusion below - on the Qt thread it froze the window, and the bar with it,
                # for as long as it took (minutes on a large project)
                shapes, refs, labels, face_colors = self._run_off_thread(
                    lambda worker_factory: self._create_napari_shapes(
                        transform_key, force_2d=force_2d, progress_factory=worker_factory,
                        weight=shapes_weight),
                    factory)

            self._clear_napari_view(self.viewer)
            # only shapes before pre-processing has run: the fused preview needs every source's
            # real msim built, and deferring that keeps it off the initial project load
            if show_images:
                with Timer('update_views: create fused data', verbose=self._timing_verbose()):
                    # the fusion runs off the Qt thread; adding the result to the viewer, below,
                    # must not (see _run_off_thread()). The worker's factory goes all the way in,
                    # so the step reports its own sub-steps instead of being one silent block
                    data = self._run_off_thread(
                        lambda worker_factory: self._create_napari_data(
                            transform_key, show_preprocessed=show_preprocessed, composite=True,
                            progress_factory=worker_factory, weight=view_data_weight),
                        factory)
                if data is not None:
                    with factory(total=1) as pbar, \
                         Timer('update_views: add fused data to viewer', verbose=self._timing_verbose()):
                        # cheap=True: this is the general overview, not the accurate fusion-tab
                        # preview (preview_fusion()) or the real exported result (fusion_process())
                        # - a naive contrast guess is fine here, see _napari_view_add_fused_data()
                        self._napari_view_add_fused_data(self.viewer, data, f'{self.reg.fileset_label} data',
                                                         cheap=True)
                        pbar.update(1)

            with factory(total=1) as pbar, \
                 Timer('update_views: add shapes to viewer', verbose=self._timing_verbose()):
                self._update_view_add_shapes(self.viewer, shapes, refs, labels, face_colors, f'{self.reg.fileset_label} shapes')
                pbar.update(1)

            with factory(total=1) as pbar, \
                 Timer('update_views: refresh overview shapes', verbose=self._timing_verbose()):
                self._refresh_overview_shapes(transform_key, shapes, refs, labels, face_colors, is_3d=is_3d)
                pbar.update(1)
        self.view_mode = ViewMode.OVERVIEW

    def _refresh_overview_shapes(self, transform_key, shapes=None, refs=None, labels=None,
                                 face_colors=None, is_3d=None):
        # the overview widget always shows a flattened, top-down layout, independent of
        # whatever the main viewer currently shows (a fused image, per-tile overview, or
        # nothing loaded yet) - a genuinely 3D shape set (drawn as oriented 3D boxes for the
        # main viewer) needs recomputing with force_2d=True for it instead of being reused as-is
        if is_3d is None:
            is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        if shapes is None or is_3d:
            shapes, refs, labels, face_colors = self._create_napari_shapes(transform_key, force_2d=True)
        self._clear_napari_view(self.overview)
        self._update_view_add_shapes(self.overview, shapes, refs, labels, face_colors,
                                     f'{self.reg.fileset_label} shapes', show_labels=False)

    def _clear_napari_view(self, viewer):
        # Avoid emitting an empty LayerList.clear() event.  Under xpra/Xvfb,
        # that event can leave napari's VisPy canvas with broken blending for
        # subsequently created Shapes and Points layers.
        if viewer is not None and len(viewer.layers) > 0:
            viewer.layers.clear()

    def _create_napari_shapes(self, transform_key, force_2d=False, progress_factory=None, weight=1):
        # `weight` is what the caller's bar allows this step, divided below between the
        # sub-steps in rough proportion to what each costs on a large project. Building the
        # geometries is the one with a real per-source count, and most of the cost with it
        def phase(share, total=None):
            return self._progress_phase(progress_factory, total=total,
                                        weight=max(weight * share, 1))

        if transform_key == self.reg.source_transform_key:
            # not yet registered (or asked for original positions): build cheap single-level
            # sims from the resolved per-source geometry, never touching the expensive msim
            # build. Any other transform_key means registration has run, so view_msims is
            # legitimately available and already carries that transform.
            #
            # promote_z mirrors the make_msims_3d() promotion view_msims gets: with no native
            # 'z' but sources at different heights, each source's z must become a real dim or it
            # is silently dropped rather than drawn at its actual height.
            promote_z = (len(set(position.get('z', 0) for position in self.reg.positions)) > 1)
            with phase(3 / 5, total=len(self.reg.sources)) as pbar, \
                 Timer(f'_create_napari_shapes: build {len(self.reg.sources)} source shape geometries',
                      verbose=self._timing_verbose()):
                # stack properties, not sims: shapes need geometry only, so this reads, creates
                # and allocates no image data at all (see build_source_stack_props)
                msims = []
                for source, translation, transform in zip(self.reg.sources, self.reg.positions,
                                                          self.reg._msim_transforms):
                    msims.append(
                        build_source_stack_props(source, self.reg._msim_output_order, translation, transform,
                                                 transform_key, z_scale=self.reg._msim_z_scale,
                                                 promote_z=promote_z))
                    if pbar is not None:
                        pbar.update(1)
        else:
            with phase(3 / 5, total=1) as pbar, \
                 Timer('_create_napari_shapes: get view_msims', verbose=self._timing_verbose()):
                msims = self.view_msims
                if pbar is not None:
                    pbar.update(1)

        with phase(1 / 5, total=1) as pbar, \
             Timer(f'_create_napari_shapes: create_image_shapes ({len(msims)} images)', verbose=self._timing_verbose()):
            shapes = create_image_shapes(msims, transform_key=transform_key, force_2d=force_2d)
            if pbar is not None:
                pbar.update(1)
        refs = [str(index) for index in range(len(msims))]
        labels = list(self.reg.file_labels)
        face_colors = [(1, 1, 1) for _ in range(len(msims))]

        # once pairwise registration has run, restrict overlap boxes to pairs actually
        # registered: an intersection appearing only after alignment, between images never
        # paired, has no quality behind it. Before registration there is no graph, so fall back
        # to every geometrically-overlapping pair.
        overlap_pairs = list(self.reg.pairs_graph.edges()) if self.reg.is_pairs_registered() else None
        with phase(1 / 5, total=1) as pbar, \
             Timer(f'_create_napari_shapes: create_overlap_shapes ({len(msims)} images,'
                  f' {len(overlap_pairs) if overlap_pairs is not None else "all"} pairs)', verbose=self._timing_verbose()):
            shapes2, pairs = create_overlap_shapes(msims, transform_key=transform_key, pairs=overlap_pairs,
                                                   force_2d=force_2d)
            if pbar is not None:
                pbar.update(1)
        shapes.extend(shapes2)
        refs += [f'{index1} {index2}' for index1, index2 in pairs]
        labels += ['' for _ in pairs]
        face_colors += [np.array(metric_to_rgb(self.reg.get_metrics(default_quality_key, pair))) for pair in pairs]
        return shapes, refs, labels, face_colors

    @staticmethod
    def _progress_phase(progress_factory, total=None, desc=None, weight=1):
        """One reporting phase of the caller's operation, or nothing to report into."""
        return (progress_factory(total=total, desc=desc, weight=weight)
                if progress_factory is not None else nullcontext(None))

    def _create_napari_data(self, transform_key, fusion_method='additive', show_preprocessed=False,
                            composite=False, progress_factory=None, weight=1):
        # `weight` is what the caller's bar allows this step, divided below between the sub-steps
        # in rough proportion to what each costs on a large project
        def phase(share, total=None):
            return self._progress_phase(progress_factory, total=total,
                                        weight=max(weight * share, 1))

        # copy_transforms_to_msims() below is the only step here that writes into a msim,
        # and before registration there is no registered transform for it to write. The
        # steps between hand back the objects they were given whenever they have nothing
        # to change, so anything added here that mutates one needs this flag too.
        writes_transforms = not (show_preprocessed
                                 and transform_key == self.reg.source_transform_key)
        if show_preprocessed:
            with phase(1 / 12, total=1) as pbar, \
                 Timer(f'_create_napari_data: copy {len(self.reg.register_msims)} register_msims',
                      verbose=self._timing_verbose()):
                msims = ([msim.copy(deep=True) for msim in self.reg.register_msims]
                         if writes_transforms else list(self.reg.register_msims))
                if pbar is not None:
                    pbar.update(1)
            # promoted here so the size estimate below sees the geometry fuse() will: it
            # promotes internally for sources at several z heights, and calc_output_properties
            # cannot combine un-promoted sims that disagree about z. Not extra work - fuse()
            # leaves an already-promoted msim alone.
            if len(set(position.get('z', 0) for position in self.reg.positions)) > 1:
                with phase(1 / 12, total=1) as pbar, \
                     Timer('_create_napari_data: promote register_msims to 3D',
                          verbose=self._timing_verbose()):
                    msims = make_msims_3d(msims, positions=self.reg.positions)
                    if pbar is not None:
                        pbar.update(1)
        else:
            # view_msims is never scale-reduced - every source's full native pyramid. Fusing
            # that at scale0 to draw an overview builds (and, for get_contrast_limits(), runs)
            # the graph for the largest levels of the combined output, when only its coarsest
            # pixels are shown until the user zooms in. Reduce to the same coarse sub-pyramid
            # create_preview() uses for its own exported preview.
            preview_scale = self.params['input_output'].get('preview_scale', default_interactive_preview_scale)
            with phase(1 / 12, total=1) as pbar, \
                 Timer('_create_napari_data: build view_msims', verbose=self._timing_verbose()):
                view_msims = self.view_msims
                if pbar is not None:
                    pbar.update(1)
            with phase(1 / 12, total=1) as pbar, \
                 Timer(f'_create_napari_data: select_msim_subpyramid_at_scale ({len(view_msims)} images)',
                      verbose=self._timing_verbose()):
                msims = select_msim_subpyramid_at_scale(view_msims, self.reg.sources, preview_scale)
                if pbar is not None:
                    pbar.update(1)
        # Whichever branch produced them, cap what this preview will fuse. The show_preprocessed
        # branch takes its resolution from pre_processing's scale, never preview_scale, so at
        # scale 1 the "preview" is the whole dataset: one run fused 396.9GB over 55 minutes to
        # draw what an 8x-reduced one drew in 9. preview_scale could not have prevented it
        # either, picking a level per source rather than bounding the combined result -
        # and select_msim_subpyramid_at_scale() cannot be used here for the same reason: its
        # level is relative to each source's own pyramid, and these are already scale-reduced.
        with phase(1 / 12, total=1) as pbar, \
             Timer('_create_napari_data: cap preview fusion size', verbose=self._timing_verbose()):
            msims = reduce_msims_to_fused_size(
                msims, transform_key, z_scale=self.reg._msim_z_scale,
                label=f'Preview fusion ({len(msims)} images)')
            if pbar is not None:
                pbar.update(1)
        with phase(1 / 12, total=1) as pbar, \
             Timer('_create_napari_data: copy_transforms_to_msims', verbose=self._timing_verbose()):
            # before registration there is no registered transform to copy, and reading
            # self.reg.msims for it forces the full build get_best_transform_key() just avoided
            if writes_transforms:
                transform_msims = self.reg.msims
                if show_preprocessed and self.reg.register_indices is not None:
                    # pre-processing may have dropped sources (filter_foreground) - pair each
                    # target with the source it came from, not with whatever sits beside it
                    transform_msims = [self.reg.msims[index] for index in self.reg.register_indices]
                copy_transforms_to_msims(transform_msims, msims, transform_key)
            if pbar is not None:
                pbar.update(1)
        if composite:
            # the main view only needs to show where the sources sit, and fusing for that costs
            # per source however small the preview (10.3 minutes for 4733) - paste them instead,
            # falling back to fusing if they cannot be placed faithfully. Also the one sub-step
            # with real per-source progress, so it gets most of what this step was allowed.
            with phase(7 / 12, total=len(msims)) as pbar, \
                 Timer(f'_create_napari_data: composite overview ({len(msims)} images)',
                       verbose=self._timing_verbose()):
                overview = composite_msims_overview(
                    msims, transform_key, z_scale=self.reg._msim_z_scale,
                    label=f'Overview ({len(msims)} images)',
                    progress=(pbar.update if pbar is not None else None))
            if overview is not None:
                return overview
        # output_chunksize is left to fuse(), which derives it after its own make_msims_3d
        # promotion - sizing it here would mean reproducing that rule against msims that may not
        # have a 'z' dim yet. No declared step count either: fuse() plans the whole graph in one
        # call, so this moves by a share of its remaining slice at each boundary it does have.
        with phase(7 / 12) as pbar, \
             Timer(f'_create_napari_data: fuse ({len(msims)} images)', verbose=self._timing_verbose()):
            if pbar is not None:
                pbar.update(1)
            fused_msim, _ = self.reg.fuse(msims,
                                          transform_key=transform_key,
                                          fusion_method=fusion_method,
                                          dimension=self.params['input_output']['registration_dimension'],
                                          extra_metadata=self.extra_metadata)
        return fused_msim

    def _update_view_add_shapes(self, viewer, shapes, refs, labels, face_colors, layer_name,
                                show_labels=True):
        # is_3d/is_multi_z_shapes read cheap per-source metadata (self.reg.sources/positions)
        # rather than self.view_msims, so drawing shapes never forces the expensive msim build
        # this is otherwise deferred to the fused image preview / pre-processing
        bb_supported = True
        if isinstance(viewer, ViewerWidget):
            viewer = viewer._qtwidget._viewer_model
            bb_supported = False
        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        is_multi_z_shapes = (len(set(position.get('z', 0) for position in self.reg.positions)) > 1)
        force_2d = not bb_supported or (is_multi_z_shapes and not is_3d)
        # a shape actually carries a 'z' coordinate whenever the source is natively a z-stack
        # (is_3d) or build_source_shape_sim()/make_msims_3d() promoted it to one (is_multi_z_shapes)
        do_3d = ((is_3d or is_multi_z_shapes) and not force_2d)

        if len(shapes) > 0:
            # Depth-tested 'translucent' made overlap boxes lose to the opaque fused image
            # volume they sit inside (worse, not better) - translucent_no_depth avoids that
            # for both 2D and 3D, so shapes always draw regardless of what else is there.
            blending = 'translucent_no_depth'
            edge_color = 'cyan'
            if do_3d:
                # a 'polygon' renders one flat face, not a non-planar box, and napari-bbox
                # 0.1.1 (which drew real 3D boxes) is incompatible with current napari - so each
                # box is 6 flat quad faces plus one edge-only 'path' wireframe, in one Shapes
                # layer. Corner order is _minimal_bb_vertices': 0-3 one face, 4-7 the opposite
                # in the same winding, so i and i+4 are the vertical edges between them.
                box_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                            [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
                edge_path = [0, 1, 2, 3, 0, 4, 7, 3, 2, 6, 7, 4, 5, 6, 2, 1, 5]
                # Matches _minimal_bb_vertices' own corner_bits convention (bottom face 0-3,
                # top face 4-7, same x/y winding), applied here to plain axis-aligned min/max
                # instead of an oriented frame.
                corner_bits = np.array([
                    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                ])

                face_shapes, face_only_colors, face_refs, face_labels = [], [], [], []
                for shape, ref, color in zip(shapes, refs, face_colors):
                    corners = np.asarray(shape)
                    # napari renders a 3D polygon's face fill only where the face's plane is
                    # axis-orthogonal (napari/napari#6860), so a rotated box gets almost none.
                    # Fill from its axis-aligned bounding box instead; the wireframe below keeps
                    # the true oriented corners, which render fine as edges at any angle.
                    mins, maxs = corners.min(axis=0), corners.max(axis=0)
                    aa_corners = mins + corner_bits * (maxs - mins)
                    centroid = aa_corners.mean(axis=0)
                    for face in box_faces:
                        quad = aa_corners[face]
                        # box_faces' index order does not wind consistently outward, even for
                        # an axis-aligned box, so flip any face whose normal points inward or
                        # napari backface-culls about half of them
                        normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
                        if np.dot(normal, quad.mean(axis=0) - centroid) < 0:
                            quad = quad[::-1]
                        face_shapes.append(quad)
                    face_only_colors += [color] * len(box_faces)
                    face_refs += [ref] * len(box_faces)
                    face_labels += [''] * len(box_faces)
                wire_shapes = [np.asarray(shape)[edge_path] for shape in shapes]

                # Napari renders 3D paths as tubes whose width is measured in world
                # coordinates, not screen pixels. Scale the tube radius to the scene so it
                # remains visible for large physical units.
                vertices = np.concatenate([np.asarray(shape) for shape in shapes])
                wire_width = np.ptp(vertices, axis=0).max() * 0.005

                shape_data = face_shapes + wire_shapes
                shape_type = ['polygon'] * len(face_shapes) + ['path'] * len(wire_shapes)
                # every face_color/edge_color entry must be the same length or napari silently
                # falls back to white for the whole layer. A 'path' has no face to color, so
                # this is a length-matched placeholder, not a meaningful value.
                face_color = face_only_colors + [(0, 0, 0)] * len(wire_shapes)
                edge_color = [(0, 0, 0)] * len(face_shapes) + [(0, 1, 1)] * len(wire_shapes)
                edge_width = [0] * len(face_shapes) + [wire_width] * len(wire_shapes)
                refs = face_refs + refs
                labels = face_labels + labels
            else:
                shape_data = np.asarray(shapes)
                shape_type = 'polygon'
                edge_width = 0.1
                face_color = face_colors

            # half napari's default (12), which reads oversized against these shapes
            # the overview draws the same shapes much smaller, where a label per shape is
            # unreadable and covers the layout it is there to show - so it takes them untexted
            text = {'string': '{labels}', 'size': 6} if show_labels else None
            # 'labels' is only ever read by that text string - with no text it is a per-shape
            # column napari carries for nothing. 'refs' stays either way as the shapes' identity
            # in the layer (the click handler below closes over its own copy of it)
            features = {'refs': refs, 'labels': labels} if show_labels else {'refs': refs}
            layer = viewer.add_shapes(shape_data, name=layer_name, shape_type=shape_type, text=text,
                                      features=features, face_color=face_color, opacity=0.5,
                                      edge_width=edge_width, edge_color=edge_color, blending=blending)

            @layer.mouse_drag_callbacks.append
            def on_shape_click(clicked_layer, event, refs=refs):
                if event.button == 1:
                    value = clicked_layer.get_value(event.position, view_direction=event.view_direction,
                                                    dims_displayed=event.dims_displayed, world=True)
                    shape_index = value[0] if value is not None else None
                    if shape_index is not None:
                        self.select_pair_preview(refs[shape_index])

    def _napari_view_add_fused_data(self, viewer, fused, layer_name, cheap=False):
        # fuse() always returns msims, and get_msim_level_data (each level's raw dask array off
        # its own Dataset) is enough to show a genuine multiscale pyramid, so nothing here needs
        # extract_sims_from_fused. `fused` is either one multiscale msim - already
        # channel-combined by fuse(), so a 'c' dim just needs channel_axis - or, in 'compose'
        # mode, a plain list of per-source msims shown as separate layers.
        #
        # cheap=True (update_views()'s overview, not the fusion tab's real preview) swaps
        # get_contrast_limits()'s per-source dask.compute() for a naive dtype-range guess: exact
        # limits don't matter for a first look, and it is the one per-source cost here that is
        # neither Qt-only nor already cheap metadata.
        channels = self.extra_metadata.get('channels', [])

        if isinstance(fused, list):
            # 'compose' mode: one napari layer per source, no real fusion. add_image() must stay
            # on the GUI thread (Qt layers aren't thread-safe), so what a pool can take off it is
            # gathering each source's metadata/contrast-limits first, joined in original order.
            def prep_layer(msim, channel):
                image0 = get_msim_image0(msim)
                scale = si_utils.get_spacing_from_sim(image0, asarray=True)
                translate = si_utils.get_origin_from_sim(image0, asarray=True)
                contrast_limits = get_contrast_limits(msim, cheap=cheap)
                return dict(data=get_msim_level_data(msim), name=channel.get('label', layer_name),
                           multiscale=True, colormap=channel.get('color', (1, 1, 1, 1)),
                           contrast_limits=contrast_limits,
                           scale=scale, translate=translate, blending='additive')

            channel_list = channels or [{}] * len(fused)
            with Timer(f'_napari_view_add_fused_data: prep {len(fused)} layers',
                      verbose=self._timing_verbose()):
                layer_kwargs = [None] * len(fused)
                if len(fused) > 1:
                    max_workers = min(default_preview_workers, len(fused))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {executor.submit(prep_layer, msim, channel): index
                                  for index, (msim, channel) in enumerate(zip(fused, channel_list))}
                        for future in as_completed(futures):
                            layer_kwargs[futures[future]] = future.result()
                elif fused:
                    layer_kwargs[0] = prep_layer(fused[0], channel_list[0])

            with Timer(f'_napari_view_add_fused_data: add_image loop ({len(fused)} layers)',
                      verbose=self._timing_verbose()):
                for kwargs in layer_kwargs:
                    data = kwargs.pop('data')
                    viewer.add_image(data, **kwargs)
            return

        image0 = get_msim_image0(fused)
        scale = si_utils.get_spacing_from_sim(image0, asarray=True)
        translate = si_utils.get_origin_from_sim(image0, asarray=True)
        data = get_msim_level_data(fused)
        with Timer('_napari_view_add_fused_data: get_contrast_limits', verbose=self._timing_verbose()):
            contrast_limits = get_contrast_limits(fused, cheap=cheap)
        if len(channels) > 1 and 'c' in image0.dims:
            channel_axis = image0.dims.index('c')
            name = [channel.get('label', index) for index, channel in enumerate(channels)]
            colormap = [channel.get('color', (1, 1, 1, 1)) for channel in channels]
            scale = [scale] * len(channels)
            translate = [translate] * len(channels)
            contrast_limits = [contrast_limits] * len(channels)
        else:
            channel_axis = None
            name = channels[0].get('label') if channels else None
            colormap = channels[0].get('color', (1, 1, 1, 1)) if channels else None
        with Timer('_napari_view_add_fused_data: add_image', verbose=self._timing_verbose()):
            viewer.add_image(data, name=name or layer_name, multiscale=True, channel_axis=channel_axis,
                             colormap=colormap, contrast_limits=contrast_limits,
                             scale=scale, translate=translate)

    def _napari_view_show_features(self, viewer, fixed_data2, fixed_points, moving_data2, moving_points, matches, inliers):
        layers = draw_keypoints_matches_napari(fixed_data2, fixed_points,
                                               moving_data2, moving_points,
                                               matches, inliers, points_color='blue')
        viewer.layers.clear()
        for data, kwargs, layer_type in layers:
            if layer_type == "image":
                viewer.add_image(data, **kwargs)
            elif layer_type == "points":
                viewer.add_points(data, **kwargs)
            elif layer_type == "shapes":
                viewer.add_shapes(data, **kwargs)

    def _napari_view_add_image(self, viewer, data, label, transform=None, color=None, affine_event=False):
        if isinstance(data, DataTree):
            # a real multiscale msim (e.g. self.reg.register_msims) - napari's affine (used here
            # for interactive per-pair drag adjustment) and multiscale lazy-loading work together
            image0 = get_msim_image0(data)
            scale = si_utils.get_spacing_from_sim(image0, asarray=True)
            position = si_utils.get_origin_from_sim(image0, asarray=True)
            layer = viewer.add_image(get_msim_level_data(data), name=label, multiscale=True,
                                     scale=scale, translate=position, affine=transform,
                                     blending='additive')
        else:
            scale = si_utils.get_spacing_from_sim(data, asarray=True)
            position = si_utils.get_origin_from_sim(data, asarray=True)
            layer = viewer.add_image(data, name=label, scale=scale, translate=position, affine=transform,
                                     blending='additive')
        if color:
            layer.colormap = color

        if affine_event:
            layer.events.affine.connect(self.on_image_data_changed)

        return layer

    def on_image_data_changed(self, event):
        self.pair_metrics_timer.stop()
        self.pair_metrics_timer.start()

    def update_pair_metrics(self):
        # filter only selected pair
        reg_msims = [self.reg.register_msims[index] for index in self.pair_indices]
        transforms = {(0, 1): self.calc_mod_pair_transform()}
        metrics = calc_msims_metrics(reg_msims, transforms, metric_methods=self.metrics_methods)
        self._remap_local_pair_metrics(metrics, self.pair_indices)
        self.populate_metrics_table(metrics)

    @staticmethod
    def _remap_local_pair_metrics(metrics, indices):
        """Remap metrics['pairs']' local (0, 1, ...) keys to the real, global source indices."""
        pairs = metrics.get('pairs')
        if pairs:
            metrics['pairs'] = {tuple(indices[i] for i in key): value for key, value in pairs.items()}

    @catch_run_errors
    def run_preview_registration(self, progress_factory=None):
        label1 = self.param_widgets.get('registration.reg_preview_image1').get_value()
        label2 = self.param_widgets.get('registration.reg_preview_image2').get_value()
        index1 = self.reg.file_labels.index(label1)
        index2 = self.reg.file_labels.index(label2)

        # as in run_pair_registration(): pre-processing gets its own bar, before this one
        if not self.reg.register_msims:
            if not self.run_pre_processing(progress_factory=progress_factory):
                return None

        with self._operation_progress('Preview registration', progress_factory, phases=2) as factory:
            def register_preview(worker_factory):
                # dask's own task counting reports as one more phase of this bar
                # (progress_class), rather than as a bar of its own
                with NapariDaskProgress(progress_class=worker_factory,
                                        desc='Preview registration'):
                    registration_params = self.params['registration']
                    channel = registration_params.get('channel')
                    cache = self._preview_overlap_cache
                    # the crop depends only on the source data (a new register_msims list
                    # whenever pre-processing re-runs) and the selected pair/channel, never on the
                    # method or its tuning - so parameter-only changes reuse it
                    if (cache is not None and cache['register_msims'] is self.reg.register_msims
                            and cache['index1'] == index1 and cache['index2'] == index2
                            and cache['channel'] == channel):
                        overlap1, overlap2, sims_pixel_space = cache['overlap1'], cache['overlap2'], cache['sims_pixel_space']
                    else:
                        msim1, msim2 = self.reg.register_msims[index1], self.reg.register_msims[index2]
                        overlap1, overlap2, sims_pixel_space = self.reg.select_pair_overlap(
                            msim1, msim2, params=registration_params)
                        overlap1, overlap2 = overlap1.compute(), overlap2.compute()
                        self._preview_overlap_cache = {
                            'register_msims': self.reg.register_msims,
                            'index1': index1, 'index2': index2, 'channel': channel,
                            'overlap1': overlap1, 'overlap2': overlap2, 'sims_pixel_space': sims_pixel_space,
                        }

                    transform, quality, results = self.reg.register_overlap(
                        overlap1, overlap2, sims_pixel_space, params=registration_params)

                    msim1, msim2 = self.reg.register_msims[index1], self.reg.register_msims[index2]
                    transforms = {(0, 1): transform}
                    qualities = {(0, 1): quality}
                    metrics = calc_msims_metrics((msim1, msim2), transforms, qualities, metric_methods=self.metrics_methods)
                    self._remap_local_pair_metrics(metrics, (index1, index2))
                    return metrics, results, overlap1, overlap2

            return self._run_off_thread(register_preview, factory)


    def preview_registration(self):
        self._clear_napari_view(self.viewer)
        result = self.run_preview_registration()
        if result is None:
            return
        metrics, results, overlap1, overlap2 = result

        self.populate_metrics_table(metrics)

        fixed_points = results.get('fixed_points', [])
        moving_points = results.get('moving_points', [])
        matches = results.get('matches', [])
        inliers = results.get('inliers', [])
        self._napari_view_show_features(self.viewer, overlap1, fixed_points, overlap2, moving_points, matches, inliers)
        self.view_mode = ViewMode.FEATURES

    def populate_metrics_table(self, metrics_dict):
        transform_keys = []
        metric_keys = []
        item_keys = []
        metrics = metrics_dict.get('summary')
        if metrics:
            item_keys.append('summary')
            for transform_key, transform_value in metrics.items():
                if transform_key not in transform_keys:
                    transform_keys.append(transform_key)
                for metric_key, metric_value in transform_value.items():
                    if metric_value is not None and metric_key not in metric_keys:
                        metric_keys.append(metric_key)
        pairs_metrics = metrics_dict.get('pairs')
        if pairs_metrics:
            file_rank = {file_index: rank for rank, file_index in
                        enumerate(sorted(range(len(self.reg.positions)),
                                         key=lambda i: position_sort_key(self.reg.positions[i])))}
            pairs_metrics = dict(sorted(pairs_metrics.items(),
                                        key=lambda item: (file_rank[item[0][0]], file_rank[item[0][1]])))
        metrics = pairs_metrics
        if metrics:
            for pair_key_indices, pair_value in metrics.items():
                pair_key = self.reg.file_labels[pair_key_indices[0]] + ' - ' + self.reg.file_labels[pair_key_indices[1]]
                if pair_key not in item_keys:
                    item_keys.append(pair_key)
                for transform_key, transform_value in pair_value.items():
                    if transform_key not in transform_keys:
                        transform_keys.append(transform_key)
                    for metric_key, metric_value in transform_value.items():
                        if metric_value is not None and metric_key not in metric_keys:
                            metric_keys.append(metric_key)

        transform_keys = [transform_key.split('_')[0] for transform_key in transform_keys]
        is_metric_cols = (len(transform_keys) <= 1 and len(metric_keys) >= 1)
        col_headers = metric_keys if is_metric_cols else transform_keys

        metrics_table = []
        for rowi in range(len(item_keys)):
            row = [None] * len(col_headers)
            metrics_table.append(row)
        item_offset = 0

        metrics = metrics_dict.get('summary')
        if metrics:
            item_offset = 1
            for transform_index, transform_value in enumerate(metrics.values()):
                for metric_index, metric_value in enumerate(transform_value.values()):
                    if metric_value is not None:
                        col_index = metric_index if is_metric_cols else transform_index
                        metrics_table[0][col_index] = metric_value
        metrics = pairs_metrics
        if metrics:
            for pair_index, pair_value in enumerate(metrics.values()):
                for transform_index, transform_value in enumerate(pair_value.values()):
                    for metric_index, metric_value in enumerate(transform_value.values()):
                        if metric_value is not None:
                            col_index = metric_index if is_metric_cols else transform_index
                            metrics_table[pair_index + item_offset][col_index] = metric_value

        table_widget = self.param_widgets.get('registration.metrics_table')
        # Table: tuple-of-values : ([values], [row_headers], [column_headers])
        table_widget.set_value((metrics_table, item_keys, col_headers))
        table_widget.set_table_column_resize_mode()
        for rowi in range(len(item_keys)):
            for coli in range(len(col_headers)):
                table_cell = table_widget.get_native_item(rowi, coli)
                if table_cell is not None:
                    table_cell.setBackground(
                        QColor(*metric_to_rgb(metrics_table[rowi][coli], max_light=0.5, output_range=255)))

    def update_registered(self, view_transform_key=None, progress_factory=None):
        msims = self.reg.msims
        coord_systems = get_transforms(msims)
        self.populate_coordinate_systems(coord_systems)
        self.populate_metadata_table(msims)
        self.populate_metrics_table(self.reg.metrics)
        self.update_views(transform_key=view_transform_key, progress_factory=progress_factory)

    def enable_modify_pair_registration(self, enabled=True):
        widget = self.param_widgets.get('registration.modify_pair_registration')
        if widget:
            widget.widget.enabled = enabled

    @catch_run_errors
    def run_pair_registration(self, progress_factory=None):
        # pre-processing, when this is what triggers it, reports its own bar and finishes before
        # the registration's starts - rather than taking a share of the registration's bar
        if not self.reg.register_msims:
            if not self.run_pre_processing(progress_factory=progress_factory):
                return None

        with self._operation_progress('Pair registration', progress_factory, phases=2) as factory:
            def register_pairs(worker_factory):
                # the progress patches belong to the thread doing the work, reporting to its own
                # factory - see _run_off_thread()
                with NapariMVSProgress(tqdm_class=worker_factory.tqdm_class, patch_registration=True), \
                        NapariDaskProgress(progress_class=worker_factory, desc='Pair registration'), \
                        Timer('pair registration', verbose=self._timing_verbose()):
                    return self.reg.register_pairs(
                        self.reg.register_msims,
                        params=self.params['registration'] | {'metrics': self.metrics_methods})

            results = self._run_off_thread(register_pairs, factory)

        qualities = {key: metric[default_transform_key][default_quality_key]
                     for key, metric in results['metrics']['pairs'].items()
                     if default_quality_key in metric[default_transform_key]}
        bboxes = {}
        for key, value in nx.get_edge_attributes(self.reg.pairs_graph, 'bbox').items():
            if 't' in value.dims:
                value = value.sel(t=0)
            bboxes[key] = np.array(value).tolist()
        self.reg.save_pair_mappings(results['pair_mappings'], qualities, bboxes)
        self.enable_modify_pair_registration()
        return results

    @catch_run_errors
    def run_global_registration(self, progress_factory=None):
        # register_global() reports its own stage boundaries (progress_factory): the optimisation,
        # applying the transforms, the summary plots, storing them, and the metrics. Declaring how
        # many there are is what keeps the last of them from taking most of the bar each; the
        # optimisation claims 4 of these 8 units, being most of the run (66 of 86 minutes once)
        with self._operation_progress('Global registration', progress_factory, phases=8) as factory:
            def register_global(worker_factory):
                with NapariDaskProgress(progress_class=worker_factory, desc='Global registration'), \
                        Timer('global registration', verbose=self._timing_verbose()):
                    return self.reg.register_global(self.reg.pair_msims,
                                                    register_indices=self.reg.register_indices,
                                                    params=self.params['registration'],
                                                    progress_factory=worker_factory)

            results = self._run_off_thread(register_global, factory)

        self.reg.save_mappings(results['mappings'])
        self.reg.save_mappings_csv(results['mappings'])
        self.reg.save_metrics(results['metrics'])
        return results

    def pair_registration(self):
        if self.reg.is_global_registered():
            message = 'Global registration was already performed. '
        elif self.reg.is_pairs_registered():
            message = 'Pair registration was already performed. '
        else:
            message = ''
        message += 'Run pair registration?'
        reply = QMessageBox.question(None, 'muvis-align', message,
                                     QMessageBox.Yes|QMessageBox.No)
        if reply == QMessageBox.Yes:
            if not self.run_pair_registration():
                return
            self.update_registered(view_transform_key=self.reg.source_transform_key)
            QMessageBox.information(None, 'muvis-align', 'Pair registration completed')

    def modify_pair_registration(self):
        if self.view_mode == ViewMode.PAIRS:
            reply = QMessageBox.question(None, 'muvis-align','Store modified registration?',
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                # update transforms back into graph
                transform = self.calc_mod_pair_transform()
                pair_transforms = nx.get_edge_attributes(self.reg.pairs_graph, default_transform_key)
                qualities = nx.get_edge_attributes(self.reg.pairs_graph, default_quality_key)
                if 't' in pair_transforms[self.pair_indices].dims:
                    transform = transform.expand_dims({'t': [0]})
                pair_transforms[self.pair_indices] = transform
                qualities[self.pair_indices] = np.array(1)    # set quality to 1
                nx.set_edge_attributes(self.reg.pairs_graph, pair_transforms, default_transform_key)
                nx.set_edge_attributes(self.reg.pairs_graph, qualities, default_quality_key)
                bboxes = {}
                for key, value in nx.get_edge_attributes(self.reg.pairs_graph, 'bbox').items():
                    if 't' in value.dims:
                        value = value.sel(t=0)
                    bboxes[key] = np.array(value).tolist()
                self.reg.save_pair_mappings(pair_transforms, qualities, bboxes)

            self.view_mode = ViewMode.OVERVIEW
            self.update_registered(view_transform_key=self.reg.source_transform_key)
            self._restore_pair_modify_state()
        else:
            self.view_mode = ViewMode.PAIRS
            labels = self.reg.file_labels
            label1 = self.param_widgets.get('registration.reg_preview_image1').get_value()
            label2 = self.param_widgets.get('registration.reg_preview_image2').get_value()
            index1 = labels.index(label1)
            index2 = labels.index(label2)
            indices = index1, index2
            colors = [(0, 1, 0), (1, 0, 1)]     # green, purple
            pair_transforms = nx.get_edge_attributes(self.reg.pairs_graph, default_transform_key)
            if indices not in pair_transforms and tuple(reversed(indices)) in pair_transforms:
                indices = tuple(reversed(indices))

            if indices not in pair_transforms:
                show_warning('No pair registration found for selected images')
            else:
                self.temp_widget_state = TemporarilyDisabledWidgets()
                self.temp_tab_states = {}
                all_widgets = self.get_all_widgets()
                all_widgets.pop('registration.modify_pair_registration', None)
                self.temp_widget_state.disable(all_widgets)
                if self.enable_tab and self.is_tab_enabled:
                    other_section_ids = [section_id for section_id in ['project'] + list(self.template.keys())
                                        if section_id != 'registration']
                    self.temp_tab_states = {section_id: self.is_tab_enabled(section_id)
                                            for section_id in other_section_ids}
                    for section_id in other_section_ids:
                        self.enable_tab(section_id, False)
                self.pair_indices = indices
                pair_transform = np.array(pair_transforms[indices].sel(t=0))
                eye = np.eye(max(pair_transform.shape))
                pair_transforms = pair_transform, eye

                # everything below can bail out (pre-processing failing) or raise, and the
                # widgets/tabs disabled just above are only ever re-enabled by leaving this mode
                # - without restoring here a failure leaves the whole plugin permanently dead
                entered = False
                try:
                    if not self.reg.register_msims:
                        if not self.run_pre_processing():
                            return
                    self._clear_napari_view(self.viewer)
                    # register_msims is a real multiscale pyramid (built by preprocess()) - lets
                    # napari lazily load whichever level it needs during interactive adjustment
                    register_images = self.reg.register_msims
                    for index, (sim_index, color) in enumerate(zip(indices, colors)):
                        self._napari_view_add_image(self.viewer, register_images[sim_index], labels[sim_index],
                                                    pair_transforms[index], color, affine_event=True)
                    self.update_pair_metrics()
                    entered = True
                finally:
                    if not entered:
                        self.view_mode = ViewMode.OVERVIEW
                        self._restore_pair_modify_state()

    def _restore_pair_modify_state(self):
        """Re-enable whatever entering pair-modify mode disabled."""
        self.temp_widget_state.restore()
        if self.enable_tab:
            for section_id, was_enabled in self.temp_tab_states.items():
                self.enable_tab(section_id, was_enabled)

    def calc_mod_pair_transform(self):
        transforms = [layer.affine.affine_matrix for layer in self.viewer.layers]
        matsize = len(si_utils.get_spatial_dims_from_sim(get_msim_image0(self.reg.msims[0]))) + 1
        transform = calculate_rigid_difference(transforms[1][-matsize:, -matsize:],
                                               transforms[0][-matsize:, -matsize:])
        return param_utils.affine_to_xaffine(transform)

    def registration_process(self):
        if 'convert' in self.params['registration']['operation']:
            # convert: each source written out individually at its own source/metadata
            # position, no registration and no fusion/blending - never reaches the fusion tab
            reply = QMessageBox.question(None, 'muvis-align', 'Convert data to OME-Zarr only?',
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                if self.run_convert():
                    QMessageBox.information(None, 'muvis-align', 'Conversion completed')
            return

        if 'merge' in self.params['registration']['operation']:
            # merge: fuse at the sources' own metadata positions, registering nothing. There is
            # no registration step to run here, so this only opens the fusion tab - the fusion
            # itself falls back to source_transform_key via get_best_transform_key()
            reply = QMessageBox.question(None, 'muvis-align',
                                         'Merge without registration, at the source positions?',
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.enable_tabs(True, 4)
                self.select_tab(4)
            return

        completion_message = 'Global registration completed'
        if self.reg.is_global_registered():
            message = 'Global registration was already performed. Run global registration?'
        elif not self.reg.is_pairs_registered():
            message = 'Pair registration not performed yet. Run both pair and global registration?'
            completion_message = 'Registration completed'
        else:
            message = 'Run global registration?'
        reply = QMessageBox.question(None, 'muvis-align', message,
                                     QMessageBox.Yes|QMessageBox.No)
        if reply == QMessageBox.Yes:
            # a bar per registration - pair, then global - each finishing before the next
            # starts. One bar over both would have the pair phases fill it, leaving the global
            # registration (just as long, and the only thing still running) with nothing to show
            with Timer('registration process', verbose=self._timing_verbose()):
                if not self.reg.is_pairs_registered():
                    if not self.run_pair_registration():
                        return
                if not self.run_global_registration():
                    return
            # ...and one more for building and drawing the view they leave on screen
            with self._operation_progress('Refreshing view', phases=2) as view_factory:
                self._copy_transforms_to_view_msims(self.reg.reg_transform_key,
                                                    progress_factory=view_factory)
                self.update_registered(view_transform_key=self.reg.reg_transform_key,
                                       progress_factory=view_factory)
            self.enable_tabs(True, 4)
            QMessageBox.information(None, 'muvis-align', completion_message)

    @catch_run_errors
    def preview_fusion(self, progress_factory=None):
        # not reg_transform_key directly: a merge fuses without ever registering, so there is no
        # 'registered' transform to fuse by - get_best_transform_key() falls back to the sources'
        # own metadata positions, and still returns the registered one wherever it exists
        transform_key = self.get_best_transform_key()
        # as in update_views(): building the data is the long half, so it is weighted as such and
        # reports from the inside rather than being one block with a tick on either side
        fusion_weight = 12
        with self._operation_progress('Fusion preview', progress_factory,
                                      phases=fusion_weight + 1) as factory:
            def fuse(worker_factory):
                with NapariMVSProgress(tqdm_class=worker_factory.tqdm_class, desc='Fusion',
                                       patch_fusion=True):
                    return self._create_napari_data(
                        transform_key, fusion_method=self.params['fusion']['method'],
                        progress_factory=worker_factory, weight=fusion_weight)

            data = self._run_off_thread(fuse, factory)
            with factory(total=1) as pbar:
                self._clear_napari_view(self.viewer)
                self._napari_view_add_fused_data(self.viewer, data, f'{self.reg.fileset_label} data')
                self.view_mode = ViewMode.FUSED
                pbar.update(1)
        # preview_fusion() is also what a resumed already-fused project draws (init_progress's
        # is_fused() branch) - unlike every other init_progress branch, it never otherwise goes
        # through update_views(), so the overview would stay empty without this
        self._refresh_overview_shapes(transform_key)

    @catch_run_errors
    def run_convert(self):
        operation = self.params['registration']['operation']
        output_folder = operation_to_past_participle(operation)
        ome_version = self.params['fusion']['ome_version']
        # source build and conversion are two phases of one operation, so they share one bar
        with self._operation_progress('Converting', phases=2) as progress_factory, \
             Timer('convert', verbose=self._timing_verbose()):
            # see run_pre_processing() - builds msims with its own progress reporting instead of
            # silently as a side effect of the save loop below
            msims = self._run_off_thread(
                lambda worker_factory: self.reg.ensure_msims(progress_factory=worker_factory),
                progress_factory)
            # not MVSRegistration.fuse(): even its 'compose' method builds one shared output
            # canvas across every source first, and is_channel_overlay would still combine them
            # into one multichannel image wherever several channels are configured. Each source
            # is written out on its own instead, keeping its native pyramid levels exactly
            # (save_native_levels()), never fused or resampled.
            #
            # Sequential, not a thread pool: zarr's async store internals aren't safe to invoke
            # concurrently from threads each running their own event loop (a Windows
            # PermissionError racing on zarr.json), and each write already parallelises its own
            # computation across every core. The per-source count is meaningful here, unlike
            # NapariDaskProgress's per-task one, which resets every iteration.
            with progress_factory(total=len(msims), desc='Converting') as pbar:
                # file_labels are already disambiguated; get_filetitle() alone is not, so two
                # sources differing only by parent directory would overwrite each other's output
                for label, position, msim in zip(self.reg.file_labels, self.reg.positions, msims):
                    output_filename = f'{output_folder}/{label}'
                    self.reg.save_native_levels(output_filename, msim, position=position,
                                                ome_version=ome_version)
                    pbar.update(1)
        return True

    @catch_run_errors
    def run_fusion(self, progress_factory=None):
        operation = self.params['registration']['operation']
        output_filename = operation_to_past_participle(operation)
        # empty means 'size it automatically': fuse() then blocks the export against what one
        # block costs in memory, rather than against a number picked for the on-disk layout. The
        # saves below are on-disk layout only, with no per-block cost, so they keep tiling.
        tile_size = self.params['fusion'].get('tile_size')
        if isinstance(tile_size, str):
            tile_size = tile_size.strip()
            if not tile_size:
                tile_size = None
            elif ',' in tile_size:
                tile_size = [int(size.strip()) for size in tile_size.split(',')]
            else:
                tile_size = int(tile_size)
        save_tile_size = tile_size or default_chunk_size

        with self._operation_progress('Fusion', progress_factory, phases=2) as factory:
            def fuse(worker_factory):
                # both the fusion and the write run on the worker thread, reporting to its own
                # factory - see _run_off_thread()
                with NapariMVSProgress(tqdm_class=worker_factory.tqdm_class, desc='Fusion',
                                       patch_fusion=True), \
                        Timer('fusion', verbose=self._timing_verbose()):
                    fused_image, is_saved = self.reg.fuse(
                        self.reg.msims,
                        fusion_method=self.params['fusion']['method'],
                        # fuse() defaults this to reg_transform_key, which a merge never
                        # writes - see preview_fusion()
                        transform_key=self.get_best_transform_key(),
                        output_spacing=self.params['fusion']['spacing'],
                        dimension=self.params['input_output']['registration_dimension'],
                        output_filename=output_filename,
                        tile_size=tile_size,
                        ome_version=self.params['fusion']['ome_version'],
                        extra_metadata=self.extra_metadata)
                    if not is_saved:
                        # save() only accepts a single-resolution sim - fused_image is always the
                        # whole multiscale pyramid now, so save its finest scale
                        save_sim = extract_sims_from_fused(fused_image)
                        self.reg.save(output_filename, save_sim,
                                      transform_key=self.get_best_transform_key(),
                                      translations0=self.reg.positions,
                                      channels=self.extra_metadata.get('channels', []),
                                      tile_size=save_tile_size,
                                      ome_version=self.params['fusion']['ome_version'])
                    return fused_image

            return self._run_off_thread(fuse, factory)


    def fusion_process(self):
        message = 'Fusion was already performed. ' if self.reg.is_fused() else ''
        message += 'Export fused data?'
        reply = QMessageBox.question(None, 'muvis-align', message,
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            fused_image = self.run_fusion()
            if fused_image is None:
                return
            # the export reported its own bar; drawing the result reports a second one
            with self._operation_progress('Refreshing view') as factory, \
                 factory(total=1):
                self._clear_napari_view(self.viewer)
                self._napari_view_add_fused_data(self.viewer, fused_image, 'Fused')
            self.reg.state = RegState.FUSED
            self.view_mode = ViewMode.FUSED
            QMessageBox.information(None, 'muvis-align', 'Fusion completed')
