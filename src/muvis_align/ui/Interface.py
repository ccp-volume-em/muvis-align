from contextlib import contextmanager
import logging
from enum import Enum, auto
from magicclass.ext.napari import ViewerWidget
from multiview_stitcher import msi_utils, ngff_utils, spatial_image_utils as si_utils, param_utils
from napari.qt.threading import create_worker
from napari.utils import progress
from napari.utils.notifications import notification_manager, show_info, show_warning
import networkx as nx
import numpy as np
import os.path
import shutil
from xarray import DataTree
from qtpy.QtCore import QEventLoop, QObject, QTimer, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication, QMessageBox

from muvis_align.constants import zarr_extension, tiff_extension, default_transform_key, default_quality_key, \
    default_interactive_preview_scale, default_chunk_size
from muvis_align.file.project_yaml import (read_params, get_template_params, get_unknown_params, write_params,
                                             update_params)
from muvis_align.MVSRegistration import MVSRegistration, RegState
from muvis_align.image.util import get_sim_physical_size, get_sim_position_final, \
    create_image_shapes, create_overlap_shapes, build_source_stack_props, \
    draw_keypoints_matches_napari, grid_point_pairs, get_transforms, copy_transforms_to_msims, NoOverlapError, \
    promoted_geometry, metric_to_rgb, get_msim_level_data, get_contrast_limits, \
    get_msim_image0, wrap_sims_as_msims, extract_sims_from_fused, extract_sims_from_msims, \
    snapshot_msims_transform, restore_msims_transform, \
    select_msim_subpyramid_at_scale, reduce_msims_to_fused_size
from muvis_align.file.resources import get_project_template
from muvis_align.logging import init_logging
from muvis_align.metrics import calc_msims_metrics
from muvis_align.Timer import Timer
from muvis_align.ui.NapariDaskProgress import NapariDaskProgress
from muvis_align.ui.MagicColorPicker import MagicColorPicker
from muvis_align.ui.NapariMVSProgress import NapariMVSProgress
from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress, SilentProgress
from muvis_align.ui.ParamWidget import create_dict_of_lists, update_dict_value
from muvis_align.image.lazy_overview import lazy_section_overview, MsimLevels
from muvis_align.ui._utils import TemporarilyDisabledWidgets, VisibleActivityDock, catch_run_errors, flush_paint_events, \
    clear_napari_dask_cache, operation_label, patch_multiscale_label_show, patch_shapes_text_coords, report_failure
from muvis_align.ui.bilayers_util import get_section_dict
from muvis_align.util import OperationCancelled, cancellable, request_cancel, print_dict_simple, set_dict_value, is_valid_value, \
    calculate_rigid_difference, operation_to_past_participle, eval_path, path_param_to_text, \
    resolve_to_project_dir, relativize_to_project_dir, parse_scale, get_filetitle, print_hbytes, progress_phase, \
    background_luminance, contrasting_color


patch_shapes_text_coords()
patch_multiscale_label_show()

# the input_output settings a saved registration depends on
source_metadata_params = ['source_position_z', 'source_position_y', 'source_position_x',
                          'source_scale_z', 'source_scale_y', 'source_scale_x',
                          'source_rotation', 'normalise_rotated_positions']


class _ProgressBridge(QObject):
    """Carries a worker thread's progress positions back to the Qt thread, where the bar lives."""

    moved = Signal(int)
    stepped = Signal(str)


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
        self._pair_metrics_worker = None
        self._pair_metrics_pending = False

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
        self.update_fusion_preview_enabled()

    def get_all_widgets(self):
        # excludes a disabled tab's widgets: their .enabled reads False from the tab, so restoring
        # that state would keep them disabled once the tab is enabled again
        return {name: param_widget.widget for name, param_widget in self.param_widgets.items()
               if self.is_tab_enabled is None or self.is_tab_enabled(name.split('.', 1)[0])}

    def get_function(self, function_label):
        if hasattr(self, function_label):
            return eval(f'self.{function_label}')
        else:
            return None

    def tab_changed(self, tab_label):
        self.current_tab = tab_label
        if tab_label != 'registration' and self.view_mode == ViewMode.FEATURES:
            self._clear_napari_view(self.viewer)
            self.view_mode = None
        self.pair_metrics_timer.stop()

    def project_path(self, path):
        self.reset()
        self.params_path = path
        self.params = get_template_params(self.template)
        if os.path.exists(path):
            project_params = read_params(path)
            unknown_params = get_unknown_params(self.template, project_params)
            if unknown_params:
                # still written back unchanged, as scripted runs may read them (e.g. extra_metadata)
                logging.warning(f'Project settings not used by the plugin: {", ".join(unknown_params)}')
            self.params = update_params(self.params, project_params)
            self.update_widgets()
        else:
            self.write_params()
        self.update_input_output_path()
        self.resolve_output_settings()

    def get_project_dir(self):
        # path params are stored relative to this directory, so a project moves with its data
        params_path = getattr(self, 'params_path', None)
        return os.path.dirname(os.path.abspath(params_path)) if params_path else None

    def update_widgets(self):
        for param_name, param_widget in self.param_widgets.items():
            # path widgets are set by update_input_output_path(), relative to the project directory
            if param_name not in ('input_output.input_path', 'input_output.output_path'):
                keys = param_name.split('.')
                value = self.params.get(keys[0], {}).get(keys[1])
                if value is not None:
                    param_widget.set_value(value)
                    if param_name == 'input_output.channels_table':
                        # set_value() blocks the table's signals, so channels_table() never sees the loaded value
                        if value:
                            self.set_channels(value)
                        # a loaded project skips update_output_channels(), so attach the color pickers here
                        self.populate_channels_table_color_pickers()

    def write_params(self):
        write_params(self.params_path, self.params)

    def get_output_dir(self):
        return resolve_to_project_dir(path_param_to_text(self.params['input_output'].get('output_path', '')),
                                      self.get_project_dir())

    def get_output_params_path(self):
        """The copy of the project file next to its output, or None for a project kept in its own output."""
        params_path = getattr(self, 'params_path', None)
        if not params_path or not os.path.exists(params_path):
            return None
        target = os.path.join(self.get_output_dir(), os.path.basename(params_path))
        return target if os.path.abspath(target) != os.path.abspath(params_path) else None

    def prepare_action(self):
        """As an action starts: settings resolved against its output, sources read again where their settings
        changed, and the project file copied next to the output. Returns whether the sources were read again."""
        self.resolve_output_settings()
        read_again = self.need_source_reinit and self.reg.is_initialised()
        if read_again:
            self._input_output_process()
        self.copy_params_to_output()
        return read_again

    def copy_params_to_output(self):
        """The project file next to its output: the settings the output was made with."""
        target = self.get_output_params_path()
        if target:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(self.params_path, target)
            logging.info(f'Project settings copied to {target}')

    def resolve_output_settings(self):
        """A registration saved with other source metadata settings either gets them back or is discarded."""
        output_params_path = self.get_output_params_path()
        if not output_params_path or not os.path.exists(output_params_path):
            return
        output = self.get_output_dir()
        operation = self.params['registration'].get('operation', '')
        fused_name = operation_to_past_participle(operation) if operation else None
        fused_paths = self._output_paths(fused_name, output) if fused_name else []
        saved = self.reg.saved_progress_paths(output) + [path for path in fused_paths if os.path.exists(path)]
        previous = read_params(output_params_path).get('input_output', {})
        current = read_params(self.params_path).get('input_output', {})
        changed = [name for name in source_metadata_params if previous.get(name) != current.get(name)]
        if saved and changed:
            if self._ask_discard_output(f'Source metadata settings differ from the registration saved in {output}:'
                                        f' {", ".join(changed)}.\nReload the previous settings?'
                                        ' Or continue and discard all previous output?'):
                self.reg.discard_saved_progress(output)
                if fused_name:
                    self._remove_output(fused_name, output)
                self.need_source_reinit = True
            else:
                for name in changed:
                    self.params['input_output'][name] = previous.get(name)
                self.write_params()
                self.update_widgets()

    def _ask_discard_output(self, text):
        # no cancel: closing the box reloads, the answer that deletes nothing
        box = QMessageBox(QMessageBox.Icon.Question, 'muvis-align', text,
                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Discard)
        box.button(QMessageBox.StandardButton.Yes).setText('Reload')
        discard = box.button(QMessageBox.StandardButton.Discard)
        discard.setText('Discard')
        box.setDefaultButton(QMessageBox.StandardButton.Yes)
        box.exec()
        return box.clickedButton() is discard

    def change_param(self, param_name, value):
        keys = param_name.split('.')
        if keys[0] not in self.params:
            self.params[keys[0]] = {}
        if isinstance(value, str):
            value = value.replace('\\', '/')
            if param_name in ('input_output.input_path', 'input_output.output_path'):
                # the file dialog reports an absolute path: store it relative, keeping the project portable
                value = relativize_to_project_dir(value, self.get_project_dir())
        self.params[keys[0]][keys[1]] = value
        self.write_params()
        if param_name == 'registration.operation':
            self.update_fusion_preview_enabled()

    def update_input_output_path(self):
        # shows paths as stored: FileEdit.set_value() would force them absolute, so the line edit is set directly
        params = self.params['input_output']
        widget = self.param_widgets.get('input_output.input_path')
        # several comma-separated globs are one path value, shown as stored
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
        self._set_source_metadata_value(['position', 'z'], value)

    def source_position_y(self, value):
        self._set_source_metadata_value(['position', 'y'], value)

    def source_position_x(self, value):
        self._set_source_metadata_value(['position', 'x'], value)

    def source_scale_z(self, value):
        self._set_source_metadata_value(['scale', 'z'], value)

    def source_scale_y(self, value):
        self._set_source_metadata_value(['scale', 'y'], value)

    def source_scale_x(self, value):
        self._set_source_metadata_value(['scale', 'x'], value)

    def source_rotation(self, value):
        self._set_source_metadata_value(['rotation'], value)

    def _set_source_metadata_value(self, keys, value):
        # an emptied field drops its override, so the source's own value applies, as for 'source'
        if is_valid_value(value):
            set_dict_value(self.source_metadata, keys, value)
        else:
            parent = self.source_metadata
            for key in keys[:-1]:
                parent = parent.get(key, {})
            parent.pop(keys[-1], None)
        self.need_source_reinit = True

    def normalise_rotated_positions(self, value):
        if value:
            self.source_metadata['norm'] = True
        else:
            self.source_metadata.pop('norm', None)
        self.need_source_reinit = True

    def registration_dimension(self, value):
        # channels depend on the registration dimension, so they are built again
        self.extra_metadata.pop('channels', None)

    def channels_table(self, value):
        old_value = self.param_widgets.get('input_output.channels_table').get_value()
        self.set_channels(update_dict_value(old_value, value))

    def set_channels(self, channels_dict):
        channels = [{'label': label} for label in channels_dict.get('label', [])]
        for channeli, channel in enumerate(channels):
            if channeli < len(channels_dict.get('color', [])):
                color = parse_channel_color(channels_dict['color'][channeli])
                if color is not None:
                    channel['color'] = color
        self.extra_metadata['channels'] = channels

    def input_output_process(self):
        try:
            if not self.prepare_action():
                self._input_output_process()
        except OperationCancelled:
            logging.info('input_output_process cancelled')
            show_info('Cancelled')

    def _input_output_process(self):
        # paths are shown normalised on Process, not live while typing
        self.update_input_output_path()
        params = self.params['input_output']
        project_dir = self.get_project_dir()
        output = resolve_to_project_dir(path_param_to_text(params['output_path']), project_dir)
        if not self.reg.is_initialised() or self.need_source_reinit:
            self.need_source_reinit = False
            # reg.init() keeps the previous registration's mappings, metrics and pre-processed msims
            self.reg.reset()
            if not output.endswith('/'):
                output += '/'
            input_path = resolve_to_project_dir(path_param_to_text(params['input_path']), project_dir)
            # one bar for reading the sources and any saved registration; the view gets its own bar after
            try:
                ok = self._init_sources(input_path, output, params)
            except OperationCancelled:
                # sources read part-way are not kept: the next Process reads them again
                self.need_source_reinit = True
                self.reg.state = RegState.UNINIT
                raise
            if ok:
                self._show_loaded_project()
                self._go_to_shown_section()
            else:
                show_warning('Invalid input or output')
                self.reg.state = RegState.UNINIT
        elif self.reg.is_global_registered():
            self.update_registered(view_transform_key=self.reg.reg_transform_key)
        elif self.reg.is_pairs_registered():
            self.update_registered(view_transform_key=self.reg.source_transform_key)
        else:
            self.update_metadata_source()

    def _init_sources(self, input_path, output, params):
        with self._operation_progress('Initialising sources', phases=1) as factory:
            ok = self.reg.init(input_path=eval_path(input_path),
                               output_path=output,
                               overwrite=params['overwrite'],
                               pairing=self.params['registration'].get('pairing', ''),
                               verbose=self.verbose)
            if ok:
                # resuming does far more than a fresh open: reserve its phases before the source read sizes
                # itself against the whole bar
                operation = self.params['registration'].get('operation', '')
                fused_name = operation_to_past_participle(operation) if operation else None
                if self.reg.has_saved_progress(fused_name, zarr_extension):
                    factory.ensure_phases(4)
                self._show_middle_section()
                # _show_loaded_project() draws the view, so drawing here would only show the unregistered transform
                ok = self.update_metadata_source(skip_view_update=True, progress_factory=factory)
                if ok:
                    self.populate_image_selection()
                    self._load_saved_progress(factory)
        return ok

    def _show_middle_section(self):
        """The middle section's sources read and drawn before the rest, so the viewer is usable while reading many
        sources takes long. The full view replaces it once they are all read."""
        indices = self.reg.middle_section_indices()
        self._shown_section_indices = indices
        if indices is None:
            return
        section = self.reg.section_registration(indices)
        # left off the bar: sized for reading every source, its phases would take a large share
        silent = SilentProgress()
        self._run_off_thread(
            lambda worker_factory: section.init_data(source_metadata=self.source_metadata,
                                                     progress_factory=worker_factory),
            silent)
        full, self.reg = self.reg, section
        try:
            self.update_views(show_images=False, progress_factory=silent)
        finally:
            self.reg = full

    def _go_to_shown_section(self):
        """Keep the section shown while the rest were read: napari's own start, the middle step of the z range, is a
        different section unless they are evenly spaced."""
        indices = getattr(self, '_shown_section_indices', None)
        dims = self.viewer.dims
        if indices and dims.ndim >= 3 and self.reg.sources[0].get_size().get('z', 0) <= 1:
            z_position = self.reg.positions[indices[0]].get('z')
            if z_position is not None:
                dims.set_point(dims.ndim - 3, z_position)

    def _run_off_thread(self, work, progress_factory):
        """Run work(progress_factory) on a worker thread and wait for it in a nested event loop, keeping the window
        responsive.

        `work` gets a headless progress factory whose positions reach the bar as a queued signal, as a worker must
        never touch a Qt widget. Patches a phase needs (dask callbacks, tqdm) belong inside `work`.
        Without a Qt application the work runs in the calling thread.
        """
        app = QApplication.instance() if QApplication is not None else None
        if app is None or progress_factory is None:
            return work(progress_factory)

        bridge = _ProgressBridge()
        bridge.moved.connect(progress_factory.set_position, Qt.QueuedConnection)
        bridge.stepped.connect(progress_factory.set_step, Qt.QueuedConnection)
        # starts where the bar has got to: a twin planning from empty would report positions the bar is past
        worker_factory = progress_factory.worker_twin(bridge.moved.emit, bridge.stepped.emit)

        def run():
            with worker_factory as factory:
                return work(factory)

        outcome = {}
        loop = QEventLoop()
        # handlers set at creation: without an 'errored' one, create_worker() adds one re-raising uncatchably
        # in the Qt event loop
        worker = create_worker(
            run,
            _start_thread=False,
            _connect={
                'returned': lambda value: outcome.__setitem__('value', value),
                'errored': lambda error: outcome.__setitem__('error', error),
                # a worker diverts warnings to this signal: shown as on the Qt thread, once per message
                'warned': lambda warning: notification_manager.receive_warning(*warning),
                'finished': loop.quit,
            },
        )
        worker.start()
        loop.exec_()
        # the twin's accounting becomes the bar's, so the next off-thread call continues from here
        progress_factory.continue_from(worker_factory)
        if 'error' in outcome:
            raise outcome['error']
        return outcome.get('value')

    @contextmanager
    def _operation_progress(self, desc, progress_factory=None, phases=1):
        """Progress for one user-facing operation: the activity dock, disabled widgets and a single
        NapariPhaseProgress bar that all its phases report into.

        An operation nested in another (its factory passed, or one already running) reports into that bar and
        leaves the dock and widgets to it. `phases` sizes each phase's slice of the bar; being out by one only
        makes it move unevenly.
        """
        factory = progress_factory or getattr(self, '_running_operation', None)
        if factory is not None:
            # the outer bar still needs this phase count, or each phase takes most of what it had left
            factory.ensure_phases(phases)
            yield factory
            return
        with cancellable(), \
             NapariPhaseProgress(progress_class=progress, desc=desc, phases=phases,
                                 min_duration=0.1) as factory, \
             self._operation_widgets(), \
             VisibleActivityDock(self.viewer):
            self._running_operation = factory
            try:
                yield factory
            finally:
                self._running_operation = None

    @contextmanager
    def _operation_widgets(self):
        """While an operation runs: every widget disabled but the Process buttons, which read Cancel, and
        every tab but the current one - as modify_pair_registration does, states restored after."""
        widgets = self.get_all_widgets() if getattr(self, 'param_widgets', None) else {}
        process_buttons = [widgets.pop(name) for name in list(widgets) if name.endswith('_process')]
        widget_state = TemporarilyDisabledWidgets()
        widget_state.disable(widgets)
        tab_states = {}
        current_tab = getattr(self, 'current_tab', None)
        enable_tab, is_tab_enabled = getattr(self, 'enable_tab', None), getattr(self, 'is_tab_enabled', None)
        if enable_tab and is_tab_enabled and current_tab is not None:
            tab_states = {section_id: is_tab_enabled(section_id)
                          for section_id in ['project'] + list(self.template.keys()) if section_id != current_tab}
            for section_id in tab_states:
                enable_tab(section_id, False)
        for button in process_buttons:
            button.text = 'Cancel'
        flush_paint_events()
        try:
            yield
        finally:
            for button in process_buttons:
                button.text = 'Process'
            widget_state.restore()
            for section_id, was_enabled in tab_states.items():
                enable_tab(section_id, was_enabled)
            flush_paint_events()

    def process_or_cancel(self, process, *_):
        """A section's Process button: runs `process`, or while an operation runs (the button then reads
        Cancel), offers to cancel that one."""
        running = getattr(self, '_running_operation', None)
        if running is None:
            # a failure outside the run_*() steps would otherwise reach only stderr, via the Qt signal
            try:
                process()
            except Exception as error:
                name = getattr(process, '__name__', 'process')
                logging.exception(f'{name} failed')
                report_failure(operation_label(name), error)
        else:
            self.cancel_operation(getattr(running, 'desc', None))

    def cancel_operation(self, desc=None):
        # the running operation stops at its next progress step, on its worker thread
        reply = QMessageBox.question(None, 'muvis-align',
                                     f'Cancel {desc or "the running operation"}? Its partial result is discarded.',
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            logging.info('Cancel requested')
            request_cancel()

    def init_progress(self):
        # the two halves of opening a project, each its own bar, one after the other
        with self._operation_progress('Initialising sources', phases=3) as progress_factory:
            self._load_saved_progress(progress_factory)
        self._show_loaded_project()

    def _load_saved_progress(self, progress_factory=None):
        output_filename = operation_to_past_participle(self.params['registration']['operation'])
        # resuming builds the per-source msims, loads the saved registration and may pre-process again
        with self._operation_progress('Initialising sources', progress_factory,
                                      phases=3) as progress_factory:
            self._run_off_thread(
                lambda worker_factory: self.reg.init_progress(output_filename, zarr_extension,
                                                              progress_factory=worker_factory),
                progress_factory)

    def _show_loaded_project(self):
        # the second bar of opening a project: building and drawing the view
        section_tab = None
        if self.reg.is_fused():
            fused_path = self.reg.current_fused_output(
                operation_to_past_participle(self.params['registration']['operation']))
            with self._operation_progress('Refreshing view', phases=2) as view_factory:
                if fused_path is not None:
                    self._show_fused_output(fused_path, progress_factory=view_factory)
                else:
                    # the export on disk predates the saved registration
                    logging.info('Fused output is older than the saved registration: previewing the fusion instead')
                    self._copy_transforms_to_view_msims(self.reg.reg_transform_key, progress_factory=view_factory)
                    self.preview_fusion(progress_factory=view_factory)
            section_tab = 4
        elif self.reg.is_global_registered():
            with self._operation_progress('Refreshing view', phases=2) as view_factory:
                self.update_registered(view_transform_key=self.reg.reg_transform_key,
                                       progress_factory=view_factory)
            section_tab = 4
        elif self.reg.is_pairs_registered():
            self.update_registered(view_transform_key=self.reg.source_transform_key)
            section_tab = 3
        else:
            # the draw update_metadata_source() skipped; before pre-processing only shapes are shown
            self.update_views(show_images=False)
        # every step can be reached from here: a later one runs whatever it needs first
        self.enable_tabs(True, 4)
        self.update_fusion_preview_enabled()
        if section_tab is not None:
            self.select_tab(section_tab)

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

            # msims are built lazily by view_msims; the cheap z remapping stays eager so the shapes are right
            z_positions = sorted(set([position.get('z', 0) for position in self.reg.positions]))
            if len(z_positions) > 1:
                for position in self.reg.positions:
                    position['z'] = z_positions.index(position.get('z', 0))
            self._view_msims = None
        # only runs before registration, so the source transform key is the only one: no need to build msims
        coord_systems = [self.reg.source_transform_key]
        self.populate_channels()
        self.populate_coordinate_systems(coord_systems)
        if self.update_output_channels():
            self.populate_channels_table()
        if self.reg.is_initialised():
            # without transform_keys the msims are not read, so None avoids forcing their build
            self.populate_metadata_table(None)
            self.check_3d_view()
            if not skip_view_update:
                # no factory: drawing is the next operation, or a phase of the caller's if one runs
                self.update_views(show_images=False)

        return True

    @property
    def view_msims(self):
        # per-source msims for the image layer, built lazily only when the fused preview needs them
        return self.ensure_view_msims()

    def ensure_view_msims(self, progress_factory=None):
        # the view_msims build, callable ahead with a progress_factory; kept 2D, as consumers place
        # sources at their z (promoting every level of every source is slow)
        if self._view_msims is None:
            with Timer('build view msims', verbose=self._timing_verbose()):
                self._view_msims = self._build_view_msims(progress_factory=progress_factory)
        return self._view_msims

    def _copy_transforms_to_view_msims(self, transform_key, progress_factory=None):
        # building view_msims takes a while, so it reports progress rather than running as a silent side effect
        with self._operation_progress('Building views', progress_factory) as factory:
            view_msims = self._run_off_thread(
                lambda worker_factory: self.ensure_view_msims(progress_factory=worker_factory),
                factory)
        with Timer('copy transforms to view msims', verbose=self._timing_verbose()):
            copy_transforms_to_msims(self.reg.msims, view_msims, transform_key)

    @view_msims.setter
    def view_msims(self, value):
        self._view_msims = value

    def _build_view_msims(self, progress_factory=None):
        # a native pyramid is used as-is; a single-resolution source is downscaled to at most 1000px
        view_msims = []
        progress_context = progress_phase(progress_factory, total=len(self.reg.sources), desc='Building views')
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
    def run_pre_processing(self, progress_factory=None, full_pyramid=False):
        params_features = self.params['pre_processing']
        # the msim build gets bar room only when pending, weighted by its cost: it dominates unless a step
        # computes over the data (flat-field, normalisation, foreground)
        build_weight = 2 if self.reg.has_eager_pre_processing(params_features) else 8
        # registration reads the finest level and the preview the coarsest; convert writes every level
        ends_only = not full_pyramid
        build_pending = self.reg.msims_build_pending(params_features.get('scale'), ends_only=ends_only)
        phases = build_weight + 1 if build_pending else 1
        with self._operation_progress('Pre-processing', progress_factory, phases=phases) as progress_factory, \
             Timer('pre_processing_process', verbose=self._timing_verbose()):
            # msims built explicitly through ensure_msims() so the build reports progress
            def preprocess(worker_factory):
                with Timer('run_pre_processing: build msims (load image data)',
                           verbose=self._timing_verbose()):
                    # at pre-processing's scale: finer levels would only be dropped again
                    msims = self.reg.ensure_msims(progress_factory=worker_factory,
                                                  target_scale=params_features.get('scale'),
                                                  ends_only=ends_only,
                                                  weight=build_weight)
                with Timer('run_pre_processing: preprocess', verbose=self._timing_verbose()):
                    return self.reg.preprocess(msims, progress_factory=worker_factory,
                                               **params_features)

            _, _, modified = self._run_off_thread(preprocess, progress_factory)
        self.pre_processing_performed = modified
        return True

    def pre_processing_process(self):
        # the work and showing its result are two operations, each with its own bar
        self.prepare_action()
        if not self.run_pre_processing():
            return
        if self.reg.is_pairs_registered():
            # register_msims changed: the prior registration is stale and must not give the view its transform
            self.reg.state = RegState.SIMS_INIT
        self.update_views(show_preprocessed=True)
        self.enable_modify_pair_registration(False)
        self.update_fusion_preview_enabled()
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
        properties = ['position', 'pixel size']
        if transform_keys is None:
            positions = self.reg.positions
            scales = self.reg.scales
        else:
            positions = [get_sim_position_final(sim, transform_keys=transform_keys) for sim in sims]
            scales = [get_sim_physical_size(sim) for sim in sims]
        for position, rotation in zip(positions, self.reg.rotations):
            if rotation is not None:
                position['r'] = rotation
        order = sorted(range(len(positions)), key=lambda i: position_sort_key(positions[i]))
        data = [[print_dict_simple(positions[i]), print_dict_simple(scales[i])] for i in order]
        row_headers = [self.reg.file_labels[i] for i in order]
        # Table: tuple-of-values : ([values], [row_headers], [column_headers])
        table_widget.set_value((data, row_headers, properties))
        table_widget.set_table_column_resize_mode()

    def update_output_channels(self):
        channels = self.extra_metadata.get('channels')
        # blank-labelled channels count as unconfigured, so they are derived again
        if not channels or not any(channel.get('label') for channel in channels):
            source0 = self.reg.sources[0]
            channels = source0.get_channels()

            dimension = self.params['input_output']['registration_dimension']
            while dimension.lower() == 'c' and len(channels) < len(self.reg.sources):
                channel = {'label': f'channel {len(channels)}'}
                channels.append(channel)

            self.extra_metadata['channels'] = channels

            # `or`, not a get() default: an unnamed OME channel has an explicit blank label
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
        table = self.param_widgets.get('input_output.channels_table').widget
        if 'color' not in table.column_headers:
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
        # written into the table so its 'changed' wiring saves it, as a manual edit would
        table.data[rowi, color_coli] = str(tuple(color))

    def populate_image_selection(self):
        labels = list(self.reg.file_labels)
        positions = self.reg.positions
        # the metadata and metrics tables' order
        if len(positions) == len(labels):
            labels = [labels[index] for index in sorted(range(len(labels)),
                                                        key=lambda index: position_sort_key(positions[index]))]
        widget1 = self.param_widgets.get('registration.reg_preview_image1')
        widget1.set_value(labels[0], choices=labels)

        widget2 = self.param_widgets.get('registration.reg_preview_image2')
        index = 1 if len(labels) > 1 else 0
        widget2.set_value(labels[index], choices=labels)

    def select_pair_preview(self, ref):
        # only an overlap shape's ref ('0 1') names a pair; single-image clicks are ignored
        indices = ref.split()
        if len(indices) == 2:
            self.set_preview_images([int(index) for index in indices])

    def metrics_table_selected(self):
        table = self.param_widgets.get('registration.metrics_table').widget.native
        header_item = table.verticalHeaderItem(table.currentRow())
        # the summary and group (section) pair rows have no image pair
        indices = getattr(self, 'metrics_row_pairs', {}).get(header_item.text()) if header_item else None
        if indices:
            self.set_preview_images(indices)

    def set_preview_images(self, indices):
        labels = self.reg.file_labels
        self.param_widgets.get('registration.reg_preview_image1').set_value(labels[indices[0]])
        self.param_widgets.get('registration.reg_preview_image2').set_value(labels[indices[1]])

    def get_best_transform_key(self):
        # from the state, not off built msims: a loaded registration's are built only when needed
        return self.reg.reg_transform_key if self.reg.is_global_registered() else self.reg.source_transform_key

    def check_3d_view(self):
        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        ndisplay = 3 if is_3d else 2
        self.viewer.dims.ndisplay = ndisplay
        #self.overview._qtwidget._viewer_model.dims.ndisplay = ndisplay

    def _timing_verbose(self):
        # some view methods are tested on a bare Interface without self.reg
        return getattr(getattr(self, 'reg', None), 'logging_time', False)

    def update_views(self, transform_key=None, show_preprocessed=False, show_images=True,
                     progress_factory=None):
        if transform_key is None:
            transform_key = self.get_best_transform_key()

        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        is_multi_z_shapes = (len(set(position.get('z', 0) for position in self.reg.positions)) > 1)
        force_2d = is_multi_z_shapes and not is_3d

        # steps weighted by cost: building the view data dominates, and reports from the inside
        view_data_weight = 12
        # shapes also grow with the source count, and report per source
        shapes_weight = 3
        phases = shapes_weight + 2 + (view_data_weight + 1 if show_images else 0)

        with self._operation_progress('Refreshing view', progress_factory, phases=phases) as factory:
            with Timer('update_views: create shapes', verbose=self._timing_verbose()):
                # pure geometry, so off the Qt thread to keep the window responsive
                shapes, refs, labels, face_colors = self._run_off_thread(
                    lambda worker_factory: self._create_napari_shapes(
                        transform_key, force_2d=force_2d, progress_factory=worker_factory,
                        weight=shapes_weight),
                    factory)

            # each section built only when viewed: image data as soon as the shapes
            with Timer('update_views: lazy overview', verbose=self._timing_verbose()):
                data = self._run_off_thread(
                    lambda worker_factory: self._create_lazy_overview(transform_key,
                                                                      show_preprocessed=show_preprocessed), factory)
            # otherwise (a rotation, or sources it cannot overlay) the fused preview
            if data is None and show_images:
                with Timer('update_views: create fused data', verbose=self._timing_verbose()):
                    # fused off the Qt thread; adding it to the viewer below must stay on it
                    data = self._run_off_thread(
                        lambda worker_factory: self._create_napari_data(
                            transform_key, show_preprocessed=show_preprocessed,
                            progress_factory=worker_factory, weight=view_data_weight),
                        factory)
            # cleared only once the new data is ready: an empty viewer's welcome screen hides the activity dialog
            self._clear_napari_view(self.viewer)
            if data is not None:
                with factory(total=1) as pbar, \
                     Timer('update_views: add fused data to viewer', verbose=self._timing_verbose()):
                    # an overview, not the fusion preview: a naive contrast guess is enough
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

    def _create_lazy_overview(self, transform_key, show_preprocessed=False):
        reg = self.reg
        transforms, output_order = getattr(reg, '_msim_transforms', None), getattr(reg, '_msim_output_order', None)
        if transforms is None or output_order is None:
            return None
        if transform_key == reg.reg_transform_key:
            # from source geometry and the mappings, as the shapes: no msim built for them
            transforms = [np.asarray(props['transform'], dtype=float) for props in reg.source_geometry(transform_key)[1]]
        readers, label = None, 'Overview'
        if show_preprocessed and reg.register_msims is not None:
            # a source pre-processing dropped (filter_foreground) is left out, not shown raw
            readers = [None] * len(reg.sources)
            for index, msim in zip(reg.register_indices, reg.register_msims):
                readers[index] = MsimLevels(msim)
            label = 'Pre-processed overview'
        channel_labels = None
        if reg.is_channel_overlay(self.params['input_output']['registration_dimension'], self.extra_metadata):
            channels = self.extra_metadata.get('channels', [])
            channel_labels = [channels[index].get('label') if index < len(channels) else f'channel {index}'
                              for index in range(len(reg.sources))]
        return lazy_section_overview(reg.sources, reg.positions, transforms, output_order, transform_key,
                                     z_scale=getattr(reg, '_msim_z_scale', None),
                                     preview_scale=self.params['input_output'].get('preview_scale'),
                                     readers=readers, label=f'{label} ({len(reg.sources)} images)',
                                     channel_labels=channel_labels)

    def _refresh_overview_shapes(self, transform_key, shapes=None, refs=None, labels=None,
                                 face_colors=None, is_3d=None):
        # the overview is always flat and top-down, so a 3D shape set is built again with force_2d
        if is_3d is None:
            is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        if shapes is None or is_3d:
            shapes, refs, labels, face_colors = self._create_napari_shapes(transform_key, force_2d=True)
        self._clear_napari_view(self.overview)
        self._update_view_add_shapes(self.overview, shapes, refs, labels, face_colors,
                                     f'{self.reg.fileset_label} shapes', show_labels=False)

    def _clear_napari_view(self, viewer):
        # no empty LayerList.clear() event: under xpra/Xvfb it breaks blending for later Shapes/Points layers
        if viewer is not None and len(viewer.layers) > 0:
            viewer.layers.clear()
        if viewer is not None and viewer is getattr(self, 'viewer', None):
            clear_napari_dask_cache()
            self._view_levels = None

    def _create_napari_shapes(self, transform_key, force_2d=False, progress_factory=None, weight=1):
        # `weight` is this step's share of the caller's bar, split between sub-steps by rough cost
        def phase(share, total=None):
            return progress_phase(progress_factory, total=total,
                                        weight=max(weight * share, 1))

        # sources at different heights without a native 'z' need it as a dim, or their z is dropped
        promote_z = (len(set(position.get('z', 0) for position in self.reg.positions)) > 1)
        with phase(3 / 5, total=len(self.reg.sources)) as pbar, \
             Timer(f'_create_napari_shapes: build {len(self.reg.sources)} source shape geometries',
                   verbose=self._timing_verbose()):
            # stack properties from source geometry and the registration's mappings: no msim built, no pixel read
            _, msims = self.reg.source_geometry(
                transform_key, promote_z=promote_z, z_scale=self.reg._msim_z_scale,
                progress=pbar.update if pbar is not None else None)

        with phase(1 / 5, total=1) as pbar, \
             Timer(f'_create_napari_shapes: create_image_shapes ({len(msims)} images)', verbose=self._timing_verbose()):
            shapes = create_image_shapes(msims, transform_key=transform_key, force_2d=force_2d)
            if pbar is not None:
                pbar.update(1)
        refs = [str(index) for index in range(len(msims))]
        labels = list(self.reg.file_labels)
        face_colors = [(1, 1, 1) for _ in range(len(msims))]

        # once registered, only registered pairs: an overlap between unpaired images has no quality behind it
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

    def _create_napari_data(self, transform_key, fusion_method='additive', show_preprocessed=False,
                            progress_factory=None, weight=1):
        # `weight` is this step's share of the caller's bar, split between sub-steps by rough cost
        def phase(share, total=None):
            return progress_phase(progress_factory, total=total,
                                        weight=max(weight * share, 1))

        # only copy_transforms_to_msims() writes into a msim; the steps between may return their input
        # unchanged, so any new mutating step needs this flag too
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
        else:
            # view_msims hold full pyramids: reduce to create_preview()'s coarse sub-pyramid before fusing
            preview_scale = parse_scale(self.params['input_output'].get('preview_scale'),
                                        default=default_interactive_preview_scale)
            with phase(1 / 12, total=1) as pbar, \
                 Timer('_create_napari_data: build view_msims', verbose=self._timing_verbose()):
                view_msims = self.view_msims
                if transform_key == self.reg.reg_transform_key:
                    # a source downscaled for the view has its own msim, without the registered transform
                    copy_transforms_to_msims(self.reg.msims, view_msims, transform_key)
                if pbar is not None:
                    pbar.update(1)
            with phase(1 / 12, total=1) as pbar, \
                 Timer(f'_create_napari_data: select_msim_subpyramid_at_scale ({len(view_msims)} images)',
                      verbose=self._timing_verbose()):
                msims = select_msim_subpyramid_at_scale(view_msims, self.reg.sources, preview_scale)
                if pbar is not None:
                    pbar.update(1)
        # sources at several heights are sized as if promoted to 3D, each at its own z as make_msims_3d() takes it,
        # without promoting them
        z_positions = None
        if len(set(position.get('z', 0) for position in self.reg.positions)) > 1:
            z_positions = [self.reg.positions[index].get('z', index) for index in range(len(msims))]
        # cap the combined fused size: pre-processed msims ignore preview_scale, which only picks a level per source
        # the cap also takes the bar share of the z promotion it replaces
        with phase((2 if z_positions is not None else 1) / 12, total=1) as pbar, \
             Timer('_create_napari_data: cap preview fusion size', verbose=self._timing_verbose()):
            msims = reduce_msims_to_fused_size(
                msims, transform_key, z_scale=self.reg._msim_z_scale,
                label=f'Preview fusion ({len(msims)} images)', z_positions=z_positions)
            if pbar is not None:
                pbar.update(1)
        with phase(1 / 12, total=1) as pbar, \
             Timer('_create_napari_data: copy_transforms_to_msims', verbose=self._timing_verbose()):
            # before registration there is nothing to copy, and reading self.reg.msims would force their build
            if show_preprocessed and writes_transforms:
                transform_msims = self.reg.msims
                if self.reg.register_indices is not None:
                    # pre-processing may drop sources (filter_foreground): pair each with its own source
                    transform_msims = [self.reg.msims[index] for index in self.reg.register_indices]
                copy_transforms_to_msims(transform_msims, msims, transform_key)
            if pbar is not None:
                pbar.update(1)
        # chunksize is left to fuse(), which derives it after its own z promotion; no step count, as
        # fuse() plans the whole graph in one call
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

    def _view_background_luminance(self, viewer):
        """The luminance of the image the main viewer shows (background_luminance), None for another viewer or none."""
        if viewer is not getattr(self, 'viewer', None) or getattr(self, '_view_levels', None) is None:
            return None
        return background_luminance(*self._view_levels)

    def _update_view_add_shapes(self, viewer, shapes, refs, labels, face_colors, layer_name,
                                show_labels=True):
        # reads source metadata, not view_msims, so drawing shapes never forces the msim build
        # labels and edges stand out against the image under them, if this viewer shows one
        line_color = contrasting_color(self._view_background_luminance(viewer))
        bb_supported = True
        if isinstance(viewer, ViewerWidget):
            viewer = viewer._qtwidget._viewer_model
            bb_supported = False
        is_3d = (self.reg.sources[0].get_size().get('z', 0) > 1)
        is_multi_z_shapes = (len(set(position.get('z', 0) for position in self.reg.positions)) > 1)
        force_2d = not bb_supported or (is_multi_z_shapes and not is_3d)
        # shapes carry a 'z' for a native z-stack or for sources promoted to one
        do_3d = ((is_3d or is_multi_z_shapes) and not force_2d)

        if len(shapes) > 0:
            # depth-tested blending hides boxes inside the opaque fused volume
            blending = 'translucent_no_depth'
            edge_color = line_color
            if do_3d:
                # a polygon is one flat face, so a box is 6 quads plus a 'path' wireframe; corners 0-3 are
                # one face and 4-7 the opposite, same winding
                box_faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                            [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
                edge_path = [0, 1, 2, 3, 0, 4, 7, 3, 2, 6, 7, 4, 5, 6, 2, 1, 5]
                # _minimal_bb_vertices' corner order, applied to an axis-aligned min/max
                corner_bits = np.array([
                    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                ])

                face_shapes, face_only_colors, face_refs, face_labels = [], [], [], []
                for shape, ref, color in zip(shapes, refs, face_colors):
                    corners = np.asarray(shape)
                    # napari fills only axis-orthogonal 3D faces (napari/napari#6860): fill the axis-aligned
                    # bounding box; the wireframe keeps the true oriented corners
                    mins, maxs = corners.min(axis=0), corners.max(axis=0)
                    aa_corners = mins + corner_bits * (maxs - mins)
                    centroid = aa_corners.mean(axis=0)
                    for face in box_faces:
                        quad = aa_corners[face]
                        # box_faces do not all wind outward: flip inward faces, or napari backface-culls them
                        normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
                        if np.dot(normal, quad.mean(axis=0) - centroid) < 0:
                            quad = quad[::-1]
                        face_shapes.append(quad)
                    face_only_colors += [color] * len(box_faces)
                    face_refs += [ref] * len(box_faces)
                    face_labels += [''] * len(box_faces)
                wire_shapes = [np.asarray(shape)[edge_path] for shape in shapes]

                # 3D path width is in world units, so scale it to the scene to stay visible
                vertices = np.concatenate([np.asarray(shape) for shape in shapes])
                wire_width = np.ptp(vertices, axis=0).max() * 0.005

                shape_data = face_shapes + wire_shapes
                shape_type = ['polygon'] * len(face_shapes) + ['path'] * len(wire_shapes)
                # color lists must match in length or napari falls back to white; a path's face color is a placeholder
                face_color = face_only_colors + [(0, 0, 0)] * len(wire_shapes)
                edge_color = ['#000000'] * len(face_shapes) + [line_color] * len(wire_shapes)
                edge_width = [0] * len(face_shapes) + [wire_width] * len(wire_shapes)
                refs = face_refs + refs
                labels = face_labels + labels
            else:
                shape_data = np.asarray(shapes)
                shape_type = 'polygon'
                edge_width = 0.1
                face_color = face_colors

            # half napari's default size, which reads oversized; the small overview's labels would be unreadable
            text = {'string': '{labels}', 'size': 6, 'color': line_color} if show_labels else None
            # 'labels' only feeds the text; 'refs' identifies the shapes either way
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
        # `fused` is one channel-combined multiscale msim, or in 'compose' mode a list of per-source msims;
        # cheap=True guesses contrast limits from the dtype instead of computing them
        channels = self.extra_metadata.get('channels', [])

        if isinstance(fused, list):
            # one layer per source; add_image() must stay on the Qt thread, so only the layer prep is pooled
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
                layer_kwargs = [prep_layer(msim, channel) for msim, channel in zip(fused, channel_list)]

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
            # the quick overview brings its own, from a sample of its sources
            channel_limits, levels = fused.attrs.get('contrast_limits'), fused.attrs.get('display_levels')
            if not channel_limits:
                channel_limits, levels = get_contrast_limits(fused, cheap=cheap, per_channel=True, with_levels=True)
        # the single-channel overview has c of size 1 even when the metadata lists more channels
        if len(channels) > 1 and image0.sizes.get('c', 1) == len(channels):
            channel_axis = image0.dims.index('c')
            name = [channel.get('label', index) for index, channel in enumerate(channels)]
            colormap = [channel.get('color', (1, 1, 1, 1)) for channel in channels]
            scale = [scale] * len(channels)
            translate = [translate] * len(channels)
            contrast_limits = (channel_limits if len(channel_limits) == len(channels)
                               else [channel_limits[0]] * len(channels))
        else:
            channel_axis = None
            name = channels[0].get('label') if len(channels) == 1 else None
            colormap = channels[0].get('color', (1, 1, 1, 1)) if len(channels) == 1 else None
            contrast_limits = channel_limits[0]
        if viewer is getattr(self, 'viewer', None):
            # the shapes drawn over it pick their colour from this (_update_view_add_shapes)
            colors = colormap if channel_axis is not None else [colormap or (1, 1, 1)]
            self._view_levels = (levels[:len(colors)] if levels else None, colors)
        with Timer('_napari_view_add_fused_data: add_image', verbose=self._timing_verbose()):
            viewer.add_image(data, name=name or layer_name, multiscale=True, channel_axis=channel_axis,
                             colormap=colormap, contrast_limits=contrast_limits,
                             scale=scale, translate=translate)

    def _napari_view_show_features(self, viewer, fixed_data2, fixed_points, moving_data2, moving_points, matches, inliers):
        layers = draw_keypoints_matches_napari(fixed_data2, fixed_points,
                                               moving_data2, moving_points,
                                               matches, inliers, points_color='blue')
        self._clear_napari_view(viewer)
        for data, kwargs, layer_type in layers:
            if layer_type == "image":
                viewer.add_image(data, **kwargs)
            elif layer_type == "points":
                viewer.add_points(data, **kwargs)
            elif layer_type == "shapes":
                viewer.add_shapes(data, **kwargs)

    def _napari_view_add_image(self, viewer, data, label, transform=None, color=None, affine_event=False):
        if isinstance(data, DataTree):
            # a multiscale msim: napari's affine (pair drag adjustment) works with its lazy level loading
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
        if self.view_mode != ViewMode.PAIRS:
            return
        if self._pair_metrics_worker is not None:
            # one at a time: the latest position is measured once this one is done
            self._pair_metrics_pending = True
            return
        # read off the layers here, on the Qt thread: only the measuring runs on the worker
        pair_indices = self.pair_indices
        reg_msims = [self.reg.register_msims[index] for index in pair_indices]
        transforms = {(0, 1): self.calc_mod_pair_transform()}

        def measure():
            metrics = calc_msims_metrics(reg_msims, transforms, metric_methods=self.metrics_methods)
            self._remap_local_pair_metrics(metrics, pair_indices)
            return metrics

        def show(metrics):
            if self.view_mode == ViewMode.PAIRS and self.pair_indices == pair_indices:
                self.populate_metrics_table(metrics)

        def done():
            self._pair_metrics_worker = None
            if self._pair_metrics_pending:
                self._pair_metrics_pending = False
                self.update_pair_metrics()

        if QApplication is None or QApplication.instance() is None:
            show(measure())
            return
        # off the Qt thread, so the layers can still be dragged while it measures
        self._pair_metrics_worker = create_worker(
            measure,
            _start_thread=False,
            _connect={
                'returned': show,
                'errored': lambda error: show_warning(f'Pair metrics failed: {error}'),
                'warned': lambda warning: notification_manager.receive_warning(*warning),
                'finished': done,
            },
        )
        self._pair_metrics_worker.start()

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
                # dask's task counting reports as a phase of this bar, not a bar of its own
                with NapariDaskProgress(progress_class=worker_factory,
                                        desc='Preview registration'):
                    registration_params = self.params['registration']
                    channel = registration_params.get('channel')
                    cache = self._preview_overlap_cache
                    # the crop depends only on register_msims and the pair/channel, so parameter changes reuse it
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

                    with Timer('preview_registration: register the overlap', verbose=self._timing_verbose()):
                        transform, quality, results = self.reg.register_overlap(
                            overlap1, overlap2, sims_pixel_space, params=registration_params)

                    msim1, msim2 = self.reg.register_msims[index1], self.reg.register_msims[index2]
                    transforms = {(0, 1): transform}
                    qualities = {(0, 1): quality}
                    with Timer('preview_registration: pair metrics', verbose=self._timing_verbose()):
                        metrics = calc_msims_metrics((msim1, msim2), transforms, qualities,
                                                     metric_methods=self.metrics_methods)
                    self._remap_local_pair_metrics(metrics, (index1, index2))
                    return metrics, results, overlap1, overlap2

            try:
                return self._run_off_thread(register_preview, factory)
            except NoOverlapError:
                show_warning(f'{label1} and {label2} do not overlap: choose two images that do')
                return None


    def preview_registration(self):
        # the view stays until there is a result to replace it with: an empty viewer's welcome screen hides the bar
        with Timer('preview_registration: register (and pre-process if needed)', verbose=self._timing_verbose()):
            result = self.run_preview_registration()
        if result is None:
            return
        metrics, results, overlap1, overlap2 = result

        with Timer('preview_registration: show metrics', verbose=self._timing_verbose()):
            self.populate_metrics_table(metrics)
            # metrics shown before the (slower) feature layers replace the view
            flush_paint_events()

        fixed_points = results.get('fixed_points', [])
        moving_points = results.get('moving_points', [])
        matches = results.get('matches', [])
        inliers = results.get('inliers', [])
        if len(fixed_points) == 0 and 'affine_matrix' in results:
            # a method giving only a transform (phase correlation, elastix): a grid shows its offsets
            fixed_points, moving_points, matches, inliers = grid_point_pairs(
                np.shape(overlap1), np.shape(overlap2), results['affine_matrix'])
        with Timer('preview_registration: show feature layers', verbose=self._timing_verbose()):
            self._napari_view_show_features(self.viewer, overlap1, fixed_points, overlap2, moving_points, matches,
                                            inliers)
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
        # a selected row picks its pair's preview images (metrics_table_selected)
        self.metrics_row_pairs = {self.reg.file_labels[indices[0]] + ' - ' + self.reg.file_labels[indices[1]]: indices
                                  for indices in (pairs_metrics or {})}
        # split pairing's group (e.g. section) pairs as rows after the tile pairs, keyed by their labels
        row_metrics = {row_label: pairs_metrics[indices] for row_label, indices in self.metrics_row_pairs.items()}
        row_metrics |= {' - '.join(labels): value for labels, value in metrics_dict.get('group_pairs', {}).items()}
        # a dict keeps the order without a quadratic `not in` over the rows
        item_keys = list(dict.fromkeys(item_keys + list(row_metrics)))
        for pair_value in row_metrics.values():
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
        for pair_index, pair_value in enumerate(row_metrics.values()):
            for transform_key, transform_value in pair_value.items():
                for metric_key, metric_value in transform_value.items():
                    if metric_value is not None:
                        col_index = (metric_keys.index(metric_key) if is_metric_cols
                                     else transform_keys.index(transform_key.split('_')[0]))
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
        with Timer('update_registered: tables', verbose=self._timing_verbose()):
            # the coordinate systems from the state: reading them off the msims would build every one
            coord_systems = [self.reg.source_transform_key] + (
                [self.reg.reg_transform_key] if self.reg.is_global_registered() else [])
            self.populate_coordinate_systems(coord_systems)
            self.populate_metadata_table(None)
            self.populate_metrics_table(self.reg.metrics)
        self.update_views(transform_key=view_transform_key, progress_factory=progress_factory)

    def enable_modify_pair_registration(self, enabled=True):
        widget = self.param_widgets.get('registration.modify_pair_registration')
        if widget:
            widget.widget.enabled = enabled

    def update_fusion_preview_enabled(self):
        widget = self.param_widgets.get('fusion.preview_fusion')
        if widget:
            # only a merge fuses at the unregistered source positions
            operation = getattr(self, 'params', {}).get('registration', {}).get('operation', '')
            widget.widget.enabled = self.reg.is_global_registered() or 'merge' in operation

    @catch_run_errors
    def run_pair_registration(self, progress_factory=None):
        # pre-processing reports its own bar, before the registration's
        if not self.reg.register_msims:
            if not self.run_pre_processing(progress_factory=progress_factory):
                return None

        # the split pairing groups sources by it (z-planes or channels)
        self.reg.registration_dimension = self.params.get('input_output', {}).get('registration_dimension')
        with self._operation_progress('Pair registration', progress_factory, phases=2) as factory:
            def register_pairs(worker_factory):
                # pairs and metrics report per batch: a dask callback would open a phase per batch
                with Timer('pair registration', verbose=self._timing_verbose()):
                    return self.reg.register_pairs(
                        self.reg.register_msims,
                        params=self.params['registration'] | {'metrics': self.metrics_methods},
                        progress_factory=worker_factory)

            results = self._run_off_thread(register_pairs, factory)

        self.reg.save_pair_results(results['pair_mappings'], metrics=results['metrics'])
        self.enable_modify_pair_registration()
        return results

    @catch_run_errors
    def run_global_registration(self, progress_factory=None):
        # register_global() reports its stages; declaring 8 units, half for the dominant optimisation,
        # keeps the last stages from each taking most of the bar
        self.reg.registration_dimension = self.params.get('input_output', {}).get('registration_dimension')
        if self.reg.pair_msims is None or self.reg.register_msims is None:
            # a loaded pair registration needs the pre-processed msims: at full resolution metrics can run out of memory
            if not self.run_pre_processing(progress_factory=progress_factory):
                return None
            self.reg.pair_msims = self.reg.register_msims
        with self._operation_progress('Global registration', progress_factory, phases=8) as factory:
            def register_global(worker_factory):
                with NapariDaskProgress(progress_class=worker_factory, desc='Global registration'), \
                        Timer('global registration', verbose=self._timing_verbose()):
                    return self.reg.register_global(self.reg.pair_msims,
                                                    register_indices=self.reg.register_indices,
                                                    params=self.params['registration'],
                                                    progress_factory=worker_factory)

            # the transform is written onto the sources before the metrics: a cancel restores it
            key = self.reg.reg_transform_key
            snapshots = [(msims, snapshot_msims_transform(msims, key)) for msims in (self.reg.pair_msims, self.reg.msims)]
            try:
                results = self._run_off_thread(register_global, factory)
            except OperationCancelled:
                for msims, snapshot in snapshots:
                    restore_msims_transform(msims, key, snapshot)
                raise

        with Timer('save registration results', verbose=self._timing_verbose()):
            self.reg.save_group_pair_mappings()
            self.reg.save_mappings(results['mappings'])
            self.reg.save_mappings_csv(results['mappings'])
            self.reg.save_metrics(results['metrics'])
            self.reg.write_ro_crates([], self.get_output_params_path() or getattr(self, 'params_path', None),
                                     self.params)
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
            self.update_fusion_preview_enabled()
            QMessageBox.information(None, 'muvis-align', 'Pair registration completed')

    def modify_pair_registration(self):
        if self.view_mode == ViewMode.PAIRS:
            reply = QMessageBox.question(None, 'muvis-align','Store modified registration?',
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                transform = self.calc_mod_pair_transform()
                pair_transforms = nx.get_edge_attributes(self.reg.pairs_graph, default_transform_key)
                qualities = nx.get_edge_attributes(self.reg.pairs_graph, default_quality_key)
                if 't' in pair_transforms[self.pair_indices].dims:
                    transform = transform.expand_dims({'t': [0]})
                pair_transforms[self.pair_indices] = transform
                qualities[self.pair_indices] = np.array(1)    # set quality to 1
                nx.set_edge_attributes(self.reg.pairs_graph, pair_transforms, default_transform_key)
                nx.set_edge_attributes(self.reg.pairs_graph, qualities, default_quality_key)
                self.reg.save_pair_results(pair_transforms, qualities=qualities)

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
            pair_transforms = (nx.get_edge_attributes(self.reg.pairs_graph, default_transform_key)
                               if self.reg.is_pairs_registered() else {})
            if indices not in pair_transforms and tuple(reversed(indices)) in pair_transforms:
                indices = tuple(reversed(indices))

            if indices not in pair_transforms:
                self.view_mode = ViewMode.OVERVIEW
                if self.reg.is_pairs_registered():
                    show_warning(f'No pair registration for {label1} and {label2}: choose a registered pair')
                else:
                    show_warning('Run pair registration first: a preview registration is not stored')
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

                # the widgets disabled above are only re-enabled on leaving this mode, so a failure must restore them
                entered = False
                try:
                    if not self.reg.register_msims:
                        if not self.run_pre_processing():
                            return
                    self._clear_napari_view(self.viewer)
                    # a multiscale pyramid, so napari loads only the level it needs while dragging
                    register_images = self.reg.register_msims
                    # each source keeps its own transform (e.g. scan rotation); the pair transform maps fixed to moving
                    self.pair_source_affines = [
                        np.asarray(msi_utils.get_transform_from_msim(register_images[sim_index],
                                                                     self.reg.source_transform_key)).squeeze()
                        for sim_index in indices]
                    fixed_source_affine, moving_source_affine = self.pair_source_affines
                    layer_affines = pair_transform @ fixed_source_affine, moving_source_affine
                    for sim_index, layer_affine, color in zip(indices, layer_affines, colors):
                        self._napari_view_add_image(self.viewer, register_images[sim_index], labels[sim_index],
                                                    layer_affine, color, affine_event=True)
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
        fixed_source_affine, moving_source_affine = self.pair_source_affines
        matsize = len(fixed_source_affine)
        fixed_affine, moving_affine = [layer.affine.affine_matrix[-matsize:, -matsize:] for layer in self.viewer.layers]
        # either layer may have been moved: the pair transform is what is left once both source transforms are taken out
        transform = (moving_source_affine @ calculate_rigid_difference(moving_affine, fixed_affine)
                     @ np.linalg.inv(fixed_source_affine))
        return param_utils.affine_to_xaffine(transform)

    def registration_process(self):
        self.prepare_action()
        if 'convert' in self.params['registration']['operation']:
            # each source written at its own position, without registration or fusion
            reply = QMessageBox.question(None, 'muvis-align', 'Convert data to OME-Zarr only?',
                                         QMessageBox.Yes|QMessageBox.No)
            if reply == QMessageBox.Yes:
                if self.run_convert():
                    QMessageBox.information(None, 'muvis-align', 'Conversion completed')
            return

        if 'merge' in self.params['registration']['operation']:
            # merge fuses at the sources' own positions (get_best_transform_key()): nothing to register
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
            if not self._run_registration():
                return
            # one more bar for building and drawing the view the registration leaves on screen
            with self._operation_progress('Refreshing view', phases=2) as view_factory:
                self._copy_transforms_to_view_msims(self.reg.reg_transform_key,
                                                    progress_factory=view_factory)
                self.update_registered(view_transform_key=self.reg.reg_transform_key,
                                       progress_factory=view_factory)
            QMessageBox.information(None, 'muvis-align', completion_message)

    def _run_registration(self):
        """Pair registration where it has not run yet, then global; False if either did not complete."""
        # a bar per registration: in one shared bar the pair phases would fill it
        with Timer('registration process', verbose=self._timing_verbose()):
            registered = ((self.reg.is_pairs_registered() or self.run_pair_registration())
                          and self.run_global_registration())
        self.update_fusion_preview_enabled()
        return bool(registered)

    @catch_run_errors
    def _show_fused_output(self, path, progress_factory=None):
        """Show the fused export on disk, read lazily at every exported level, without fusing again."""
        transform_key = self.get_best_transform_key()
        with progress_phase(progress_factory, total=1, desc='Reading fused output') as pbar, \
             Timer('show fused output', verbose=self._timing_verbose()):
            fused = ngff_utils.read_msim_from_ome_zarr(path, transform_key=transform_key, array_backend='dask')
            self._clear_napari_view(self.viewer)
            self._napari_view_add_fused_data(self.viewer, fused, 'Fused')
            self.view_mode = ViewMode.FUSED
            if pbar is not None:
                pbar.update(1)
        logging.info(f'Fused output shown from {path}')
        self._refresh_overview_shapes(transform_key)

    def preview_fusion(self, progress_factory=None):
        # a merge has no registered transform, so this falls back to the source positions
        transform_key = self.get_best_transform_key()
        # building the data dominates, so it is weighted as such and reports from the inside
        fusion_weight = 12
        with self._operation_progress('Fusion preview', progress_factory,
                                      phases=fusion_weight + 1) as factory:
            def fuse(worker_factory):
                with NapariMVSProgress(tqdm_class=worker_factory.tqdm_class, patch_fusion=True):
                    return self._create_napari_data(
                        transform_key, fusion_method=self.params['fusion']['method'],
                        progress_factory=worker_factory, weight=fusion_weight)

            data = self._run_off_thread(fuse, factory)
            with factory(total=1) as pbar:
                self._clear_napari_view(self.viewer)
                self._napari_view_add_fused_data(self.viewer, data, f'{self.reg.fileset_label} data')
                self.view_mode = ViewMode.FUSED
                pbar.update(1)
        # this path skips update_views(), so the overview would stay empty
        self._refresh_overview_shapes(transform_key)

    @catch_run_errors
    def run_convert(self):
        operation = self.params['registration']['operation']
        output_folder = operation_to_past_participle(operation)
        ome_version = self.params['fusion']['ome_version']
        # self.reg.msims stays at source resolution; the scaled pyramid is register_msims, with
        # every level (pre-processing otherwise keeps only the ends)
        if not self.run_pre_processing(full_pyramid=True):
            return False
        msims = self.reg.register_msims
        # foreground filtering may drop sources, so labels/positions follow register_indices
        labels = [get_filetitle(self.reg.filenames[index]) for index in self.reg.register_indices]
        positions = [self.reg.positions[index] for index in self.reg.register_indices]
        with self._operation_progress('Converting') as progress_factory, \
             Timer('convert', verbose=self._timing_verbose()):
            # not fuse(): it would build one shared canvas; each source keeps its native levels
            # sequential: concurrent zarr writes race on zarr.json (Windows PermissionError)
            with progress_factory(total=len(msims), desc='Converting') as pbar:
                for label, position, msim in zip(labels, positions, msims):
                    output_filename = f'{output_folder}/{label}'
                    self.reg.save_native_levels(output_filename, msim, position=position,
                                                ome_version=ome_version)
                    pbar.update(1)
        return True

    def _fusion_tile_size(self):
        """The configured tile size, or None: fuse() then sizes the export's blocks by what one costs in memory."""
        tile_size = self.params['fusion'].get('tile_size')
        if isinstance(tile_size, str):
            tile_size = tile_size.strip()
            if not tile_size:
                tile_size = None
            elif ',' in tile_size:
                tile_size = [int(size.strip()) for size in tile_size.split(',')]
            else:
                tile_size = int(tile_size)
        return tile_size

    def _fusion_size_text(self):
        """' Estimated output: ...' for the export question, logged per level; empty if it cannot be estimated."""
        try:
            with Timer('fusion size estimate', verbose=self._timing_verbose()):
                estimate = self.reg.estimate_fusion_size(
                    self.params['fusion']['method'], output_spacing=self.params['fusion']['spacing'],
                    transform_key=self.get_best_transform_key(),
                    dimension=self.params['input_output']['registration_dimension'],
                    tile_size=self._fusion_tile_size(), extra_metadata=self.extra_metadata)
        except Exception as error:
            # the question still stands without a size
            logging.warning(f'Could not estimate the fusion size: {error}')
            return ''
        if estimate is None:
            return ''
        levels = estimate['levels']
        logging.info(f'Fusion size estimate: {print_hbytes(estimate["bytes"])} uncompressed, per level '
                     + ', '.join(f'{level["spacing"]:.4g}: {print_hbytes(level["bytes"])}' for level in levels))
        return f'\nEstimated output: {print_hbytes(estimate["bytes"])}'

    @catch_run_errors
    def run_fusion(self, progress_factory=None):
        operation = self.params['registration']['operation']
        output_filename = operation_to_past_participle(operation)
        tile_size = self._fusion_tile_size()
        # the saves below are on-disk layout only, with no per-block cost, so they keep tiling
        save_tile_size = tile_size or default_chunk_size

        # written straight to zarr, the fusion is the whole export: a second phase would hold its bar at half
        fuses_to_zarr = self.reg.fuses_to_zarr(self.params['fusion']['method'],
                                               self.params['input_output']['registration_dimension'],
                                               self.extra_metadata)
        with self._operation_progress('Fusion', progress_factory, phases=1 if fuses_to_zarr else 2) as factory:
            def fuse(worker_factory):
                with NapariMVSProgress(tqdm_class=worker_factory.tqdm_class, patch_fusion=True), \
                        Timer('fusion', verbose=self._timing_verbose()):
                    fused_image, is_saved = self.reg.fuse(
                        self.reg.msims,
                        fusion_method=self.params['fusion']['method'],
                        # fuse()'s default reg_transform_key does not exist for a merge
                        transform_key=self.get_best_transform_key(),
                        output_spacing=self.params['fusion']['spacing'],
                        dimension=self.params['input_output']['registration_dimension'],
                        output_filename=output_filename,
                        tile_size=tile_size,
                        ome_version=self.params['fusion']['ome_version'],
                        extra_metadata=self.extra_metadata)
                    if not is_saved:
                        # save() takes a single-resolution sim: the pyramid's finest scale
                        save_sim = extract_sims_from_fused(fused_image)
                        self.reg.save(output_filename, save_sim,
                                      transform_key=self.get_best_transform_key(),
                                      translations0=self.reg.positions,
                                      channels=self.extra_metadata.get('channels', []),
                                      tile_size=save_tile_size,
                                      ome_version=self.params['fusion']['ome_version'])
                    image_filenames = [path[len(self.reg.output):] for path in self._output_paths(output_filename)
                                       if os.path.exists(path)]
                    self.reg.write_ro_crates(image_filenames,
                                             self.get_output_params_path() or getattr(self, 'params_path', None),
                                             self.params)
                    return fused_image

            try:
                return self._run_off_thread(fuse, factory)
            except OperationCancelled:
                self._remove_output(output_filename)
                raise

    def _remove_output(self, output_filename, output=None):
        # a cancelled export's partly written file or store, or one fused from a discarded registration
        for path in self._output_paths(output_filename, output):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
            if os.path.exists(path):
                logging.warning(f'Could not remove the output {path}')

    def _output_paths(self, output_filename, output=None):
        output = os.path.join(output, '') if output else self.reg.output
        return [output + output_filename + extension for extension in (zarr_extension, tiff_extension)]


    def fusion_process(self):
        self.prepare_action()
        needs_registration = (not self.reg.is_global_registered()
                              and 'merge' not in self.params['registration']['operation'])
        if self.reg.is_fused():
            message = 'Fusion was already performed. Export fused data?'
        elif needs_registration and self.reg.is_pairs_registered():
            message = 'Global registration not performed yet. Run global registration and export fused data?'
        elif needs_registration:
            message = 'Registration not performed yet. Run pair and global registration, and export fused data?'
        else:
            message = 'Export fused data?'
        reply = QMessageBox.question(None, 'muvis-align', message + self._fusion_size_text(),
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if needs_registration and not self._run_registration():
                return
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
