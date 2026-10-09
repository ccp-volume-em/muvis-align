"""Drive the napari plugin without user input and capture the screen while it works.

    python testing/napari_ui_capture.py C:/project/slides/muvis_align_project.yml shots \
        --action pre_processing --slow make_msims_3d --slow Interface._update_view_add_shapes

Opens the project, runs the action on the Qt thread (as a button click would), and saves a
screenshot every second from a background thread - Qt's own grab would stall with a blocked
Qt thread. Each file is named <index>_<seconds>_<step>_dlg<activity dialog visible>.png.
With --tabs, the plugin is also grabbed on each of its tabs once the action is done (tab_<n>_<label>.png),
and with --window the whole window (window.png) - as the docs show them.
"""
import argparse
import ctypes
import faulthandler
import logging
import os
import sys
import threading
import time

import napari
from PIL import ImageGrab
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QMessageBox

import muvis_align.ui.Interface as interface_module


ACTIONS = {
    'open': lambda interface: interface.input_output_process(),
    'pre_processing': lambda interface: interface.pre_processing_process(),
    'pair_registration': lambda interface: interface.pair_registration(),
    'preview_registration': lambda interface: interface.preview_registration(),
    'registration': lambda interface: interface.registration_process(),
    'fusion': lambda interface: interface.fusion_process(),
    'modify_pair_registration': lambda interface: (interface.registration_process(),
                                                   interface.modify_pair_registration()),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project')
    parser.add_argument('output_dir')
    parser.add_argument('--action', choices=ACTIONS, default='open')
    parser.add_argument('--slow', action='append', default=[],
                        help='Interface-module name to delay, e.g. make_msims_3d or Interface.update_views')
    parser.add_argument('--delay', type=float, default=12)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--cancel-after', type=float, default=None,
                        help='seconds into the action to press its (by then Cancel) Process button')
    parser.add_argument('--tabs', action='store_true',
                        help='once the action is done, save the plugin on each tab')
    parser.add_argument('--window', action='store_true', help='once the action is done, save the whole window')
    parser.add_argument('--pair-offset', type=float, default=0,
                        help='modify_pair_registration: shift the moving image by this much in x (world units)')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    state = {'step': 'start', 't0': time.monotonic(), 'done': False}

    def slowed(name, func):
        def wrapper(*func_args, **kwargs):
            state['step'] = name
            logging.info(f'capture: entering {name}')
            time.sleep(args.delay)
            try:
                return func(*func_args, **kwargs)
            finally:
                state['step'] = f'after {name}'
        return wrapper

    for name in args.slow:
        owner_name, _, attribute = name.rpartition('.')
        owner = getattr(interface_module, owner_name) if owner_name else interface_module
        setattr(owner, attribute, slowed(name, getattr(owner, attribute)))

    # answer the plugin's confirmations and notices: a modal box would wait for a user who is not there
    def answered(kind, reply):
        def answer(_parent, title, text, *_args, **_kwargs):
            logging.info(f'capture: {kind} {title!r}: {text!r} -> {reply}')
            return reply
        return staticmethod(answer)
    for kind, reply in (('question', QMessageBox.Yes), ('information', QMessageBox.Ok),
                        ('warning', QMessageBox.Ok), ('critical', QMessageBox.Ok)):
        setattr(interface_module.QMessageBox, kind, answered(kind, reply))

    # a box with its own buttons: closed unanswered, as with the window's close button
    def closed(box):
        logging.info(f'capture: box {box.text()!r} -> closed')
        return 0
    interface_module.QMessageBox.exec = closed

    viewer = napari.Viewer()
    # fully on screen: the activity dialog sits at the window's bottom right
    viewer.window._qt_window.move(0, 0)
    viewer.window._qt_window.resize(1500, 950)
    # on top of the IDE that started it, or the screenshots show the IDE
    viewer.window._qt_window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    viewer.window._qt_window.show()
    _, widget = viewer.window.add_plugin_dock_widget('muvis-align')
    interface = widget.interface
    dialog = viewer.window._qt_window._activity_dialog

    def capture():
        index = 0
        while not state['done']:
            seconds = time.monotonic() - state['t0']
            try:
                ImageGrab.grab().save(os.path.join(
                    args.output_dir, f'{index:03d}_{seconds:05.1f}_{state["step"]}_dlg{dialog.isVisible()}.png'))
            except OSError as error:
                # no screen to grab for a moment (locked, secure desktop): skip the frame, keep capturing
                logging.info(f'capture: screen grab failed ({error})')
            index += 1
            time.sleep(args.interval)

    def run():
        interface.project_path(args.project)
        if args.action != 'open':
            state['step'] = 'open'
            interface.input_output_process()
        state['step'] = args.action
        state['t0'] = time.monotonic()
        threading.Thread(target=capture, daemon=True).start()
        if args.cancel_after is not None:
            # fired from the nested event loop the running operation keeps going, as a click would be
            QTimer.singleShot(int(args.cancel_after * 1000),
                              lambda: interface.process_or_cancel(lambda: logging.info('capture: nothing to cancel')))
        try:
            ACTIONS[args.action](interface)
            if args.pair_offset:
                # the moving image is the pair's second layer; its metrics follow once it has moved
                moving_layer = viewer.layers[-1]
                affine = moving_layer.affine.affine_matrix.copy()
                affine[-2, -1] += args.pair_offset
                moving_layer.affine = affine
        finally:
            state['step'] = 'finished'
            # waited for with Qt running, so the last frames show the result painted
            QTimer.singleShot(int(2 * args.interval * 1000), finish)

    def save_window():
        geometry = viewer.window._qt_window.frameGeometry()
        # a screen grab: Qt's own leaves the OpenGL canvas black
        ImageGrab.grab(bbox=(geometry.left(), geometry.top(), geometry.right(), geometry.bottom())).save(
            os.path.join(args.output_dir, 'window.png'))

    def save_tabs():
        size = widget.size()
        for index, label in enumerate(widget.tab_labels):
            widget.setCurrentIndex(index)
            QApplication.processEvents()
            # each tab at its own content's height: the dock stretches a short tab to the longest one
            widget.resize(440, widget.tabBar().sizeHint().height() + widget.currentWidget().sizeHint().height() + 8)
            widget.grab().save(os.path.join(args.output_dir, f'tab_{index}_{label}.png'))
        widget.resize(size)

    def finish():
        state['done'] = True
        if args.window:
            save_window()
        if args.tabs:
            save_tabs()
        # still alive two minutes after closing: show every thread's stack
        faulthandler.dump_traceback_later(120)
        QTimer.singleShot(1000, viewer.close)

    QTimer.singleShot(3000, run)
    napari.run()
    # ended here, not by the interpreter: its shutdown has hung in native code after napari closed (on Windows
    # even os._exit, which still runs the libraries' detach code - TerminateProcess does not)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == 'win32':
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0)
    os._exit(0)


if __name__ == '__main__':
    main()
