"""Drive the napari plugin without user input and capture the screen while it works.

    python testing/napari_ui_capture.py C:/project/slides/muvis_align_project.yml shots \
        --action pre_processing --slow make_msims_3d --slow Interface._update_view_add_shapes

Opens the project, runs the action on the Qt thread (as a button click would), and saves a
screenshot every second from a background thread - Qt's own grab would stall with a blocked
Qt thread. Each file is named <index>_<seconds>_<step>_dlg<activity dialog visible>.png.
"""
import argparse
import logging
import os
import threading
import time

import napari
from PIL import ImageGrab
from qtpy.QtCore import QTimer

import muvis_align.ui.Interface as interface_module

ACTIONS = {
    'open': lambda interface: interface.input_output_process(),
    'pre_processing': lambda interface: interface.pre_processing_process(),
    'pair_registration': lambda interface: interface.pair_registration(),
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

    viewer = napari.Viewer()
    # fully on screen: the activity dialog sits at the window's bottom right
    viewer.window._qt_window.move(0, 0)
    viewer.window._qt_window.resize(1500, 950)
    _, widget = viewer.window.add_plugin_dock_widget('muvis-align')
    interface = widget.interface
    dialog = viewer.window._qt_window._activity_dialog

    def capture():
        index = 0
        while not state['done']:
            seconds = time.monotonic() - state['t0']
            ImageGrab.grab().save(os.path.join(
                args.output_dir, f'{index:03d}_{seconds:05.1f}_{state["step"]}_dlg{dialog.isVisible()}.png'))
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
        try:
            ACTIONS[args.action](interface)
        finally:
            state['step'] = 'finished'
            time.sleep(2 * args.interval)
            state['done'] = True
            QTimer.singleShot(1000, viewer.close)

    QTimer.singleShot(3000, run)
    napari.run()


if __name__ == '__main__':
    main()
