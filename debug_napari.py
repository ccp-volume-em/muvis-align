# Install your plugin in editable mode in your virtual environment.
# For example, you could do this by running pip install -e .
# in the root directory of your plugin’s repository.

from napari import Viewer, run


# the guard matters: registration's worker processes re-import this script, and would each open napari
if __name__ == '__main__':
    viewer = Viewer()

    dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget('muvis-align')
    run()
