import numpy as np

from src.muvis_align._widget import (
    MainWidget,
)


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# you don't need to import it, as long as napari is installed
# in your testing environment
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    main_widget = MainWidget(viewer)

    # read captured output and check that it's as we expected
    #captured = capsys.readouterr()
    #assert captured.out == "napari has 1 layers\n"
