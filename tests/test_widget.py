import numpy as np

from muvis_align._widget import (
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


def test_during_an_operation_only_the_process_buttons_work_and_read_cancel(make_napari_viewer):
    main_widget = MainWidget(make_napari_viewer())
    interface = main_widget.interface
    widgets = interface.param_widgets
    process = widgets['registration.registration_process'].widget
    pairing = widgets['registration.pairing'].widget
    method = widgets['registration.method'].widget
    method.enabled = False
    # as with a project open, on the registration tab
    main_widget.enable_tabs(True)
    main_widget.select_tab(main_widget.tab_labels.index('registration'))

    with interface._operation_widgets():
        assert process.text == 'Cancel' and process.enabled
        assert not pairing.enabled
        assert main_widget.is_tab_enabled('registration') and not main_widget.is_tab_enabled('fusion')

    assert process.text == 'Process'
    assert pairing.enabled and main_widget.is_tab_enabled('fusion')
    # disabled before the operation, so still disabled after it
    assert not method.enabled
