from unittest.mock import patch

from muvis_align._widget import MainWidget


def test_main_widget_creation(make_napari_viewer):
    viewer = make_napari_viewer()
    with patch('muvis_align._widget.ViewerWidget'), patch.object(viewer.window, 'add_dock_widget'):
        widget = MainWidget(viewer)

    assert widget.viewer is viewer
    assert widget.interface is not None
    assert 'project' in widget.tab_labels
    # only the project tab is enabled until a project is open
    assert widget.isTabEnabled(0)
    assert not widget.isTabEnabled(1)


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


def test_a_tab_enabled_out_of_sight_is_scrolled_into_view_but_not_selected(make_napari_viewer):
    main_widget = MainWidget(make_napari_viewer())
    main_widget.resize(200, 400)
    main_widget.show()
    registration, fusion = main_widget.tab_labels.index('registration'), main_widget.tab_labels.index('fusion')
    main_widget.enable_tabs(True, registration)
    main_widget.select_tab(registration)
    changes = []
    main_widget.currentChanged.connect(changes.append)

    main_widget.enable_tabs(True, fusion)

    tab_bar = main_widget.tabBar()
    assert tab_bar.rect().contains(tab_bar.tabRect(fusion))
    assert main_widget.currentIndex() == registration and not changes
