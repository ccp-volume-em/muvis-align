---
name: napari-ui-capture
description: Drive the muvis-align napari plugin from a script and screenshot it, to see what it actually shows (progress bars, dialogs, layers) without asking the user. Use when checking or debugging a viewer/UI change.
---

# Running the napari UI without user input

- To see what the plugin actually shows (progress bars, dialogs, layers), drive it from a script instead of asking the user: `testing/napari_ui_capture.py <project.yml> <shots_dir> --action open|pre_processing|pair_registration`.
- It opens the project through `Interface.project_path()` + `input_output_process()`, runs the action from a `QTimer.singleShot` on the Qt thread (as a button click would), and closes the viewer when done.
- It answers every `QMessageBox` itself (question -> Yes, notices -> Ok, logged as `capture: ...`): a scripted run must never wait on a dialog. Cover any other dialog an action opens before running it. To screenshot a box, `--show-boxes <seconds>` draws it (on top) for that long before closing it with the same answer.
- Screenshots come from `PIL.ImageGrab` on a background thread, once a second - a Qt grab would stall along with a blocked Qt thread. Filenames carry the elapsed time, the current step and `dialog.isVisible()`.
- Make short steps long enough to watch with `--slow <Interface-module name>` (e.g. `make_msims_3d`, `Interface._update_view_add_shapes`), which sleeps before calling it.
- Keep the window fully on screen (the script moves it to 0,0): the activity dialog sits at its bottom right. Qt saying a widget is visible does not mean it is painted - check the image (view it, or compare a cropped region's pixel spread across frames).
- Test project: `C:/project/slides/muvis_align_project.yml` (328 tiffs, 5 sections).
