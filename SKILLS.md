# Working conventions for this repo

Distilled from recurring feedback across sessions. Follow these when making changes here.

## Comments
- Code comments: 1-2 lines max, even for non-obvious WHY reasoning. No multi-line rationale blocks.
- Never narrate what changed, reference the current fix/task, or mention specific callers/sessions in a comment - that belongs in the commit message or chat response, not the code.
- Comments explain WHY (a hidden constraint, a subtle invariant), never WHAT (identifiers already say that).

## Testing
- For a small/localized fix, run only the targeted test file(s) for the area touched - not the full suite (10+ min).
- Reserve full-suite runs for larger/riskier changes, or once before pushing a batch of combined fixes.
- Don't create a new permanent test file per change - add tests to the existing file matching the module under test (e.g. `tests/test_utils.py` for `util.py`). Only give a feature its own test file when it's substantial/self-contained enough to warrant one.

## Running the napari UI without user input
- To see what the plugin actually shows (progress bars, dialogs, layers), drive it from a script instead of asking the user: `testing/napari_ui_capture.py <project.yml> <shots_dir> --action open|pre_processing|pair_registration`.
- It opens the project through `Interface.project_path()` + `input_output_process()`, runs the action from a `QTimer.singleShot` on the Qt thread (as a button click would), and closes the viewer when done.
- Screenshots come from `PIL.ImageGrab` on a background thread, once a second - a Qt grab would stall along with a blocked Qt thread. Filenames carry the elapsed time, the current step and `dialog.isVisible()`.
- Make short steps long enough to watch with `--slow <Interface-module name>` (e.g. `make_msims_3d`, `Interface._update_view_add_shapes`), which sleeps before calling it.
- Keep the window fully on screen (the script moves it to 0,0): the activity dialog sits at its bottom right. Qt saying a widget is visible does not mean it is painted - check the image (view it, or compare a cropped region's pixel spread across frames).
- Test project: `C:/project/slides/muvis_align_project.yml` (328 tiffs, 5 sections).

## Code style
- Avoid `continue` statements - restructure the loop body (e.g. invert the condition) instead.

## Git
- Small, focused commits with a "why" in the message, not a changelog of "what".
