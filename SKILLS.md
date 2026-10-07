# Working conventions for this repo

Distilled from recurring feedback across sessions. Follow these when making changes here.

## Comments
- Code comments: 1-2 lines max, even for non-obvious WHY reasoning. No multi-line rationale blocks.
- Never narrate what changed, reference the current fix/task, or mention specific callers/sessions in a comment - that belongs in the commit message or chat response, not the code.
- Comments explain WHY (a hidden constraint, a subtle invariant), never WHAT (identifiers already say that).
- No performance statistics (timings, sizes, speedups) in comments - those go in the commit message or notes.

## Testing
- For a small/localized fix, run only the targeted test file(s) for the area touched - not the full suite (10+ min).
- Reserve full-suite runs for larger/riskier changes, or once before pushing a batch of combined fixes.
- Never assert a registration's accuracy (an expected shift from phase correlation/SIFT, or how close two solvers' fits come): platform numerics and RANSAC vary, so such tests fail at random. Stub the registration result with a known transform and assert the code's handling of it exactly; check input preparation by value; compare two deterministic paths for identical output.
- Before adding a test, check for an existing one covering the same path - extend or parametrize it rather than adding a near-duplicate.
- Don't create a new permanent test file per change - add tests to the existing file matching the module under test (e.g. `tests/test_utils.py` for `util.py`). Only give a feature its own test file when it's substantial/self-contained enough to warrant one.
- Test on the least data that exercises the change: a fusion rule needs a few sources at their metadata positions, not a registered project; repo `data/` or a subset glob before any full project.
- Plan one combined check per change instead of a run per question: decide up front what it must show (output, sizes, times), and reuse results rather than rerunning. Run a full project (e.g. meatballs) at most once, when a whole-run number (size, time) is the point.
- Keep written output small: no full-size fusions/exports unless their size or speed is what's measured, in the scratchpad, deleted once read.

## Running the napari UI without user input
- To see what the plugin actually shows (progress bars, dialogs, layers), drive it from a script instead of asking the user: `testing/napari_ui_capture.py <project.yml> <shots_dir> --action open|pre_processing|pair_registration`.
- It opens the project through `Interface.project_path()` + `input_output_process()`, runs the action from a `QTimer.singleShot` on the Qt thread (as a button click would), and closes the viewer when done.
- It answers every `QMessageBox` itself (question -> Yes, notices -> Ok, logged as `capture: ...`): a scripted run must never wait on a dialog. Cover any other dialog an action opens before running it.
- Screenshots come from `PIL.ImageGrab` on a background thread, once a second - a Qt grab would stall along with a blocked Qt thread. Filenames carry the elapsed time, the current step and `dialog.isVisible()`.
- Make short steps long enough to watch with `--slow <Interface-module name>` (e.g. `make_msims_3d`, `Interface._update_view_add_shapes`), which sleeps before calling it.
- Keep the window fully on screen (the script moves it to 0,0): the activity dialog sits at its bottom right. Qt saying a widget is visible does not mean it is painted - check the image (view it, or compare a cropped region's pixel spread across frames).
- Test project: `C:/project/slides/muvis_align_project.yml` (328 tiffs, 5 sections).

## Code style
- Avoid `continue` statements - restructure the loop body (e.g. invert the condition) instead.
- No single-letter variable names, comprehension variables included: `field for field in fields`, not `f for f in fields`.
- Reuse and extend existing functionality instead of writing a new module that parallels it (as `lazy_overview.py` and `fusion_slabs.py` did).
- No atomic one-line wrapper functions; keep code intuitive over clever.
- Weigh an optimisation against the code it adds: a ~15% gain rarely justifies a new class or module. Threaded/pooled efficiency code should take an argument to run without threading.
- For a restructure, write a detailed plan first and agree it before editing.

## Carrying work across sessions
- Record the current task, its plan and how far it got under "In progress" in `notes/todo_known_issues.md` before editing code, and keep it updated. Clear it once the task is done.
- On "continue" with no other context, read that section first and resume from it.

## Git
- Small, focused commits with a "why" in the message, not a changelog of "what".
- Work on `main` by default; create a branch only for major restructures or features.
- Never create git worktrees (no `EnterWorktree`, no `isolation: "worktree"` agents, no `git worktree add`).
