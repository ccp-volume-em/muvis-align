# Known issues and TODO

## Known issues

### Refresh view progress bar disappears for long periods, especially after pre-processing

On large projects the "Refreshing view" bar disappears (or stops moving) for minutes at a time,
most noticeably in the refresh that follows pre-processing, so the application looks stalled.

Seen on a 33996-source project (HPC log, 2026-09-22), in the refresh after pre-processing:

| Step (`_create_napari_data` / `update_views`) | Time     | Bar          |
|-----------------------------------------------|----------|--------------|
| promote register_msims to 3D                  | 8.9 min  | stuck at 12% |
| cap preview fusion size                       | 5.2 min  | stuck at 19% |
| composite overview                            | 30.5 min | moves (per source) |
| add shapes to viewer                          | 40 s     | stuck at 81% |
| refresh overview shapes                       | 52 s     | stuck at 88% |

Each of the stuck steps is one progress phase with `total=1`, so nothing is reported until it
finishes; only the composite overview reports per source. The first refresh after opening the
project shows the same pattern on a smaller scale (create shapes, add shapes to viewer).

Two ways the bar goes quiet in the current code:

- Off the Qt thread (`_run_off_thread`): promoting to 3D and capping the preview size report
  nothing until they finish, so the bar holds its position.
- On the Qt thread: adding the fused data and shapes to the viewer and refreshing the overview
  shapes touch napari, so they have to run there. While they run Qt processes no events and
  cannot repaint anything, the bar included - worse under xpra, where the display is remote.

Earlier fixes in this area, for context:

- `16771aa`: the bar froze partway and then filled and closed in the same instant ("stuck, and
  disappears before completion") - each `_run_off_thread` call re-planned the bar from zero.
  The same work moved the heavy steps off the Qt thread behind a nested event loop.
- `96192ce`: the bar's repaint (`processEvents()`) also delivered queued input, which under
  xpra left the pointer grab stuck; it now flushes paints only (`flush_paint_events()`).

Not yet confirmed whether the bar actually disappears (the activity dock or window not painted)
or only stops moving - worth checking under xpra against the log's heartbeat, which keeps
reporting the stuck percentage throughout.

## TODO

- [ ] Keep the refresh view bar visible and moving throughout: give the long single-step phases
      (promoting to 3D, capping the preview fusion size, adding and refreshing shapes) per-source
      or per-batch progress, and find out why the bar disappears rather than just stalling.
- [ ] Speed up the slow single-step phases themselves: promoting 34k msims to 3D (8.9 min) and
      the preview size estimate (5.2 min) are both pure metadata/object construction.
