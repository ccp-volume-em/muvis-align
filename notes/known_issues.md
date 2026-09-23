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

## TODO

- [ ] Keep the refresh view bar visible and moving throughout: give the long single-step phases
      (promoting to 3D, capping the preview fusion size, adding and refreshing shapes) per-source
      or per-batch progress, and find out why the bar disappears rather than just stalling.
- [ ] Speed up the slow single-step phases themselves: promoting 34k msims to 3D (8.9 min) and
      the preview size estimate (5.2 min) are both pure metadata/object construction.
