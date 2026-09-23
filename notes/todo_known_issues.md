# Known issues and TODO

## Known issues

### Refresh view progress bar hidden while the viewer is empty

napari shows its welcome screen whenever the viewer has no layers, and that screen is drawn over
the activity dialog: the bar disappears although napari still reports the dialog as visible and
the work carries on (the log's heartbeat keeps reporting). Reproduced on Windows as well as
under xpra, with the slides test project (C:/project/slides) and delays added to the slow steps.

- Fixed for a refresh with a view already showing (e.g. after pre-processing): `update_views()`
  cleared the old layers before building the new view, which left the viewer empty - and the
  bar hidden - for the whole build, 44.6 minutes on a 33996-source project. It now clears them
  only once the new data is ready.
- Fixed for the first refresh after opening a project, when there is nothing to keep on screen:
  the welcome screen is switched off while an operation's activity dock is up
  (`VisibleActivityDock`), and restored afterwards.

The bar also stops moving (without disappearing) in steps that report once, when they finish -
promoting to 3D (8.9 min at 34k) and capping the preview size (5.2 min) - and while viewer steps
run on the Qt thread (adding and refreshing shapes, ~1.5 min), where nothing repaints.

Earlier fixes in this area: `16771aa` (bar froze partway, then filled and closed at once - each
off-thread call re-planned it from zero) and `96192ce` (the bar's repaint delivered queued input,
leaving the pointer grab stuck under xpra).

## TODO

- [ ] Keep the refresh view bar moving: give the long single-step phases (promoting to 3D,
      capping the preview fusion size, adding and refreshing shapes) per-source or per-batch
      progress.
- [ ] Speed up the slow single-step phases themselves: promoting 34k msims to 3D (8.9 min) and
      the preview size estimate (5.2 min) are both pure metadata/object construction.
