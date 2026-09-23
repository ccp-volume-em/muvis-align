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

### Pair registration mixing up pairs' crops (fixed)

dask's linear fusion renames a fused chain to a 115-char prefix plus 4 hex digits of `hash()`,
so two pairs' crop chains in one compute could share a key and one pair registered the other's
crop - silently, or failing with `ValueError: inhomogeneous shape` in phase correlation when the
shapes differed. `register_pairs` now computes with `optimization.fuse.active` off. Still in
dask 2026.8.0.

## In progress

Huge memory use on the HPC, where all tasks were effectively spawned at the same time instead of
a bounded number running at once. Test project (local):
`C:/Project/slides/EM04652-02_slice17_spaghettiandmeatballs2` (51 sources, 179 pairs).
Plan:
- Find which step schedules everything at once (pair registration/metrics batches, overview,
  preview fusion) and measure its peak rss on the test project.
- Bound the number of tasks in flight, then re-measure.
Progress (no memory changes yet):
- Pair registration on the test project: peak rss 2.8GB -> 4.8GB, threads ~100 -> ~410, of
  which only ~35 are Python threads. Not nested dask computes (only 2 top-level computes).
- Native pools (2x OpenBLAS, OpenMP, OpenCV) each default to the core count, so likely one set
  per dask worker - 64x64 on a 64-core node. Next: cap them to 1 inside the pair batches
  (`threadpoolctl.threadpool_limits(1)`, `cv2.setNumThreads(1)`) and re-measure.
- Local env updated to multiview-stitcher 0.1.62 (matches HPC). Measured there (24 cores):
  - native threads capped to 1: peak 4.67GB vs 4.73GB uncapped - not the cause.
  - pairs per compute 8 / 96 (default) / 179: peak 3.8 / 4.7 / 4.2GB, wall 1.7min / 54s / 54s.
    CPU ~8 of 24 cores at best (436s cpu in 54s). No blow-up with batch size at this scale:
    crops are ~200x200 at registration resolution. The HPC memory is not reproduced here.
- Measured before the fused key collision fix (see Known issues); re-measured with it,
  headless on 51 tiffs: pairs per compute 8 / 24 / 96 / 179 -> peak +1.1 / +1.7 / +2.0 / +1.9GB,
  wall 149 / 101 / 80 / 80s, 2.9 / 4.8 / 7.4 / 8.1 cores. Memory still flat with batch size.
- CPU (cProfile, 12 tiffs, single-threaded): ~0.9s per pair, mostly multiview_stitcher's
  `link_quality_metric_func` - spearmanr (16s of 31s) and SSIM (9s), ~11 candidate shifts per
  pair. Tile reads 0.8s, affine resampling 2.8s. Likely GIL-bound, hence ~8 cores at most.
- Done: pairs per compute capped at 2x cores (`default_pair_batch_size`), and
  register_pairs logs, when verbose: each batch's pairs, time and rss/peak; per-pair CPU vs
  wall (`format_phase_timing`); the scoring share (SSIM, spearman); pair metrics time.
- 51 tiffs, 48 per compute: 65.6s wall (80s at 96). Pairs 302s CPU of 683s pair wall, process
  8.5 cores; scoring 238s of the 302s (spearman 160s, SSIM 78s). Metrics 18s single-threaded,
  ~0.1s a pair - over 3h serial at 115k pairs.
- Peak in the batch lines is the process's lifetime peak, so a batch shows only if it raises it.
- Next: run on the HPC and read those lines.

## TODO

- [ ] Other computes over many similar per-source chains (fusion, overview, metrics outside
      register_pairs) can hit the same dask fused-key collision - check, or switch linear fusion
      off process-wide. Worth reporting upstream to dask.
- [ ] Keep the refresh view bar moving: give the long single-step phases (promoting to 3D,
      capping the preview fusion size, adding and refreshing shapes) per-source or per-batch
      progress.
- [ ] Speed up the slow single-step phases themselves: promoting 34k msims to 3D (8.9 min) and
      the preview size estimate (5.2 min) are both pure metadata/object construction.
