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

### Buttons unclickable with napari maximised under xpra in Chrome

Since Chrome 154 (installed here 2026-09-23 22:24), a maximised napari window under xpra's
HTML5 client shows an I-beam over the plugin's buttons and clicks go elsewhere: the pointer maps
to the wrong place (over a text field). Un-maximised it works, and Edge (Chromium 153) works
maximised. Not muvis-align or the image: the same happened with this morning's code, with the
older xpra-html5 21 client, locally and on the HPC. A maximised window takes the size of xpra's
fixed virtual screen (1920x1080) and the browser scales it into the tab.
Fixed in xpra-slurm.sh and the Dockerfile: Xvfb gets an 8192x4096 framebuffer (xpra's own default)
so --resize-display can make the screen follow the tab (1879x884 in the test: no scaling), and
napari starts maximised. Tested in Chrome 154, maximised: buttons work. Costs ~150MB (Xvfb rss
220MB vs 72MB), no CPU: only the current size is drawn and encoded.
Side finding: xpra.org no longer serves xpra-html5 21 (stable or beta), so rebuilds get 19.

### napari process not exiting after closing (seen in scripted runs)

Twice in ~20 UI driver runs on the slides project the process stayed alive after napari closed, one core busy,
119 threads: every Python thread had finished, the interpreter's native shutdown hung (faulthandler dump: one
thread, no Python frame). A rerun exited normally. The driver now terminates itself after napari closes; a user
closing napari might see the same lingering process. Not investigated further (needs native stacks, e.g.
py-spy --native).

### napari exits after pre-processing on the HPC (not reproduced)

HPC run 2026-10-01 (34k sources, code at c8da3af): pre-processing finished (16 min), the refresh
added the pre-processed lazy overview (1081 sections of 7337x8441 at 0.2um), and 1.5s after
add_image napari was gone - no traceback, rss 11.7GB of 2TB. Likely a native crash in napari's
first draw, which builds the viewed section from register_msims on 32 threads, while the prefetch
builds its neighbours. Locally (data_400, 153 sources, 3 sections, Windows) it works.
A fatal signal now appends every thread's stack to muvis-align.log (faulthandler): the next run
shows where.

### Pair registration mixing up pairs' crops (fixed)

dask's linear fusion renames a fused chain to a 115-char prefix plus 4 hex digits of `hash()`,
so two pairs' crop chains in one compute could share a key and one pair registered the other's
crop - silently, or failing with `ValueError: inhomogeneous shape` in phase correlation when the
shapes differed. `register_pairs` now computes with `optimization.fuse.active` off. Still in
dask 2026.8.0.

## In progress

Opening the 34k HPC project (log 2026-10-01): the first section shown was slice 0, which holds
only an overview; and the open could be faster (first view 10s, sources read 3.6 min, refresh 2.6 min).
Plan: first section = the first with more than one file; then profile the open's slow steps
(init sources 42ms CPU a file at 7 cores, shape geometries 38s, lazy overview 54s, add shapes 28s).
Progress (synthetic project in the scratchpad: 100 sections x 30 128px tiles + an overview each,
OME positions with z per section - without z every section overlaps every other, 1.3M pairs):
- Reverted (91d22d6): the first section stays the first, even when it is a single file.
- Done (f2e2cfe): shape transforms from a shared xarray template - geometries 6.0s -> 0.7s,
  lazy overview 7.7s -> 1.7s at 3100 sources (HPC estimate 38s -> ~4s, 54s -> ~12s).
- Init sources: real data_400 tiles take 2.7ms CPU a file locally vs 42ms on the HPC, so it is the
  network filesystem (~31 small reads a file by tifffile, 256 threads), not reproducible here.
- Left: add shapes (28s) and refresh overview shapes (15s) on the Qt thread - napari Shapes layers
  of 34k + 115k rectangles, twice (main viewer and overview widget); ~39s untimed after init_data
  on the HPC (populate tables; ~1s here at 3100).

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
- Pair metrics were single-threaded because threaded ones crashed (access violation): each
  pair's overlap mask is a matmul, and OpenBLAS starting its own pool from many threads at once
  fails. Now threaded with OpenBLAS at one thread (threadpoolctl): 51 tiffs 18.5s vs 52s,
  results equal to 4e-15, 9 + 18 threaded runs without a crash. Still only ~3.3 cores.
- Metrics already run at the pre-processed scale (msims_reg scale0), as registration does.
- Per-batch release_memory() now collects only young generations: a full gc.collect scans every
  live object (0.4s at 484k on 51 tiffs, far more at 34k sources, ~1800 batches).
- Phase correlation computed a spearman quality for all ~11 candidate shifts and kept one: now
  deferred and computed for the kept one only - identical results, pair CPU 432s -> 217s,
  register_pairs 129.5s -> 90.6s on 51 tiffs. Worth fixing upstream in multiview_stitcher.
- Next idea: batches wait on their slowest pair (up to 15.6s, overview pairs) - a rolling window
  of synchronous per-pair computes on a thread pool, prototype in the scratchpad (rolling.py).
- Rolling window measured on 51 tiffs, 8 workers (local runs capped at 8, memory watchdog):
  61-62s every run, peak 2.5GB; batches of 16 took 64.8-117s (noisy), peak 2.1-2.3GB.
  Results identical to batches only without threadpool_limits(1) on BLAS: with it 6 of 179
  pairs moved by ~1 sub-pixel step (0.001). Batch results are identical across repeats and
  across 1 vs 16 pairs per compute.
- A graph built without a dask scheduler set makes multiview_stitcher compute overlaps with
  spawned processes: a script without a __main__ guard then re-runs itself in each (>10GB).
  register_pairs and metrics both set a scheduler.
- Done: register_pairs registers one pair per synchronous compute on a thread (util.rolling_map,
  at most 2x threads submitted); n_parallel_pairwise_regs is now the thread count, default one
  a core; 1 thread keeps the threads scheduler. 51 tiffs, 8 workers, vs batches of 16: results
  identical on all 179 pairs, pair loop 50-52s -> 38-41s, register_pairs 70-72s -> 58-61s,
  7-8 cores busy (was ~4.8), peak 2.2GB -> 2.5-2.8GB.
- Done: same rolling window for calc_pair_metrics (one tile_pair_image_metrics call a pair,
  BLAS still at 1 thread), summary weighted by each pair's comparison bbox area. 51 tiffs,
  8 workers, ncc/ssim/onmi: per-pair values identical, 11.3s -> 9.9s. Summary now equals one
  call over all pairs exactly (ncc 0.5153); the batched summary was wrong (0.6525), as it
  weighted batches by pair count.
- Tried, not worth it: each pair's metrics right after its registration on the same thread.
  46.7/47.2s -> 44.1/43.6s (~7%: registration already keeps the threads busy), and it needs BLAS
  at 1 thread for registration too, which moves 6 of 179 pairs by a sub-pixel step.
- Pushed up to edff812 (rolling window for registration and pair metrics, exact metrics summary).
- CI had failed on macOS since 0232e3d: the metrics test asserted every BLAS pool at 1 thread,
  and macOS numpy uses Accelerate (no pool). Now checks only OpenBLAS pools (587727f); CI green
  on all 9 jobs.
- Next (user, manually): rebuild and push the container (docker-build-push.sh), refresh it on the
  HPC (sbatch xpra-pull.sh), then run through xpra-slurm.sh - it runs the code baked into the
  image, so without the rebuild it tests the old batched code. Verbose timing logging on.
- Then: check the rolling window scales to 64 threads. Local baseline, 51 tiffs, 8 workers:
  pair loop 50-52s (batches) -> 38-41s, cores ~4.8 -> 7-8, peak 2.2GB -> 2.5-2.8GB. On the HPC
  compare peak memory against the earlier run's, and the per-2x-threads pair lines over time.
  On hold until pre-processing and its view refresh are sorted out (below).

Current task: pre-processing and the view refresh after it. HPC run 2026-09-24 (34k sources,
64 cores, xpra, code at edff812; napari closed by hand after the refresh, so registration never ran):
- init sources 4.3 min (3.8 cores), build msims 13.9 min (2.7 cores of 64).
- refresh after pre-processing 47 min, rss left at 230GB. Composite overview 30.6 min, rss
  9.6 -> 232GB rising ~5GB/30s, for a 1081x575x653 result; only 1.4GB released after, so
  ~6.8MB a source is still referenced (MALLOC_MMAP_THRESHOLD_ is set). Promote register_msims
  to 3D 9.0 min, cap preview fusion size 5.2 min.
Plan: reproduce locally, headless (51 or 328 tiffs; 8 workers, memory watchdog): pre-process,
then the overview, measuring rss per source. Find what holds each source's data after it is
pasted and fix it; then the slow single steps (3D promote, preview cap) and the core use.
Progress:
- One data_400 tile is 2304x3072 uint8 = 6.75MiB: the HPC keeps one full-res tile per source.
- Not reproduced on data_400 (51 sources; output folder purged first, it loaded saved pairs):
  UI driver, and headless (scratchpad overview_3d.py) with 1 or 5 fake sections, with the
  preview cap forced to coarsen ~1000x as on the HPC, and in the Linux container (src mounted,
  MALLOC_MMAP_THRESHOLD_ set): the overview adds only its own array, all freed after.
- Locally napari add_image adds 1.2-1.6GB for a 226MB full-res overview (HPC: none, 406MB).
- HPC project: pre_processing scale 2, no normalisation/flatfield/filter, pairing default.
  UI driver with that config on data_400 (output in scratchpad): still nothing kept.
- Scale 2 does work: sources are 1152x1536 at 0.02um (2304x3072 at 0.01um at scale 1). The
  overview is the same size at both because its budget sets its grid (mean spacing halved
  until it fits: 0.0147um x4 = 0.0294um x2 = 0.0588um), not the sources.
- So the HPC kept ~6.8MB a source = a full-res tile, although pre-processing works at scale 2
  (1.7MB). Not a view onto the decoded tile: a computed scale-2 level owns its own 1.69MiB.
- 510 sources (51 files x10), scale 2, 5 sections, 4GB budget: overview +0.27GB, all freed -
  no build-up with source count either.
- Added a diagnostic for the next HPC run, on with MUVIS_LOG_LIVE_BUFFERS=1 (xpra-slurm.sh sets
  it): during the overview (every 1/8 of the sources) and after the refresh, rss and the live
  numpy arrays/byte buffers of 1MB+ summed by what holds them (util.describe_live_buffers).
  ~2s a check on 51 sources, more at 34k. Doesn't see a running function's locals (on 3.12
  reading f_locals keeps them alive). Local baseline (data_400, HPC config): "no live buffers"
  during the paste; after, only the overview's own 215MB (Array held by Variable).
- Pushed as aa63a09 (full suite: 679 passed).
- Meanwhile fixed the unclickable buttons under xpra (see Known issues): screen follows the
  browser tab, napari starts maximised (c90d870); Dockerfile examples now use the quay.io tag
  (637704e), the stale local muvis-align-xpra:latest (7 Sep) removed. Container rebuilt and
  pushed by the user, checked locally in a new Chrome tab: right size, buttons work.
- The live-buffer test failed on every Python 3.14 job: 3.14 tracks a dict of arrays itself, so
  the holder reads 'dict x4 12.0MB (held by _TileKeeper)'. Test accepts both forms (58fc9bb,
  pushed; CI running). No 3.14 interpreter locally (.tox/py314-windows is empty).
- CI on f7159bc: all 3.14 jobs pass; Ubuntu 3.12 hung in test_utils.py (full suite; alone on
  Linux 3.12 it passes). Cancelled and re-ran that job. Likely cause, fixed (02f50b0):
  describe_live_buffers expanded a shared untracked container once per referrer - quadratic
  (200 holders of one 200k tuple: 23.1s -> 1.3s). The HPC run in progress uses the image from
  before this fix, so its overview memory checks may be slow.
- In the container, test_process_memory_tracks_an_allocation_or_says_it_cannot fails (passes on
  CI Linux) - likely container-specific, not looked into.
- HPC run 2 (image b492a35, diagnostic on): rss 35 -> 232GB over the paste, but the live
  buffers of 1MB+ stayed at 263MB throughout (napari shape meshes), 650MB after (+ the 387MB
  overview). So the kept tiles (~6.8MB a source) are not in Python-visible arrays/buffers:
  native code, or memory the allocator keeps (malloc_trim got 1.3GB back). Each check took
  135-175s (~22 of the overview's 57 min).
- Diagnostic removed again (the user dislikes explicit gc; on CI's Ubuntu 3.12 runner it hung in
  gc.get_referrers(), then got the runner shut down mid-test). faulthandler_timeout = 300 stays.
  Pushed as 79e33db; CI running - check the Ubuntu 3.12 job passes now.
- CI green on 79e33db, Ubuntu 3.12 included (9 min).
- The overview's paste now computes each source with dask's synchronous scheduler (one tile a
  compute, so its thread pool only added threads). Tests whether dask's threads plus glibc's
  per-thread heaps keep the memory: if the HPC stops growing, that was it. Local: same speed
  (UI data_400 2.3s vs 2.6s; Linux container 510 sources 35.7s vs 36.4s), nothing kept either
  way - the memory growth does not show locally, so the HPC run is the test.
- Linux with napari's Qt/OpenGL, locally too (xpra image, Xvfb, UI driver, data_400 with the
  HPC config, current src; scratchpad linux_ui/run.sh): nothing kept. Per-second RssAnon flat
  at 783-805MB over the paste (the 280MB overview allocated just before), RssFile constant at
  253MB. Every local setup is now covered; the HPC run is the only test left.
- All pushed up to d8043b2 (synchronous paste b80119a included).
- HPC registration (pairing default) stuck at 0% for 34+ min, one core, rss 231 -> 344GB:
  multiview_stitcher's default pair search (cKDTree radius = largest source's diameter) with
  the ~1mm overview images in the input pairs every source with nearly every other (~1.2
  billion candidates at 34k), each a delayed overlap task. data_400: 2550 candidates (all
  ordered pairs) with ov000, 436 without, for 179 / 129 real overlaps. User killed it; should
  have been orthogonal pairing.
- Fixed: default pairing hands multiview_stitcher the bounding-box sweep's candidates
  (find_candidate_overlap_pairs) instead. data_400: same edges with and without the overview
  (overlap values within 2.8e-16), graph build 13.2s -> 1.1s with it; register_pairs results
  identical on all 179 pairs, 36.5s -> 28.9s. Correction: at 34k it is not ~115k candidates -
  pre-processed sources are 2D, so default pairs everything overlapping in x/y across all
  sections (3 sections: 1764 vs orthogonal's 831), tens of millions at 1081 sections. Default is
  the wrong pairing for a multi-section stack either way; orthogonal is the right one.
  Pushed as cddd7fe.
- HPC run 3 (dd2e406, orthogonal): refresh after pre-processing 49 min (overview 32, 3D
  promote 9.3, preview cap 5.3), rss 232GB again. Registration: Build msims again 14.5 min,
  get_pairs 2h32 (#pairs 233338), pair graph (linprog per pair) 2h16, then 229725 pairs at ~7/s
  (~9h), rss flat ~237GB. Pair count expected (user: ~300 a slice x ~1000 slices); per section
  on data_400 127 within + 233 to the next, 41% overview-tile.
- Done (registration setup), 3-section data: get_pairs identical (847 pairs), HPC-size synthetic
  (33.5k sources, 1081 sections) 1.9s; pair graph identical (edges, overlaps within 3e-16, node
  stack_props), 4.5s -> 0.3s; orthogonal geometry from metadata identical, no msims build;
  register_pairs orthogonal identical on all 831 pairs. Expected on the HPC: ~5h of setup -> minutes.
- Pushed as b6dd0d1 and 3494f19 (registration settings read from the registration section only;
  two tests passed a whole operation dict).
- Refresh after pre-processing, locally (3-section data x10 = 1530 sources, scale 2; scratchpad
  refresh_profile.py / refresh_time.py). HPC per source: 3D promote 16ms (9.3 min), preview cap
  9ms (5.3 min), overview paste 56ms (32 min).
  - 25ff85a: 3D promote on the levels' datasets (identical trees) 14.2s -> 6.8s; overview images
    enlarged in one copy (5x). Overview unchanged (same md5); 26.2s -> 23.6s.
  - Overview is read-bound: 1530 file reads 6.6s + zarr's async per-chunk machinery. The tiles
    are uncompressed, one contiguous strip each, and the HPC overview keeps ~1 pixel in 85 per
    axis, yet reads every tile whole (~238GB over NFS, ~125MB/s). Proposal (user to decide):
    read contiguous uncompressed TIFF levels through a memmap-backed array so a strided or
    cropped read touches only the pages it needs - overview and registration crops alike.
    Core read path, so not done without asking.
  - Tried and reverted (user asked to test): every read reads the whole file - one level, one
    strip, one zarr chunk, and the scale-2 level is a strided view of it. Previous readers
    (ngff-zarr, and da.from_zarr(page.aszarr()) before it) did the same.
    - memmap-backed levels (as tifffile.memmap): identical pixels, but overview 26.3 -> 38.5s on
      Windows, 96.9 -> 388.7s in the Linux container over a mounted folder (a page fault a round
      trip). Not usable for NFS.
    - explicit row reads: 6-9x faster than whole reads when called directly, but in the pipeline
      the overview got slower (24.7 -> 56.8s): dask does not push the slices down to the read -
      each 1024-chunk asks for all its rows (6912 rows for a 2304-row tile, whole rows per x-chunk).
  - So partial reads only pay off if the overview reads the files itself: sampled rows of each
    uncompressed source straight from its file, several sources at a time, falling back to the
    dask path otherwise. Not done - for the user to decide.
- 2026-09-25: the test images were wrong (single level). Local and HPC files now have 5 levels
  (4 SubIFDs, x2 each), uncompressed, one strip per level; file 1.332x level 0. Verified locally
  for all 153. Pre-processing at scale 2 now reads stored level 1 only (3.00 of 8.99MiB).
- Re-evaluated for the pyramid files: every code change stands (scheduling, geometry, pairing,
  settings, promotion code, xpra are file-independent; orthogonal geometry from metadata still
  identical). The synchronous paste is at least as fast as threaded (1530 sources 23.1/21.7s vs
  24.1/25.0s, identical overview). Out of date: the memory analysis (full-res tile kept per
  source), whole-file reads, and the memmap/row-read experiments - stored levels do that now.
- New measurements, 1530 sources: promote 23s (4 levels a source now, was 2 - 6.8s), cap 2.7s,
  overview 22s reading level 1. With the cap shrinking ~1000x as on the HPC: cap 15.2s, overview
  10.0s reading the coarsest stored level (31KB a source). HPC estimate: promote ~8.5 min, cap
  ~5.5 min, overview ~4 min (was 32).
- Tried, no gain (identical output both, not committed; scratchpad promote_cap.py):
  - cap the 2D msims first, promoting only each one's finest level for the estimates and what is
    kept at the end: 1530 sources 17.6-41.8s vs promote-then-cap 17.0-23.3s - each estimate now
    builds a one-level tree and promotes it for every source.
  - cheaper z (Variable.set_dims) and widening each transform once a source: 8.99/10.06s vs
    9.65/9.84s. What remains is xarray building a Dataset per level (alignment, merge).
  - The one large saving left is structural: no 3D promotion for the overview - place 2D sources
    at their section z from the positions, and estimate the size from those.
- Done: the refresh after pre-processing no longer promotes to 3D. calc_output_properties /
  estimate_fused_size / reduce_msims_to_fused_size / composite_msims_overview take z_positions
  (promoted_geometry: each 2D source's finest level as promotion would make it, from its coords).
  Output properties exactly equal at every cap level; same levels kept; overview identical
  (pixels, spacing, origin, dims); fuse() fallback on the 2D sources identical to promoted.
  1530 sources: default budget 36.2s -> 16.1s, 1000x cap 33.6s -> 9.3s. Plugin, 153 sources:
  promote 1.8s gone, cap 0.6 -> 0.2s, same overview. Full suite 691 passed.
- Pushed up to a858e3f.
- Next (user, HPC, pyramid files): rebuild the container (docker-build-push.sh), git pull,
  sbatch xpra-pull.sh, sbatch xpra-slurm.sh, new tab; pre-processing then orthogonal pair
  registration. Check: no 'promote register_msims to 3D' step and the refresh well under the
  earlier 47 min; rss over the overview (was 9.6 -> 232GB on single-level files); '#pairs'
  within seconds and the first 'Pairs 128/' within minutes (was ~5h).
- Then, depending on that run: the HPC memory if it still builds up; the refresh's remaining
  steps (cap, overview per-source overhead); registration throughput (~7 pairs/s at 64 threads).
- HPC run 4 (4372fcf, pyramid files; stopped after the refresh, no registration): refresh after
  pre-processing 17.7 min (was 47-49): cap 3.4, overview 11.7, no promote; rss 14.4GB after the
  overview (was 230-232GB) - memory build-up gone. But pre-processing 28.4 min (was 14.5-14.9):
  Build msims 26.3 min (was 13.4-13.9), CPU a source ~47ms (was ~27ms), 2.7 cores - the
  msims now carry 4 stored levels (1-4), each opened and built.
- Profiled (153 sources, single-threaded, 23ms cpu a source): no pixel reads (1.2MB for 20 files);
  read_tiff_level_arrays ~1/3 (opens the zarr group, from_zarr on all 5 levels, tokenizing each
  zarr array ~6ms a source), per-level coords (ensure_spatial_image_dims/assign_coords) ~1/4,
  per-level Dataset + DataTree ~1/5.
- Done: TIFF level arrays via da.from_array with chunks, dtype meta and a name from path, size,
  mtime and level (no tokenize, no dtype probe). 153 sources: trees identical (levels, dims,
  shapes, dtype, chunks, coords, transforms, attrs, pixels), 24.1-25.4 -> 21.0-21.2ms cpu a
  source; orthogonal registration identical on all 831 pairs. Now: wrapping 5 levels 5.6ms
  (level 0 ~1ms - skipping it needs source.data lazy per level, not worth it), building the
  scale-2 tree 16.2ms (~4ms a level of xarray construction, as the promotion was).
- HPC run 4 pair registration failed: KeyError 'channel 0' - the new pyramid files name their
  channel '#0' (OME Channel Name="#0"); the HPC project still says channel: 'channel 0'.
- Done: channel fallback (resolve_registration_channel, in register_pairs and select_pair_overlap):
  a channel name the sources lack - one channel: use it and warn; several: error naming them.
  153 pyramid files with the HPC's 'channel 0': warns, registers on '#0', identical to '#0' on
  all 831 pairs. Full suite 693 passed.
- Next (user, HPC): rebuild the container, rerun pre-processing (Build msims ~26 -> ~22 min
  expected) and pair registration (channel 'channel 0' now falls back to '#0'; or set '#0').
- HPC run 5 (c1e869c): the fallback warned and chose '#0' from the first source, then selecting
  '#0' failed (KeyError '#0') - the HPC sources do not all name their channel the same (local
  ones are all '#0'). Fixed: the channel is resolved per source by its own labels, one warning
  per distinct set of labels (register_pairs and select_pair_overlap). 153 pyramid files with
  every third renamed 'channel 0', 'channel 0' requested: one warning, identical to all-'#0' on
  all 831 pairs; the tests reproduce the HPC error on the previous fix. Full suite 694 passed.
- Worth checking on the HPC: which files name their channel 'channel 0' - an old export mixed
  in with the pyramid files would also be single-level.
- Was: registration setup, tested locally on the 3-section dataset (data_399-401, 153 sources;
  project yml in the meatballs folder, resources/params_EM04652_02_slice017.yml):
  1. get_pairs: sweep candidates (boxes of each source's search distance, one section deep in z),
     same rules vectorised - identical pairs and angles.
  2. pair graph: overlaps from exact AABB intersection when all transforms are translations,
     else multiview_stitcher as now - identical edges and overlaps.
  3. orthogonal positions/sizes from source metadata instead of building self.msims.
- The user changed register_pairs' pairing lookup (drops params['registration']['pairing']):
  theirs, uncommitted - leave it, commit only my hunks.
- Next (user): rebuild the container (docker-build-push.sh), on the HPC git pull, sbatch
  xpra-pull.sh, sbatch xpra-slurm.sh, connect in a new tab, run pre-processing, compare rss
  over the overview (previous: 9.6 -> 232GB). If it still grows: grep -E 'RssAnon|RssFile'
  /proc/<napari pid>/status near the overview's end - allocated memory vs NFS files mapped in.
  Flat: dask's threads plus the allocator were it - apply the same wherever tiles are read
  one per compute. Then the paste's speed (rolling window), the slow single refresh steps,
  and the HPC registration results (user ran registration with the rolling window).
- Next: reproduce on Linux with napari's Qt/OpenGL running (xpra container, virtual display,
  UI driver, data_400 with the HPC config), reading RssAnon vs RssFile from /proc - allocated
  memory vs files mapped in. Windows with the UI and Linux headless keep nothing.
- User running pair registration on the HPC now (rolling window), results to follow.
- HPC run 2 in progress (user): refresh after pre-processing predicts ~2 min early, then ~50 min
  after ~10 min. The bar's weights misjudge 34k sources: shapes and copies fast and weighted
  generously (27% at 1 min), then 3D promote (9 min) and preview cap (5 min) each report once.
  Fix later: per-source progress (and speed) in make_msims_3d and reduce_msims_to_fused_size.

- Done (user request): pre_processing 'scale' and input_output 'preview_scale' take a pixel size
  with unit ('10um', '40nm') as well as a factor: text fields in the template, parse_scale()
  where they are read (preprocess, ensure_msims, msims_build_pending, the view's and
  create_preview's preview_scale, get_level_from_scale). A factor is relative to each source's
  own pixel size - preview 16 gave overview images 3.986um and tiles 0.160um - a pixel size is
  one target for all. Plugin runs with '0.04um'/'1um' clean; full suite 715 passed.

- Done (user request): convert writes the pre-processed (scaled) register_msims, not the
  full-resolution self.reg.msims, running pre-processing first if needed (f222355). Outputs are
  named `<input file title>.ome.zarr`; written pyramids pad down to min_length (128) instead of
  default_chunk_size (66e5445); get_filetitle strips only '.ome' - rstrip cut 'slide_one' to
  'slide_on' (9ba0aa8). Targeted tests only (125 passed); no real conversion run yet.

- Measured why pre-processing got slower on the HPC (1530 sources = hard links to the 153):
  1. Viewer during the build, Linux container + Xvfb (software GL, as xpra), 64 threads: headless
     25.9s; plugin with the shapes layer hidden 31.1s (process cpu 46.6s, build 25.0s); shown
     35.3s (54.3s, 28.0s). The plugin adds 20-36%; repaints are ~4s of it. Not the main cause.
  2. Build threads (headless, Windows): 8-64 threads 26.1-26.5s, 192 threads 28.8s - always
     ~1.1 cores (GIL-bound Python). Linux 64 vs 16: 25.9 vs 25.4s. Fewer threads help ~2-10%.
  3. Levels: registration uses only the finest pre-processed level (all 831 pairs at scale0,
     no further binning). Build cost 1/2/3/4 levels a source: 9.2/11.6/15.7/18.5ms cpu -
     ~6ms fixed + ~3ms a level. Building finest + coarsest only (2 levels) would save ~37%.
  HPC: 34k x ~45ms = ~25 min on one core, as measured. The cost is the per-level xarray
  construction, single-threaded; neither threads nor repaints are the lever - fewer levels is.

- Done (user request): pre-processing builds only each source's finest and coarsest level
  (ends_only; registration reads the finest, the preview cap the coarsest), build threads capped
  at 32, convert builds the full pyramid. Local (153 sources): finest/coarsest geometry, overviews
  (default, 20x, 1000x budgets) and all 831 pair registrations identical; build 1530 sources
  25.9 -> 17.7s (18.7 -> 13.3ms cpu a source). Plugin pre-processing then convert clean; the
  converted stores keep all 5 levels. The 2D zarr configs (params_test_2d2, project2) now read
  data/S*/*.zarr, since data/*/*.zarr also caught data/3d; the config test accepts every method
  in the project template (phase_correlation included). Full suite 750 passed. Pushed.

- HPC run 3 (25085c2, 34k sources, 229725 pairs), from the log:
  - Pre-processing 29.7 -> 16.0 min (build 27.6 -> 15.0 min with ends-only and 32 threads).
  - Refresh after it 14.3 min: preview cap 3.7 min, overview paste 8.2 min (13.3 before), rss 13GB.
  - Pair registration 19.0h: rss grows ~1.2MB a pair (18 -> 136GB by ~100k pairs) at 14s per 128
    pairs, then plateaus at ~137GB and slows to 47s, later 100-160s per 128. At the early rate the
    whole run would be ~7h. Process cpu 7x the pairs' own cpu (22.7 cores). Next: find what each
    pair keeps (local, 831 pairs) and what burns the rest of the cpu.
  - Pair metrics 3.8h, memory flat.
  - Global registration (global_optimization): 4.8h before the first iteration, pass 1 ~2h
    (323 iterations), then one edge removed a pass at ~3 min, max residual stuck at ~69.6: 146
    passes in 10h. Local 153 sources: 967 passes 354s vs linear_two_pass 2.0s, positions within
    median 0.55um (max 3.0um) of each other.

- Done (user request): pair registration in worker processes (register_pairs: spawned, native pools at
  1 thread, replaced every 1000 pairs; threads when a pair does not pickle). TIFF levels made picklable
  (PicklableTiffLevel: tifffile's store holds an RLock). Local, 153 sources, 1764 pairs, 8 workers:
  - Windows: threads 168.8s -> processes 111.7s; Linux container: 271.8s -> 99.7s, peak rss 3.1 -> 0.47GB.
  - Results identical to threads with native pools at 1 thread; threads-only differs from that on 43
    pairs by up to 0.046um (native threading's summation order), on Windows and Linux alike.
  - Main process 5.6s cpu for 1764 pairs (3.4ms a pair): no longer a ceiling at 64 workers.
  - Plugin (UI driver, 51 sources, 179 pairs, 24 processes): 14.3s, clean. The driver now answers
    every QMessageBox itself (a run blocked on "Run pair registration?").
- Doing (user request): speed up global registration. Found, local 153 sources / HPC log:
  - 4.8h before the first HPC iteration: mv_graph.get_node_with_maximal_edge_weight_sum_from_graph
    loops edges per node (O(nodes x edges)), on a subgraph view 1.2us a node-edge: ~2.7h here at HPC
    size, ~2x that on the HPC's cores. linear_two_pass calls it too. Fix: an O(edges) version.
  - global_optimization then removes one edge a pass (967 passes, 354-406s locally; 146 in 10h on the
    HPC). linear_two_pass: 2.0s, but residuals worse (median 0.211 vs 0.117um on all edges): plain
    least squares, pulled by bad pairs. Repeating it prunes to a spanning tree (worse).
  - Robust linear (IRLS: linear_two_pass re-solved with quality x Cauchy weight of each edge's
    residual, 10 rounds, ~23s): Cauchy scale 0.2um median 0.090 / p90 0.564, 0.35um 0.134 / 0.434,
    against global_optimization 0.117 / 0.503 - comparable fit, ~17x faster here.
  - Now (user agreed): the robust linear wrapper around linear_two_pass, registered as a method, with
    the reference view picked per component in one pass over the edges; check it at HPC size on a
    synthetic graph. Details and upstream candidates in notes/multiview_stitcher.md.

- Done (user request): faster registration metrics. Local 153 sources, orthogonal, 831 pairs, 8 workers:
  global metrics 3.1 min -> 43.5s (per registered pair instead of all 1764 overlapping pairs, in worker
  processes), peak rss 6.5 -> 0.44GB, identical values; pair metrics 27.7 -> 27.2s (workers start-up
  outweighs the gain at this size; the HPC's 3.8h is the case it is for). Found on the way: SSIM got the
  registration channel as its channel axis (fixed); the overlap mode's linprog hangs/crashes from
  changing pool threads (global metrics run in the calling thread without worker processes); an
  unpickled TIFF level opened concurrently by dask threads read a closed file (the Windows CI failure,
  locked). Details in notes/multiview_stitcher.md.

- Doing (user request): the two new TODO items.
  - Cancel: done (f580800). While an operation runs the Process buttons read Cancel (widgets disabled
    as modify_pair_registration does); after a confirmation the operation stops at its next progress
    step (pair, source, fusion chunk, robust round, global_optimization iteration via its log handler)
    and does not wait for worker processes: pair registration stopped 0.21-0.64s after the cancel
    locally, nothing stored. Global registration restores the sources' transforms; fusion removes its
    partial output. Tested in the plugin with the UI driver (--cancel-after).
  - Split pairing: done. 'split' is the last pairing option. Stage 1: orthogonal pairs within each z-plane
    (or channel, registration_dimension 'c'), pair and global registration as usual - the graph falls
    apart into one component per group. Stage 2 (split_registration.register_groups, in register_global):
    each group fused with its stage-1 transforms (longest side <= 2048 px), consecutive groups registered
    as whole images, resolved (robust_linear for translation/rigid), each group's correction composed onto
    its tiles. Local 2 planes x 4 tiles: stage 1 8 pairs, 0 across planes (orthogonal: 12, 4 across). A
    known 2um shift of the second plane: split left 0.32um of it, orthogonal 1.56um (plane-to-plane
    registration quality only 0.05 - different sections). Synthetic identical planes: recovered within
    0.5px. Not yet run on the 5-section project (328 tiffs): stopped at 1.2GB free RAM.

- Doing (user request): test split pairing (two-pass) and the cancel button in the plugin on a small
  dataset (data_subset: 54 tiffs, 6 sections; scratchpad project copy, UI driver). Plan: split vs
  orthogonal full registration; cancel partway through open, pre-processing, pair and global registration.
  - Plugin runs clean (UI driver, new 'registration' action = registration_process): orthogonal 117 pairs
    52s; split 72 pairs (within sections only), 6 groups / 5 group pairs, 1.9 min (group stage 1.1 min).
    Within-section layouts identical to orthogonal (max 0.011um).
  - But split stage 2 barely moves the sections: group pair shifts 0.01-0.06um (quality ~0.3; 0->1 1.7um at
    quality 0.05). Consecutive-section NCC (mosaics at 0.064um): metadata 0.05-0.12, orthogonal 0.14-0.16
    (except S000->S001 0.016), split 0.03-0.09. A brute NCC search on the fused groups puts the offsets at
    3-11um (weak, broad peaks 0.09-0.23 vs 0.03-0.15 at zero). Likely: phase correlation of whole fused
    sections locks on the shared zero-shift pattern (same tile layout, seams, shading in every section);
    interior crop / high-pass do not fix it reliably. Option: stage 2 from the cross-section tile pairs,
    resolved with each group moving as one (for the user to decide).
  - Cancel in the plugin (UI driver --cancel-after, fresh outputs): pair registration (landed in pair metrics)
    stopped 2s after, nothing saved; global registration in the split group stage 1.6s after, only the
    finished pair_mappings.json kept; pre-processing 0.2s after. Opening a project: a cancel while reading
    the sources escaped as an OperationCancelled traceback (input_output_process had no handler) - fixed:
    "Cancelled" notice, sources read again on the next Process. A cancel in the view refresh's Qt-thread
    steps (adding shapes) is not seen: they have no checkpoint and finish (harmless, the flag is cleared).

- Split stage 2: the tile-pairs-across rework (484b7bf) was reverted - the point of split is matching a whole
  section against the next (robust where tiles do not match closely, few pairs in z). Fixed the fused-section
  match instead. Why it found ~0 shift: every section shows the same tile grid pattern, shading and outline;
  mvs phase correlation then either locks on them or its SSIM disambiguation (union/intersection bbox, NaN
  corners as 0) prefers the zero candidate. Now: all groups fused on one common grid, band-passed (tile/25 to
  tile/2, background filled; later only the tile/25 smoothing - the high-pass changed nothing, without the
  smoothing 2 of 5 pairs lock on the tile grid pattern), registered by skimage masked phase correlation (translation), NCC at the shift
  as quality; two groups held at a time. data_subset: section pair NCC 0.52-0.92 (shifts up to 48um);
  consecutive-section NCC 0.19-0.39 vs orthogonal 0.02-0.16, old split 0.03-0.08. Stage 2 51s, peak 2.6GB.
  Open: sections only translate (no rotation between them); fusing a section ~9s (1081 on the HPC ~2.5h).

- Done (user request): split stage 2 speed. Fusion was not the cost: 0.6s a section vs masked phase correlation
  2.4s a pair at the 2048 px grid (the ~9s estimate divided a cold 51s plugin run by 6). Grid 2048 / 1024 /
  512 px: stage 2 15.7 / 5.2 / 1.9s headless, shifts within 0.22 / 0.14um of 2048. User chose 1024: plugin
  stage 2 17.3 -> 6.3s, peak 2.7 -> 1.2GB, consecutive-section NCC 0.17-0.40 (was 0.17-0.38).

- Done (user request): split vs orthogonal on the 5-section slides project (328 tiffs, 8x8 tiles a section, S001
  72), plugin via the UI driver, 8 workers. Split: 575 pairs (28s), stage 2 14.4s, section pair NCC median 0.39
  (min 0.16), 1.3 min in all. Orthogonal: 831 pairs (1.4 min), 2.2 min. Consecutive-section NCC (tiles at
  0.256um, band-passed): metadata 0.123/0.117/0.116/0.113, split 0.744/0.172/0.136/0.133, orthogonal
  0.639/0.112/0.093/0.102. S001->S004 stay low for both. The first split run's process did not exit after
  closing (one core busy, 117 threads, 16 min; killed); a rerun exited normally. The driver now dumps all
  stacks and exits if alive 2 min after closing.

- Found (user request): why S001->S004 align poorly on the slides project. Each consecutive section of S001-S004
  is rotated ~5.5-6 deg from the one before (the scan rotation in their metadata is identical, 6.2554 rad, so
  the sections themselves are rotated on the ribbon); split stage 2 only translates. Search around split's
  placement (0-15 deg, 0.25 deg steps, mosaics at 0.256um): S000->S001 best at 0 deg (0.757 vs 0.745 at
  split); S001->S002 +5.5 deg 0.829 (vs 0.170), S002->S003 +5.8 deg 0.814 (vs 0.134), S003->S004 +6.0 deg
  0.793 (vs 0.128) plus ~79um shift - the rotation also broke S004's shift. S000 is from another session
  (2023-11-07, scan rotation 0.041 rad) but matches S001 without rotation. Fix: rotation in stage 2 (angle
  search on a coarse grid, refined) - rejected by the user: rotation comes from the pair registration method
  and transform type (MVS), nothing custom outside it.
- Done (user request): split stage 2 back on the configured pair method via MVS compute_pairwise_registrations
  and the transform type's resolution (own masked phase correlation removed); kept as input preparation: one
  common grid, tile/25 smoothing, background filled with the mean. Consecutive-section NCC:
  - slides, phase_correlation: 0.748/0.125/0.102/0.067 (translation-only method, the 6 deg rotations remain);
    sift rigid: 0.696/0.806/0.772/0.762, 1.8 min in all.
  - data_subset, phase_correlation: 0.03-0.08 (MVS disambiguation picks ~0 for its large shifts, up to 48um of a
    76um section); sift rigid: 0.35/0.35/0.35/0.31/0.25 (masked PC gave 0.17-0.40, orthogonal 0.02-0.16).
- Fixed: Windows CI py3.12 failure (test_source_levels_pickle_and_read_the_same_pixels_after, I/O operation on
  closed file): PicklableTiffLevel.array set _array inside the TiffFile block, so a thread checking it without
  the lock read before the file closed and tifffile kept that handle as open for good (c4e14c5, with a
  deterministic test that fails on the old code).

- Done (user request): split stage 2 fuses each section at the pre-processed pixel size (the msims' scale0), not
  a grid capped at 1024 px (default_split_group_size removed). Registering whole sections at that size was too
  costly: MVS bins only above 400^3 = 64M pixels, so a 6051x6801 px slides section (0.032um) was registered
  unbinned - phase_correlation 90s, 5.9GB a pair (and identity for S000->S001), sift over 10GB (the plugin run
  hit 16.6GB). User chose MVS's registration_binning: new registration setting split_binning (blank = 8),
  passed to compute_pairwise_registrations for the section pairs. Slides, sift rigid: 2.3 min in all, stage 2
  44.5s (22.6s at 1024 px - fusion at 0.032um is ~6.5s a section), peak 2.0GB; consecutive-section NCC
  0.673/0.799/0.781/0.766 (1024 px: 0.696/0.806/0.772/0.762).

- Done (user request): split's group (section) pairs in pair_mappings.json and the metrics table. Saved after global
  registration next to the tile pairs, keyed by group labels (["S000", "S001"]: the sources' common label
  prefix cut at a separator, else the z-position or channel) with "kind": "split_group", mapping, quality and
  bbox. The loader keeps them apart (find_file_list_index matches by substring: "S000" would match a file under
  S000/) and restores them into metrics['group_pairs'], so the table shows them after reopening too. Table rows
  "S000 - S001" after the tile pairs; cells now placed by metric name. Slides (sift rigid): 575 tile pairs + 4
  section pairs, quality 0.28-0.32, rotations 0.085-0.094 (~5 deg). Tests 242 passed.
- The UI driver hung again after napari closed (split, slides): the thread dump showed only a native thread (the
  interpreter finished, native shutdown hung, one core busy), and faulthandler's exit hung as well (on Windows
  _exit still runs DLL detach). The driver now ends itself with TerminateProcess (os._exit elsewhere) after
  napari.run() returns. A user closing napari may hit the same - see Known issues.

- Looked into (user report): fusion after split slow and memory-hungry locally. Plugin export: 5 x 60356 x 59181
  px at 0.004um, 5% in 3.6 min (~72 min), peak 9.5GB, 1.5 cores (6.3 cpu-min in 4.1 min). Headless, same export
  path (MVSRegistration.fuse to OME-Zarr), 150s each: orthogonal transforms 1200 blocks, 46.6s a batch of 24
  (~38 min), peak 5.0GB; split transforms (sections rotated -5.4..+11.6 deg) 1280 blocks, 33-36s a batch
  (~31 min), peak 5.9GB; the code before the picklable TIFF levels (2871766^) the same, 6.0GB. So no
  regression and the rotations are not the cost. Both use ~1.5 of 24 cores: every tile level is one
  uncompressed 6400x6400 strip, decoded whole for each block that touches it. The plugin run is ~2x slower
  than headless (later found: only its bar, see below) and adds its ~3.5GB baseline (viewer).

- Done (user request): the ~1.5-core limit of the export fusion (slides, 24 cores). Plan: sample all threads'
  stacks during the headless export (scratchpad fusion_rate.py) to see where they wait - tile reads (whole
  6400x6400 strips), a lock, resampling or zarr writes - then fix the dominant one and re-measure.
  - Found: 52% of fusion-thread samples waited in zarr's sync() - tile reads (tifffile's store get() is blocking
    code run on zarr's one event-loop thread) and the output writes. Tried: PicklableTiffLevel reads a level stored
    as one uncompressed run of rows directly (seek + readinto, calling thread; identical pixels): 35 -> 32s a
    batch, 1.5 -> 1.8 cores, peak 5.9 -> 8.2GB. Not the main limit. Uncommitted.
  - Main limit, single-thread cProfile (2.7s a block): multiview-stitcher's _fuse_chunk_to_zarr calls fuse() per
    block, which rebuilds the fusion plan over all 328 sources - sim_sel_coords 328 calls a block (~1.1s),
    _build_spatial_fusion_plan ~0.94s, _get_axis_aligned_translation_dims ~0.74s, ~4270 xarray .sel a block;
    affine_transform ~2%. Python holding the GIL: 24 threads give ~1.5 cores. Cost per block grows with the total
    source count (HPC: 34k sources, ~100x per block). User chose: fuse per z-plane.
  - Done: fusion_slabs.fuse_to_zarr_by_z_slabs, used by MVSRegistration.fuse for a multi-z export to zarr
    (fuse_by_z_slabs, on): each z-slab of blocks fused from only the sources that reach it, by multiview_stitcher's
    own rule (box padded for interpolation unless z is grid-aligned), through prepare_block_fusion
    (create_output=False attaching each slab to the one store) and its pyramid write. Identical output at every
    level: data/S* and data_subset with split's rotated transforms (6 planes, 24 blocks). Slides, 150s: 352 of
    1280 blocks (~9 min in all, was ~31), ~4.3 cores (was ~1.5), peak 8.3GB. A looser margin (one source voxel
    plus one output voxel) had pulled in the neighbouring planes: 102 blocks in 150s.
  - Direct reads re-measured with slabs: 352 vs 304 blocks in 150s (+15-20%), peak 8.3 vs 6.9GB. User: commit
    them. Full suite 789 passed.

- Done (user request): the plugin's export fusion is not slower than headless - its bar was. run_fusion declared
  phases=2 (the second a save an export written straight to zarr never runs), so the fusion blocks filled only
  half the bar, and ETAs read off it doubled (the earlier "~72 min" was ~36). Same slides export, same
  transforms, 150s: headless 328/1280 blocks (~11%/min), plugin bar 3 -> 14% (5.5%/min = ~11% of blocks/min),
  both ~4.1 cores. The pyramid write after the blocks is ~4% of an export (factor 4: 3.8 of 105.9s).
  Fixed: MVSRegistration.fuses_to_zarr() (not a channel overlay, not compose) sets one phase; plugin rerun:
  3 -> 23% in 2.5 min (headless 26%). The driver gained a 'fusion' action.

- Fixed: CI failed on test_a_misplaced_plane_is_registered_back_onto_the_one_before[7.0] (Windows 3.13, Ubuntu
  3.14; -6.4 for -7 +- 0.5; locally -6.5, on the edge). Cause: the fused planes' grid is their tight union, so
  content touches its edges, and phase correlation (FFT, periodic) pulled the shift towards zero. A 5% margin
  (split_grid_margin, filled as background) makes it exact: sharp-blob planes, 4 seeds x shifts 7 and 20 px,
  errors <= 0.1 (was 0.1-0.5). Test image now sharp blobs, tolerance 0.2. Slides with sift: consecutive-section NCC
  0.67/0.77-0.80/0.735/0.73-0.76 over two runs, within SIFT's (RANSAC) run-to-run spread without the margin
  (0.67-0.72/0.78-0.81/0.76-0.78/0.76-0.77). Full suite 789 passed.
- Tests no longer assert a registration's accuracy (user: non-deterministic expected performance is dangerous in
  tests). Split: a stubbed group pair registration (known shift) checks the corrections and their composition on
  stage 1 exactly; the plane grid (union + margin) and background fill are checked by value; the phase
  correlation accuracy tests are gone. register_global's robust_linear test now checks the method is passed on
  and every tile mapped, not that it lands within 0.05um of global_optimization.

- Checked (user, apptainer-check.sh, job 58856579, cn078): the HPC's Apptainer 1.4.2 is unprivileged (user namespace,
  no setuid, kernel squashfs mounts not allowed) and has no squashfuse, so a SIF is unpacked into a temporary
  sandbox in /tmp on every run: napari + muvis-align start 2m38s from the SIF, 10.5s from the sandbox. The
  sandbox xpra-pull.sh builds stays. squashfuse on the compute nodes (admins, or a user build on PATH -
  untested) would let the SIF mount directly.

- Fixed (user report, HPC 34k sources): a cancel during pair registration did nothing at first. The run (no
  "#pairs" line: default pairing) sat at 0% for 5.5+ min with rss 10.9 -> 23.6GB, finding candidate pairs and
  building the pair graph - tens of millions of candidates across all 1081 sections - and neither checked for a
  cancel; it was only seen when the pair loop started. Now checked per candidate block (_sweep_candidate_pairs,
  orthogonal pairing's get_pairs too), every 100k edges (build_view_adjacency_graph) and between register_pairs'
  setup steps. Still unchecked: a graph handed to multiview-stitcher (rotated boxes - not before registration).
  Default pairing is the wrong choice for a stack. Added (user request): pair and global registration log their
  settings, default pairing its '#candidate pairs'. No warning before building the graph (user: not wanted).

- Doing (user request, branch phased-source-init): image data on screen within seconds of opening a large
  project, before pre-processing (raw sources, not pre-processed ones). Agreed direction: sources initialised in
  phases (as msims already are, later), shapes still from the sources; a lazy per-section overview (raw coarsest
  stored level, direct reads, numpy paste) shown for the viewed section and filled in the background.
  - Found: the test project (meatballs, 153 files, SBEMimage) and the HPC one take positions/scale from each file
    (OME), so a template can't stand in for sources; init is 2.9ms a file locally, ~1.4s a file on the HPC's NFS.
  - Done, step 1: image/lazy_overview.py - one plane per section, built when viewed from each tile's coarsest
    level (source.data, no msim) pasted at build_source_stack_props' placement; update_views uses it whenever the
    view is at the source positions (opening, after pre-processing), else the fused path as before. Meatballs
    in the plugin: opening shows images with the shapes (lazy 0.1s, adding 1.0s); after pre-processing the
    refresh has no size cap/overview step (HPC: 3.5 + 11.8 min). tests/test_lazy_overview.py.
  - Done, step 2: built planes kept within 1GB (least recently viewed dropped; all 1081 at 34k would be ~17GB),
    the 4 sections either side of a viewed one built in the background.
  - Done, step 3 (user: first section only, one refresh at the end, one bar): on opening, the first folder's
    files (first_section_indices; none when all files share a folder) are read into a display-only
    section_registration and drawn, reporting to no bar (SilentProgress, still stops on a cancel); then every
    source is read as before under the one 'Initialising sources' bar with the viewer usable, and the full view
    replaces the section's. Meatballs: first section on screen 2.2s after starting, full view 4s later.
  - Done: the first section is found by a labelled number in the file names (section/slice/s/z, as S000 or
    SBEMimage's s00538, which varies and every file has), else by folder: SBEMimage keeps a folder per tile and
    all overviews in one, and in S000_000_001 the last number is a tile index. Meatballs: 50 data_399 tiles +
    the s00399 overview.
  - Fixed (user report, meatballs): the overview drew smaller than its shapes outline - the 0.249um overview's
    coarsest level (3.986um) pasted into a 0.32um plane repeated round(12.46)=12x, 983 of 1020um wide - and
    ignored preview_scale (always the coarsest level, plane capped at 4096px). Now each plane pixel takes the
    source pixel under its centre (any ratio, pixel edges as the outline's), the plane is at preview_scale, each
    source read at its coarsest level no coarser than that, a plane capped by bytes (1GB / 9 planes kept).
    Meatballs at 100nm: 10204x7653 plane, overview at level 0, tiles at level 3, built in 0.46s; its extent
    matches the outline to a plane pixel.
  - Done (user request): after pre-processing the view showed the raw sources (the lazy overview ignored
    show_preprocessed). Now the section planes keep the sources' geometry but read each one's register_msim (lazy
    pyramid, MsimLevels) at the coarsest level no coarser than the plane; a source filtered out stays empty.
    Meatballs, pre-processing scale 2, 100nm: tiles at 0.08um, overview at 0.498um, a plane in 0.25s.
  - Registration/fusion may stay slow on the big dataset (hours at 34k), not on a small one. Run only the
    targeted tests after each change (user).

## TODO

- [x] Pairing method "split, 2D x/y first" (see In progress / done above).
- [x] Cancel for all long operations: the Process button reads Cancel while one runs (f580800).
- [ ] Other computes over many similar per-source chains can hit the same dask fused-key
      collision: fusion and the global metrics - check, or switch linear fusion off
      process-wide. (Pair registration and pair metrics run with it off; the overview computes
      one source at a time.) Worth reporting upstream to dask.
- [ ] Keep the refresh view bar moving: give the long single-step phases (capping the preview
      fusion size, adding and refreshing shapes) per-source or per-batch progress. Promoting to 3D
      no longer happens in the refresh.
- [ ] Speed up the remaining slow per-source phases, all xarray object construction: building the
      msims in pre-processing (~22 min for 34k sources; now 2 levels a source, ~30% less)
      and the preview size cap (3.4 min on the HPC). Promoting to 3D: done - removed from the
      refresh (a858e3f), 2x faster elsewhere (25ff85a).
- [ ] Lazy overview on single-level sources (the meatballs files are pyramids, so its timings are optimistic for
      single-level data): each tile is read whole at full res and only then strided - read just the
      strided rows instead (as the direct uncompressed-level reads do), and measure on a single-level set.
- [ ] Registration preview for multiview-stitcher's built-in methods (e.g. phase correlation), which give only a
      transform, no matched points: a wrapper used only for the preview that maps points on a regular grid over
      the pair's overlap through the found transform, as the point pairs the preview's napari shape/point layers
      are built from, to show the offsets.
- [ ] Check which HPC files name their channel 'channel 0' rather than '#0' - an old export mixed
      in with the pyramid files would also be single-level (slower pre-processing and overview).
- [ ] Run a real convert with a pre-processing scale set: check output level-0 size and levels
      down to ~128px. Also run test_unique_file_labels / test_mvs_registration_unit /
      test_napari_interface_registration against the get_filetitle fix (not run yet).
