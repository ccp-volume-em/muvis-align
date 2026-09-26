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

## TODO

- [ ] Other computes over many similar per-source chains can hit the same dask fused-key
      collision: fusion and the global metrics - check, or switch linear fusion off
      process-wide. (Pair registration and pair metrics run with it off; the overview computes
      one source at a time.) Worth reporting upstream to dask.
- [ ] Keep the refresh view bar moving: give the long single-step phases (capping the preview
      fusion size, adding and refreshing shapes) per-source or per-batch progress. Promoting to 3D
      no longer happens in the refresh.
- [ ] Speed up the remaining slow per-source phases, all xarray object construction: building the
      msims in pre-processing (~22 min for 34k sources with 4 stored levels each, ~4ms a level)
      and the preview size cap (3.4 min on the HPC). Promoting to 3D: done - removed from the
      refresh (a858e3f), 2x faster elsewhere (25ff85a).
- [ ] Check which HPC files name their channel 'channel 0' rather than '#0' - an old export mixed
      in with the pyramid files would also be single-level (slower pre-processing and overview).
- [ ] Run a real convert with a pre-processing scale set: check output level-0 size and levels
      down to ~128px. Also run test_unique_file_labels / test_mvs_registration_unit /
      test_napari_interface_registration against the get_filetitle fix (not run yet).
