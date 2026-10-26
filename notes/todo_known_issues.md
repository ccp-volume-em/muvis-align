# Known issues and TODO

Test projects: slides `C:/project/slides` (328 tiffs, 5 sections; the repo's `muvis_align_project.yml`,
12193 data), meatballs `C:/Project/slides/EM04652-02_slice17_spaghettiandmeatballs2` (153 pyramid tiffs,
3 sections, SBEMimage), Ciqtek `C:/Project/slides/Ciqtek`, data_subset (54 tiffs, 6 sections).
HPC: 34k sources in 1081 sections, xpra container (software GL).

## HPC runs compared

34k sources, 229725 pairs, 64 CPUs. Job 60804208 (2026-10-14) hit its 6h limit at 9% of pair registration (~10s a
pair); the 2026-10-12/13 logs are not kept (numbers in the TODO items below).
The 2026-10-17 run times as 2026-10-15; its refresh after fusion redrew the shapes twice on the Qt thread (fixed in
2d7c1cb). The 2026-10-22 run had a 160GB job (640GB before) and times as 2026-10-15/17: only fusion level 0.247 needs
the memory, and its peak came within 1.1GB of the job's limit (see Done). Job 61222229 (2026-10-21, b505577, 64GB)
timed as the 160GB run through level 0.16 (peak 38.8GB), then was OOM-killed 2.5 min into level 0.247 at 63.4GB.
The 2026-10-24 run (64GB, 64 CPUs, 1GB blocks of a 16GB pool) ran end to end at a 33.3GB peak, 0.3h longer
than the 160GB run: levels 0.01-0.02 are 22 min slower (smaller blocks, more of them), level 0.247 4 min faster.

| Task | 2026-10-09 (9eb38ac) | 2026-10-14/15 (b70c752) | 2026-10-15 (c1baed5) | 2026-10-17 (e57e425) | 2026-10-22 (b505577, 160GB) | 2026-10-24 (1b5b48b, 64GB, 1GB blocks) |
|---|---|---|---|---|---|---|
| Open: init sources + view | 2.4 + 1.2 min | 3.1 + 1.2 min | 2.6 + 1.3 min | 3.0 + 1.3 min | 3.4 + 1.3 min | 3.8 + 1.3 min |
| Pre-processing (msims build) | 16.4 min (15.4) | 17.2 min (16.2) | 16.8 min (15.8) | 17.3 min (16.4) | 17.7 min (16.8) | 17.4 min (16.4) |
| Pair registration | 5.1h | 3.5h | 3.5h | 3.5h | 3.2h | 3.4h |
| - setup (geometry, pairs, channel, graph) | 6.5 min | 6.7 min | 6.9 min | 6.8 min | 6.7 min | 6.9 min |
| - register pairs (workers busy of 64) | 4.1h (47.1) | 2.7h (37.4) | 2.8h (38.6) | 2.6h (37.1) | 2.5h (37.6) | 2.6h (38.0) |
| - pair metrics | 56 min | 45 min | 37 min | 41 min | 36 min | 41 min |
| Global registration (+ msims build) | 2.4h (+19 min) | 2.7h (+19 min) | 31 min (+18 min) | 33 min (+22 min) | 32 min (+18 min) | 34 min (+18 min) |
| - robust linear (8 rounds) | 1h23 | 2h26 | 17 min | 19 min | 18 min | 19 min |
| - apply + store transforms | 8.7 min | 9.5 min | 9.9 min | 10.0 min | 9.9 min | 10.7 min |
| - global metrics | 48 min | 0.5 min | 0.5 min | 0.5 min | 0.5 min | 0.5 min |
| Registration in all | 8.2h | 6.9h | 4.7h | 4.7h | 4.4h | 4.7h |
| Fusion | 9.8h | 9.4h | 3.4h | 3.5h | 3.4h | 3.7h |
| - before level 0 (make_msims_3d ~13 min) | ~16 min | 18.6 min | 19.0 min | 19.2 min | 19.2 min | 17.6 min |
| - memory peak | 248GB | 368GB | 152GB | 152GB | 159GB | 33.3GB |
| View refresh after fusion | 1.1 min | 1.2 min | 1.0 min | 6.9 min (shapes redrawn, twice) | 5.0 min (overlap shapes 1.4) | 4.7 min (overlap shapes 1.3) |
| Pre-processing + registration + fusion | 18.3h | 16.6h | 8.4h | 8.5h | 8.1h | 8.4h |

| Fusion level | 2026-10-14/15 | 2026-10-15 | 2026-10-17 | 2026-10-22 | 2026-10-23 (800MB blocks) | 2026-10-24 (256MB blocks) | 2026-10-24/25 (1GB blocks, 64GB) |
|---|---|---|---|---|---|---|---|
| 0.01 um | 2h37 | 1h46 | 1h53 | 1h46 | 2h16 | 3h41 | 2h02 |
| 0.02 / 0.04 um | 1h34 / 1h10 | 14.0 / 9.4 min | 14.5 / 10.0 min | 15.1 / 8.6 min | 20.6 / 9.9 min | 32.8 / 13.8 min | 20.7 / 9.2 min |
| 0.08 / 0.16 um | 43.5 / 40.4 min | 5.0 / 4.5 min | 5.0 / 4.5 min | 4.5 / 5.0 min | 5.1 / 3.0 min | 6.6 / 3.0 min | 5.4 / 2.8 min |
| 0.247 um (fused from sources) | 46.2 min | 34.9 min | 34.4 min | 34.9 min | 37.6 min | 72.3 min | 30.8 min |
| 0.494 um | 26.8 min | 2.5 min | 2.5 min | 3.0 min | 4.3 min | 5.8 min | 3.0 min |
| 0.988 - 15.8 um (5 levels) | 9.9 - 17.8 min each | 0.9 - 2.0 min each | 0.7 - 2.5 min each | 0.9 - 2.1 min each | 0.9 - 2.0 min each | 0.5 - 3.0 min each | 0.6 - 3.2 min each |

## Known issues

### Refresh bar stalls during single-step phases
It stops moving (without disappearing) in steps that report once, when they finish - capping the preview size
(minutes at 34k) - and while viewer steps run on the Qt thread (adding and refreshing shapes, ~1.5 min at 34k),
where nothing repaints. The bar hidden behind napari's welcome screen is fixed (old layers cleared only once the new
view is ready; welcome screen off while an operation's dock is up). See TODO.

### napari process sometimes lingers after closing
Seen twice in ~20 scripted runs: every Python thread finished, the interpreter's native shutdown hung (one core
busy). The UI driver now ends itself with TerminateProcess; a user closing napari could see the same lingering
process. Not investigated (needs native stacks: WinDbg's cdb is installed now).

### HPC draw crash: Mesa 22.3 worked around, root cause unknown
napari segfaulted drawing the pre-processed overview under xpra (Mesa 22.3 llvmpipe: draw_find_shader_output with
no vertex shader bound, from QOpenGLWidget::resizeEvent). Reproduced 100% in the Linux container, never in plain
napari. Fixed by Mesa 25 from bookworm-backports in the image (43ac613), confirmed on the HPC. Which GL program is
invalid at that draw was never found.

### Transform direction: synthetic test contradicts real data
On registered pairs multiview-stitcher's affine_matrix takes fixed crop pixels to the moving crop (10/10 meatballs
pairs by brute-force shift search; the pair NCC improves with it). A synthetic scipy-shifted pair suggested the
reverse. The preview grid follows the real data; the synthetic result is unexplained.

### Fusion pyramid levels use ~10-15% of RAM: expected
Not a leak, nor dask filling what is free. Measured on the 2026-10-03 run's pyramid step
(ngff_utils.write_sim_to_ome_zarr), one thread a CPU, each reading 2x2 (2x2x2 past 200 z) parent chunks and averaging
as float64: ~6-12 level-0 chunks a worker (measured locally, synthetic zarr, 1-8 workers). Its replacement
(fusion_blocks.average_into_zarr_array) does the same per block, 2x2 only, within the fusion memory pool (a
quarter of the job's memory), so the peak scales with the job: 33.3GB on a 64GB job.

## In progress

Nothing.

## TODO

ESSENTIAL

- [ ] 19:30 CET time close napari - long time to actually close node

MINOR

- [ ] **Elastix registration (a reference method)** - tested 2026-10-24 on data/S000 (2D) and data/3d with
      itk-elastix 0.25.4 from a side folder (not a dependency: pip also pulls a numpy of its own). Fixed: the
      3D access violation (the deferred link quality read Elastix's freed resampled image; only phase correlation
      defers now), and the stages now follow the project's transform type (translation, then rigid, similarity
      or affine; translation alone for translation). The rest is the data, not muvis-align: multiview_stitcher's
      own register() fails identically on data/3d's 13-voxel overlap ('Too many samples map outside moving image
      buffer') and scores data/S000's pairs as low (-0.07 to 0.5) - as SIFT finds too few points on some data.
      A pair Elastix cannot register now gets the identity with quality 0 and a warning, as the feature methods
      give for too few points: data/3d completes (27s), its 13-voxel pair skipped.
      Pairs scored 0 or below (failed: identity with 0; or a negative spearman, -0.20 on data/3d) are now left
      out of the global solve for every method, kept in the pairs graph: a tile with no other pair keeps its
      metadata position, named in a warning (data/3d's X01_Y01 under Elastix), where it took a neighbour's
      correction. Phase correlation on data/S000 and data/3d unchanged.
      run.py now exits 1 when an operation raised ('Done: N of M operations failed'), so Slurm marks the job
      FAILED; an operation skipped on purpose (existing output, no images) is no error.
- [ ] **napari not maximised through xpra-connect.sh** - it starts maximised to the virtual screen before the
      browser connects; the screen then follows the tab's size (--resize-display) but the window keeps its size.
      Maximising by hand works fine. Maybe maximise once the first client connects.
- [ ] **Faster per-source xarray construction** - not worth a second build path on its own: building the msims is
      CPU 1.9ms opening a source + 3.1ms xarray a source locally, 1.3ms for xarray in one pass (final dims, coords and
      transform to the constructor): ~25-30% of each build. The HPC run of 2026-10-13 built them three times - 15.1
      min pre-processing, 18.4 global registration, 13.4 fusion's make_msims_3d (47 min of ~20h, saving ~12) - so
      building fewer times (see The msims build in global registration) gains more. The preview cap half has gone:
      it only runs where the lazy overview declines.
- [ ] **Scaling in the registration preview** - possible to add scaling to the registration preview?
- [ ] **Instrument persistent identifier** - an optional project setting (e.g. `instrument_id`) for a facility's
      registered instrument PID (PIDINST/DataCite handle, RRID), used as the zarr crate's instrument `@id`.
- [ ] **Keep the refresh bar moving** - per-source or per-batch progress for the preview size cap and the Qt-thread
      shape steps.
- [ ] **Faster opening of large projects** - napari's per-shape Python cost (~10s at 150k shapes, main viewer and
      overview widget alike), ~39s untimed after init_data on the HPC (populating tables?), and init sources over
      NFS (42ms CPU a file on the HPC vs 2.7ms locally: ~31 small reads a file by tifffile).
- [ ] **Lazy overview: strided reads of single-level sources** - each tile is read whole at full res, then strided;
      read only the strided rows (as the direct uncompressed-level reads do), and measure on a single-level set.
- [ ] **Upstream fixes to multiview-stitcher** - phase correlation's spearman quality for every candidate shift
      (only the kept one is needed), the O(nodes x edges) reference-node search, the HiGHS thread pool from
      changing threads. Details in notes/multiview_stitcher.md. And to dask: linear fusion's key names (a 115-char
      prefix + 4 hex digits of hash()) collide between similar chains in one graph (see Done).
      To multiview_stitcher also: OME-Zarr 0.6 in its writers; zarr writing that fuses each level for multiscale input
      (as its lazy path does, with custom levels and per-level sources - see native fusion); level names that sort as
      text past 10 levels (its writer's '10' lands after '1' in napari's own reader).

From the HPC run of 2026-10-02 (34k sources, 229725 pairs; registration 8.9h in all, no errors):
- [ ] **The msims build in global registration** - 18 min on one core (GIL-bound) for the full per-source pyramids
      (18.4 again on 2026-10-13). The view refresh no longer reads them (PR #66); check what still does -
      save_mappings_csv, the metadata table - and whether source_geometry could serve them, as fusion builds its own
      in make_msims_3d (13.4 min) anyway.

From the HPC run of 2026-10-15 (c1baed5, averaged levels; 34k sources, 229725 pairs; registration 4.7h, fusion
3.4h, no errors; fusion 10.4h on 2026-10-12):
- [ ] **Fusion's 18 min before level 0** - make_msims_3d 12.8, output properties 1.1, export chunk sizes 1.0, then
      3 min from 'Fusion levels' to the first block. The msims are built three times (pre-processing 15.8, global
      registration 18.4, fusion 12.8 min): see The msims build in global registration.
- [ ] **Pair registration dispatch-bound once pairs are cheap** - largest first: the first 10% of pairs took 64 min,
      the next 37, then ~8 min a 10%; the last half at 0.09s a pair ran 128 pairs in ~2.7s with 4-6 of 64 workers
      busy (process cpu 1.6 cores): ~20ms a pair in the main process. 38.6 of 64 busy overall (39.7 before).
- [ ] **Channel name warnings** - 'Registration channel 'channel 0' not found - using the only channel, '#0'' and
      'Fusion: 137 source(s) with another channel name given '#0'': check the 137 (overviews?) and the setting.
      2026-10-22 the other way round: channel '#0' (the setting now) not found, the only one is 'channel 0'; the
      same 137 sources at fusion.
- Robust linear 17 min (8 rounds, ~110s each; 78 min before), pair metrics 37 min, global 31 min (+18 min msims).
  Refresh after registration not exercised (fusion ran straight after); after fusion 1 min.
- [ ] **Faster refresh after registration** - 48 min on the HPC (2026-10-03). Per source on registered meatballs:
      image shapes 1.3ms, tables 1.8 min; the preview cap and fused preview only where the lazy overview
      declines (a rotation, 2D and 3D sources together). Check on the next HPC run. (Promotion, the second transform
      copy, the cap's estimate and the lazy overview after registration: see Done.)

REFACTORING

- [ ] **Keep or drop** estimated_pair_cost (2026-10-15: workers busy 38.6/64, 39.7 without; the slow part is now
      the cheap pairs' dispatch) and SectionPlanes' prefetch of neighbouring sections (not seen in the log).


## Done

- **Physical units in napari** - images and shapes already had the sources' scale and positions in um, but no
  unit, so napari read them as pixels: every placed layer now has units='micrometer' (constants.spatial_unit),
  as empanada-napari's layers got (836aff0). The feature-match view stays in pixels, which it is drawn in.
- **Shape layers kept across steps** - each viewer's shape layer is keyed by the geometry, pairs graph and metrics
  it was drawn from: pre-processing and fusion (export, preview, the output shown) swap only the image under it,
  its line colour reset; pair and global registration draw it again. Preview and modify pair registration set the
  view aside and put it back on leaving (tab change, either answer), a stored pair recoloured with quality 1 in
  the metrics too. At 34k on the HPC a refresh built and added shapes for ~1 min (build 21s, add 21.5s, overview
  15s); locally at 149k shapes adding takes 7.4s, recolouring 1.3s, re-attaching 0.9s. The plugin's viewer is
  napari's PublicOnlyProxy, which wraps each layer anew at every access: layers are compared unwrapped.
  Overlaps without a quality (not registered) are grey, not black; on heavily overlapping tiles (data/S000) the
  stacked half-opaque shapes still hide the image.
- **Native levels from the level above, measured** (2026-10-15, meatballs headless): a level with the same sources
  as the one above (all but 0.01 and 0.2492 here) averaged 2x2 from it instead of fused from the sources:

  | Level | Fused from sources | Averaged from above | Pixels differing | Mean / p99 diff (uint8) |
  |---|---|---|---|---|
  | 0.02 um | 13.7 s | 3.5 s | 91% | 5.2 / 24 |
  | 0.04 um | 8.6 s | 1.1 s | 96% | 9.5 / 35 |
  | 0.08 um | 5.7 s | 0.4 s | 97% | 12.1 / 45 |
  | 0.16 um | 3.9 s | 0.1 s | 98% | 14.4 / 55 |
  | 0.50-7.97 um (5 levels) | 2.2-4.4 s each | ~0.1 s each | 61-77% | 7.7-11.1 / 42-57 |

  All levels 82 -> ~40s, and no coarse-level memory peak (HPC 2026-10-15: 368GB at 0.494). Not a shift (best-fit
  0.00-0.05 px) and not blur: the fused levels are aliased. Fine detail (mean squared Laplacian) stays ~5000-8500
  at every fused level, the averaged ones drop to 0.26 -> 0.06 of it, as averaging noisy data should. Each fused
  level reads its sources one pyramid level too fine: the level chosen with tolerance 0 against spacings read off
  coordinates (a tile's 0.04 level as 0.040000000000020464, the output's 0.03999999999996362), so output 0.02
  reads tiles at 0.01 ... 0.16 at 0.08, overviews likewise; below 0.16 tiles (their coarsest) are sampled from
  0.16 down to 7.97 - point sampling, the per-level fixed cost and the coarse levels' memory. Done (branch
  averaged-levels): such levels are averaged from the one above, and the levels still fused read their sources at
  their own size (the default 1e-6 relative tolerance). Meatballs native fusion 90.6 -> 50.0s; fine detail now
  falls level by level (1803, 1207, 1085, then 822 -> ~400) instead of staying at ~5000-8500.
- **One-pass average fusion** - of a meatballs level-0 block (profiled headless, 40 blocks): resampling 48%,
  multiview_stitcher's simple_average_fusion 22% (a stack and a NaN-free copy built per view), per-block fuse()
  setup 6%, dask and the zarr write ~12%. FusionMethodAverage does the same sums in one pass: bit-identical
  (every level of the meatballs fusion compared), 3.2x faster on a 3-view 4096x4096 block, the full meatballs
  fusion 116.6 -> ~94s (-19%). Additive is the same average clipped (was numpy's nanmean: bit-identical, 2.2x
  faster). What is left around the resampling is small and spread out.
- **Fusion methods weighed for speed** (2026-10-15, meatballs headless, all 11 levels, saved registration):

  | Method | Wall | CPU |
  |---|---|---|
  | average (1st run, includes warm-up) | 116 s | 438 s |
  | average (repeat) | 99.6 s | 394 s |
  | exclusive (custom) | 95.3 s | 361 s |
  | additive (custom) | 92.1 s | 360 s |

  The custom methods are 4-8% faster, as combining is ~20% of a block (resampling ~48%, the rest plan, dask and
  zarr write). Not a speed option: exclusive and additive are different outputs, and fusion speed-ups must leave
  the fused pixels identical (no cheaper method, no lower interpolation order).
- **Odd source sizes no longer add a native level** - the HPC levels ran 0.16, 0.2469, 0.3322, 0.6644 (2026-10-13):
  3 of 1081 overview sections (e.g. ov000_s00033) are at 0.3322, the rest at 0.2492/0.2469, debris images too;
  every source size got a level. Sizes under sqrt(2) above a group's finest now join its level (upsampled by less
  than that), export_levels selecting by the group's size: the HPC set gives 0.16, 0.2469, 0.4938, 0.9876..
  Meatballs locally unchanged (0.2492, 0.4983..).
- **Fusion's per-block plan on numpy** - each block's fusion plan (multiview_stitcher _build_spatial_fusion_plan)
  checked every source's affine through an xarray sel per entry: most of a small level's time (meatballs level 9:
  3 blocks, 5817 sel calls). The three checks are swapped for numpy versions during our fusion only; checks and
  fused output identical. Meatballs fusion 102 -> 80s, small levels ~2 -> ~1.2s each.
- **Fusion bar linear over its levels** - each level was a phase of its own, so level 0 ended at 90% and the rest
  crawled through the last 10. Now one bar over all levels, each its share of the square root of the size
  estimate's bytes (the export question's, else computed for the run; level voxels as fallback), the last level
  filling it; labelled 'Fusion (2/11 @ 0.02 µm)'. Square root, as a level's time grows slower than its size: on
  meatballs the bar at each level's end is within 5 points of the time taken (raw bytes: 27, cube root: 13); in the
  plugin 41% at 46% of the time, 72% at 73%, 99% at 99%.
- **Faster registration: robust linear and metrics** - from the HPC run of 2026-10-13 (registration 9.3h).
  Robust linear (78 min): each round's linear_two_pass built xarray transforms and per-node identity xarrays for
  its residuals; swapped for template/numpy versions during our calls, identical results, 8x a round locally
  (3600 tiles, 3 rounds 24.8 -> 3.1s). Pair (1.2h) and global (1.0h) metrics computed every pair's overlap images
  with no image metric configured, only to attach the quality: now skipped, the same pairs and summary built
  directly. With metrics asked for, a pair no longer rescans every library for its BLAS limit nor searches sys.path
  for xarray's optional dask_array: -32% a pair on tiny tiles. Expected on the next HPC run: ~3h less registration.
- **Progress bars update at most once a second** - NapariPhaseProgress shows a move (a redraw, or a worker's
  signal to the Qt thread) at most every `min_update_seconds`; held-back moves raise the target, and a phase's end
  and the operation's completion always show.
- **Faster refresh at scale** (branch `refresh-speed`) - timed on testing/make_synthetic_project.py (33990 tiles,
  1133 sections, 245733 pairs) with the UI driver's refresh_pairs/refresh_global actions: open 4.9-6.0 -> 3.0-3.7
  min, refresh after pairs 2.1 -> 1.4 min, after global 7.9 -> 1.8 min, pre-processing 5.0 -> 4.5 min (3.3 of it the
  processing). No view msims after global registration (5.5 min); metrics and metadata tables as a Qt model
  (ui/ModelTable, ~40s each refresh); a saved registration's transforms from cached templates (load 2.4 min -> 20s);
  overlap shapes reused after pair registration (~45s). A loaded global registration's summary is under the
  registered key, as the metrics table expects.
- **Restructure for maintainability** (PRs #63, #65) - one overview (the quick view takes 3D and multichannel
  sources), one OME-Zarr writer (fusion_blocks.fuse_levels_to_ome_zarr), _paste rewritten, the duplication scan
  merged and ~40 unreferenced functions deleted. Weighed and left - a shared helper would remove a few lines for
  flags or a one-line wrapper: the pair worker pool (register_pairs vs map_pair_metrics), the per-source
  build_source_stack_props loops, output filename expressions, the multi-z check, section z spacing, the
  quality-into-metrics merges, the pair-key parse, the two affine widenings. metrics.calc_msims_metrics still builds
  its graph with mv_graph's linear programs (moving it would change behaviour). Routing the quick view through fuse()
  was weighed too: ~5 shared lines for msims=None, readers and a view-only fusion method.
- **Efficiency audit** (PR #65) - nearly every optimisation has an order-of-magnitude gain at 34k sources, prevents
  out-of-memory, or works around a library bug (sweep pairing, polygon clipping, worker processes, rolling_map,
  deferred link quality, lazy overview, stack props, fusion budgets, preview cap, napari cache clearing,
  single_threaded_highs, ...). Removed as debt: the pair-scoring CPU instrumentation (a diagnostic monkeypatching multiview_stitcher), threads over
  GIL-bound per-source preview preparation; the 'Preview cap: one rebuild' TODO dropped (~3.5% of a refresh).
- **Direct TIFF reads restored** - removing them (c5e66f1, judged on one local export) made the 2026-10-12 HPC run
  slower: pairs +34%, pair metrics +103%, global metrics +63% against 2026-10-09; meatballs on 8 worker processes:
  pairs 73.5 -> 93.9s, metrics 14.4 -> 19.7s without them. Native fusion locally: 94 vs 101s, the same memory.
  With them back, main registers meatballs (8 processes, 831 pairs) as the 2026-10-09 code did: 112.7 vs 110.3s
  (pairs 72.0 vs 66.0s, within run-to-run spread; metrics 14.8 vs 14.4s; global metrics 21.6 vs 21.0s).
- **Fusion bar counts only the blocks it fuses** - those no source reaches were counted done at the start, so each
  level's bar sat at 88-99% for hours.
- **Run without threading** (PR #65) - every worker pool runs in the calling thread with one worker (rolling_map,
  thread_map, the quick view's reads and prefetch, contrast sampling, fusion batches); fuse() and
  lazy_section_overview take a `workers` argument.
- **Comments to why, without figures** (PR #65) - 1250 -> 764 comment lines, docstrings shortened, no timings or
  history; testing/code_unchanged.py proves no code changed. tiff_metadata.py left as its napari-meta-tiff copy.

- **Shape labels and edges contrast with the image** - the contrast-limit sample (quick view, or the coarsest level)
  also gives where the typical value shows (get_display_level); with the channels' colours that is the background's
  luminance, and labels and edges take a saturated colour apart from greyscale: dark magenta on a bright one, gold on
  a dark one, magenta between, unknown, or for a volume (dark as a slice, bright as its 3D projection) - the same in
  the 2D and 3D view.
- **Msim-free registered open** (branch `registered-open`) - init_progress keeps a loaded registration in
  self.mappings, applied when msims are first built (_apply_mappings), and finds files by dict; the pairs graph,
  registered shapes, the quick view's transforms and the size estimate come from MVSRegistration.source_geometry
  (equal to the registered msims', test); the transform key and coordinate systems from the state; no
  pre-processing re-run on open (global registration runs it when it needs the pairs' msims); view msims only for a
  fused preview. Local meatballs opens with no msim built: registered 9s, fused 7s (was ~15s). On the HPC's 109
  min this removes the msim builds (30.5 + 9.2), the re-run (10.2), view msims (3.3) and the file lookups of the
  mappings loads (~20, to confirm) - measure on the next HPC run.
- **A fused project opens its export** - _show_loaded_project reads the fused zarr (read_msim_from_ome_zarr) when it
  was written after every saved registration file (MVSRegistration.current_fused_output), else previews as before.
  The HPC open spent ~41 of 109 min on a capped preview fusion, at 1.13um where 100nm was asked (343GB capped to
  2GB, the overview averaged into the tiles). Local meatballs: open ~15s, the export shown in 0.6s, every level sharp.
- **Duplication list** (branch `restructure`) - ~40 unreferenced functions and ome_ngff_helper deleted (652 lines);
  one progress_phase (was 2 copies + 7 inline), one pooled_map for building sources and msims (single-threaded with
  one worker), one um unit table (util lacked nanometre/inch, tiff_metadata picometre), msims_stack_props for both
  pair graphs, sims_from_sims_or_msims and sim_views used where written inline, sim_with_data for the processed-sim
  re-wraps, save_pair_results for the three pair-saving paths, z_positions computed once in _create_napari_data.
- **One zarr writer** (branch `restructure`) - every OME-Zarr export goes through fusion_blocks.fuse_levels_to_ome_zarr
  over export_levels: native as before; a set spacing fuses level 0 from the sources reaching each block and averages
  each level after from the one above (average_into_zarr_array), z never averaged across sections. Gone: the z-slab
  writer, multiview_stitcher's pyramid step and fuse()'s fusion.fuse zarr branch, the OME-Zarr 0.6 refusal (a set
  spacing writes 0.6 too), and 3 of 4 coarsest-level lookups (image.util.coarsest_level_within; split registration
  now compares y/x only - its max over z's 1.0 placeholder kept it at full resolution). Slides 2 sections x 9 tiles
  at 0.016um: stack 2.5 -> 2.1s, one section 1.6 -> 1.1s, native 4.8 -> 4.5s, every level byte-identical. Fusing
  each level from the sources was tried for a set spacing: 70% slower (interpolation at every level), dropped.
  test_split_registration's group-correction test fails alone (robust_linear unknown) with or without this.
- **One overview instead of two** (branch `restructure`) - the quick view takes 3D and multichannel sources: (c, y, x)
  planes, volumes a plane per z at the finest z read, each source's nearest z, a volume's decoded z chunk kept for
  the planes after it. composite_msims_overview and its branch are gone: the view is the quick view, else the
  capped fused preview (a rotation, 2D and 3D sources together). data/3d and a multichannel set: the same pixels as
  the composite (one more empty edge row/column); data/3d all 200 planes 1.05s vs the composite's 0.28s up front,
  the first viewed plane 0.08s. Plugin captures of slides and the 3d project show the overview under its shapes.
- **init_data / ImageSource overlap** - a source no longer builds its own transform (init_data's alone reaches the
  msims), so init_sources builds every source in one pool; each file picks its own source_metadata entry in
  ImageSource (its label's matrix, a list by index); one x/y translation fill (complete_translation). Geometry of
  the slides, Ciqtek (both), meatballs and 3d projects identical before and after. One fix: a file without a
  labelled matrix used to inherit the rotation of the labelled file before it.
- **Fusion bar names its level** - 'Fusion: Level 2 at 0.04 µm' (native), 'Fusion: Pyramid level 1'
  (z-slab), in the bar and the heartbeat log. The z-slab path's pyramid levels (multiview_stitcher's ngff_utils)
  now move the bar - the HPC's 5.5h at '100%' - but only within the last ~10% left after level 0: weighting
  them needs their block counts before level 0 is fused.
- **Overview dock left behind** - closing the plugin's dock (its x), or deselecting it in the Plugins menu (which only
  hides it), left the overview, its own dock,
  behind; it now closes, hides and shows with it. napari's disable already removed both (by name).
- **Native fusion lost the overviews' level** - the HPC output had 0.01..0.16, then 0.3322 doubling, no 0.249: some
  sources are at 0.3322 (4/3 of 0.249), and a source size under sqrt(2) above the previous level replaced it, even
  when that level was another source's size. Now only a doubled level is replaced (0.01..0.16, 0.249, 0.3322,
  0.6644..). Tiles at half or double the usual size were already fine. Native fusion logs its levels and how many
  sources have each pixel size - see which are at 0.3322 on the next HPC run.
- **Composite overview collapsed sections spaced other than 1** - with no z scale given, calc_output_properties took
  a size-1 z's reported 1.0 as the output's z spacing, so meatballs' sections 0.05 apart all landed in the first
  plane (each pasted over the last). A stack of single-plane sections now takes the spacing between them, as
  extract_z_scale and the lazy overview do; fusion and the preview cap share it.
- **Lazy per-section overview after registration** - the refresh pasted all 34k sources one by one before showing
  any (11.9 min on the HPC, NFS reads); it now builds each section when viewed, as before registration, at the
  registered transforms already on the sources' msims - no file opened for them (counted: 0 on meatballs). Meatballs:
  created in 0.04s, a section in 0.28s, vs 1.8s for the composite; a single registered source lands within one
  output pixel of its transform's box in both. The preview cap and composite are left as the fallback.
- **Preview cap: not the sections, the reduction rounds** - the HPC's 6.3 min is ~11ms a source, the same per source
  as meatballs forced to the same 85x reduction (3.3ms locally): size estimates and msim rebuilds per round.
  promoted_geometry widened each affine through an xarray only to read it back (widened_affine_matrix now): an
  estimate 0.40 -> 0.19ms a source, the cap 3.3 -> 2.6ms. The composite overview and fusion setup share it.
- **Refresh after registration: view msims kept 2D, transforms copied once** - make_msims_3d promoted every level of
  every view msim (11.8ms a source: the HPC's ~13 untimed minutes); the preview steps now take z_positions as the
  pre-processed branch does, and shapes take each source's z from promoted_geometry. The preview msims no longer
  get the transforms a second time (2.0ms a source, 1.4 min on the HPC): the view msims already carry them.
  Meatballs: shapes and composite overview identical to before, these steps 5.6 -> 2.8s.
- **7.6 min before the first pairs on the HPC** - per source on meatballs: pairing geometry ~2.6ms (a shape sim
  each), channel selection ~2ms (multiscale_sel_coords), adjacency graph ~1.3ms (a sim each) - ~3.3 min at 34k
  locally, the HPC's 2.8 + ~3.5 min. Now timed (pair registration: source geometry / get_pairs / select channel /
  view adjacency graph). The first batch was mostly worker start-up: each worker imported napari and Qt through
  the package's __init__ (MainWidget, now imported on use): 8 workers ready in 7.6s instead of 10.2s, every
  respawn too. Left out as one-off gains (user decision): stack-props geometry, a leaner channel selection (1.70 ->
  1.24ms a source), direct stack props for the graph, starting the pool before the setup.
- **Pair registration 'slowed' towards the end: the cheap pairs came first** - a pair's cpu follows its overlap in
  pixels at the coarser of its two pixel sizes (r=0.97 on meatballs); in source order the overviews (indices 0-2)
  put every tile-overview pair (0.06s, at 0.498um) first and the tile-tile ones (0.3-7s) last - meatballs 42%,
  the HPC turning slow at 44%. register_pairs now submits the largest estimated overlap first (estimated_pair_cost).
  Meatballs wall unchanged on 8 workers (91.5 vs 88.9s); check the bar's pace and workers busy on the next HPC run
  (39.7 of 64 before).
- **Fusion level 0 peaked at 11GB on meatballs** - a single-plane source's z spacing is a placeholder 1.0, so with
  sections 0.05 apart multiview_stitcher's interpolation padding made every section reach every other's blocks
  (busiest block 147 of 153 sources, each transformed in full for no weight): source_bounds no longer pads a
  single-plane dim. The z-slab path also budgets its blocks from the sources' bounds (budget_chunksize, as native
  fusion does): clustered tiles beat the even-density estimate. 'average' at 'max': peak 11.4GB -> 1.0GB, level 0
  20 -> 9s. The HPC's section step is 1.0 (aligned), so this was not its 250GB.
- **robust_linear stops on the 99th percentile change** - the largest change never settled on the HPC (a few of
  229k edges flip every round); each round logs both, and the tolerance. Meatballs: 9 rounds either way. Confirm
  the early stop on the next HPC run.
- **Global registration bar moves per robust_linear round** - it sat at 0% for the whole solve.
- **Native-resolution fusion** (PR #60) - output_spacing 'native' (None when writing a file; default for new projects):
  levels at the sources' own pixel sizes, each from the sources at least that fine, tiles over the overview, every
  block from only the sources reaching it, blocks budgeted per level, level names zero-padded past 10. Meatballs
  native 5 min / 1.2GB vs mean 5 min / 3.4GB; HPC projection ~0.3TB vs 7.6TiB - measure on the next HPC run.
- **A real convert** - meatballs tiles (2304x3072, levels 0.01..0.16um) at pre-processing scale 2: level 0 1152x1536 at
  0.02um, then the source's own levels and one made at 0.32um (72x96), the largest dim under 128px.
- **dask fused-key collision elsewhere?** - no: fusion's graphs (lazy, and each export block with its zarr write) keep
  their names through dask.optimize (dask 2025.10, no 4-hex-digit fused keys); the global metrics already compute
  with linear fusion off (metrics.py).
- **HPC pair registration memory** - rss grew ~1.2MB a pair and slowed after ~100k pairs (19h for 229k pairs, before
  worker processes); the 2026-10-02 run took 4.6h at a flat ~18GB.
- **Faster refresh after registration** (merged 2026-10-03) - set_msim_affine for every transform write (a third of
  msi_utils.set_affine_transform's cost), tables filled in linear time (Qt header signals held: 74s -> 7s at 229k
  rows), timers on the untimed steps, 2D overlap shapes by polygon clipping instead of linprog (4.7 -> ~0.3ms a pair).
- **Global registration metrics progress** - per registered pair, not one step for ~1h (44 min on the 2026-10-03
  run, the bar moving throughout).
- **Fusion on mixed channel labels** - the HPC fusion (2026-10-03) failed with KeyError '#0' after writing 471MB:
  fusion selects every source by the first one's 'c' labels. fuse() now gives a single-channel source named
  otherwise the common label (unify_msim_channels), and stops with an error on differing multichannel labels.
  Confirmed on the next HPC run (137 sources relabelled, fusion completed).
  The names come from the files: SBEMimage writes Name="#0", and a channel without a name gets
  muvis-align's 'channel N' - so the HPC set mixes SBEMimage tiles with unnamed files.
- **HPC pair registration speed** - 3.2h on 64 worker processes on the 2026-10-03 run (4.6h before), rss flat
  16-18GB; the exact overlap test drops the ~3.6k of 233k candidates that only touch.

### Opening large projects (PR #56)
- **Lazy per-section overview** - raw sources shown one plane per section, built when viewed from each source's
  coarsest level no coarser than the preview scale; planes kept within 1GB, the 4 sections either side prefetched.
  After pre-processing it reads the pre-processed sources the same way.
- **Middle section first** - the middle section (by the section number in file names, else by folder) is read and
  shown before the rest, and the view stays on it (napari starts on the middle step).
- **Faster shapes** - transforms from a shared template (geometries 6.0 -> 0.7s at 3100 sources); napari's shape
  labels patched to read the shape list once (add_shapes 22.8 -> 10.2s at 150k shapes).
- **Project file in the output folder** - copied there as each action starts.

### HPC memory and speed
- **Fusion memory pool and block-size dropdown** - every CPU fuses; blocks share a quarter of the job's memory,
  each reserving its sources x chunk x 12 bytes (averaged levels: their float64 parents) before it runs, and
  are sized so 16 fit at once (pool / min(workers, 16): 1GB a block on a 64GB/64-CPU job), so a coarse level
  still splits among the workers. 'Tile size(s)' is now the 'Block size' dropdown (Automatic, 1024-8192);
  an old value maps to its listed size. Locally: meatballs 67.9 -> 48.4s; repo 2D/3D registration and fusion
  identical. The same block size with any pool gives identical pixels; a different size moves ~1 pixel in
  5e7 by one grey level (float rounding at block edges), as memory-dependent sizes always did.
- **Job memory read inside the container** - fusion's block budget and napari's dask cache from the job's
  allocation (Slurm variables passed in, else the process's own cgroup), not the node's ~2TB: fusion peak
  148.5 -> 23.9GB on a 200GB job for ~10% more fusion time. Job scripts back to 64G.
- **Fusion level 0.247 out of memory on a 64GB job** - the only coarse level fused from its sources (the overviews
  join there; 0.16 -> 0.247 is not 2x), it took rss 26 -> 159GB on a 160GB job and was OOM-killed on a 64GB one:
  its blocks were budgeted from the node's ~2TB (fixed above). With the pool and 1GB blocks (2026-10-24, 64GB,
  full run): 30.8 min, peak 33.3GB; fusion 3.7h against 3.4h at 160GB, the difference in levels 0.01-0.02 (more,
  smaller blocks: a block costs ~70ms fixed). Larger blocks would win that back only with more memory: not pursued.
- **Refresh after pre-processing: 47 min / 232GB -> ~18 min / 14GB** - no 3D promotion for the overview (2D sources
  placed at their section z); stored pyramid levels read instead of whole tiles; one source per synchronous compute.
- **Pre-processing 29.7 -> 16 min** - only each source's finest and coarsest level built (registration reads the
  finest, the preview cap the coarsest; convert builds the full pyramid); build threads capped at 32.
- **Cheaper TIFF levels** - da.from_array with a cheap name (no tokenize); picklable for worker processes.
- **Pairing setup: ~5h -> minutes** - default pairing gets the bounding-box sweep's candidates; orthogonal pairs,
  the pair graph (exact AABB overlaps for translations) and orthogonal geometry from metadata. Default pairing is
  the wrong choice for a multi-section stack: orthogonal (or split) is.
- **Pair registration in worker processes** - native pools at 1 thread, replaced every 1000 pairs, threads when a
  pair does not pickle: 153 sources 168.8 -> 111.7s on Windows, 271.8 -> 99.7s in Linux. Phase correlation's
  quality computed for the kept shift only (pair CPU halved).
- **robust_linear global registration** - IRLS around linear_two_pass: comparable fit to global_optimization,
  ~17x faster locally.
- **Metrics in worker processes** - per registered pair (3.1 min -> 43.5s, 6.5 -> 0.44GB); summary weighted exactly.
- **Registration channel per source** - HPC files name it 'channel 0' or '#0'; a warning once per label set.
- **napari's dask cache emptied on view change** - it kept removed layers' chunks, up to a quarter of RAM (~500GB
  on the HPC).

### Split pairing and cancel
- **Cancel** - the Process button reads Cancel; operations stop at their next progress step, restoring transforms
  and removing partial output (f580800).
- **Split pairing** - stage 1 pairs within each z-plane (or channel), stage 2 registers consecutive fused sections
  with the configured method and transform type through multiview-stitcher (one common grid, 5% margin, tile/25
  smoothing, background filled), binned by split_binning (blank = 8). Group pairs saved in pair_mappings.json and
  shown in the metrics table. User decision: no custom rotation search in stage 2 - rotation comes from the method
  and transform type (sift rigid handles the slides project's ~6 deg section rotations).

### Fusion
- **Output size in the export question** - 'Export fused data?' shows the uncompressed output, per level in
  the log, from source geometry and the registration's mappings (no msims; 0.1s for meatballs). Native fusion's
  is exactly the blocks it writes (meatballs 2.0GB, where 'Fusing 21.8GB' was the full-resolution bounding box).
- **Dask 'input Dask array will be rechunked' warning** - raised for every fused block at the array's edge,
  whose zarr chunk the array ends in: a false alarm, as each block is one chunk with one writer. Filtered around
  the fusion call: a napari worker (superqt) puts 'always' ahead of module-level filters. Chunks are also clipped
  to the output's extent.
- **Export per z-slab** - fused from only the sources reaching each slab (was ~1.5 cores: fusion re-planned over all
  sources per block); direct reads of uncompressed TIFF levels; the plugin's bar no longer counts a save phase that
  never runs.
- **Exclusive fusion** - each pixel from the earliest imaged (source order) view, one pass, works on 3D chunks.

### Registration preview
- **Grid for transform-only methods** - phase correlation and elastix show point pairs mapped through the transform:
  30 along the crop's longest side at one spacing (at least 3 a side, 2.5 ring sizes apart), every point kept;
  points and lines sized to the image shown.
- **No-overlap warning, steadier view** - two images without overlap warn instead of failing; the preview keeps the
  view until its result is ready, shows its metrics first and logs its step timings.
- **Faster scikit-image SIFT** - keypoints sampled before descriptors, no 2x upsampling (Ciqtek full-res pair
  124s / 6GB -> 22s / 2GB). sift and orb use scikit-image (user decision).

### Crashes, hangs and stalls
- **macOS hang in _run_off_thread** - work done before the nested loop started could lose its quit, leaving the
  Qt thread waiting: hit by test_shape_layers_are_kept... in 4 of 7 CI runs. The worker now keeps its outcome and
  sets an Event the loop checks (skipped when already done, else polled). macOS CI, test_interface.py x8 per job:
  7/16 hung before, 0/16 after.
- **Preview hang (HiGHS)** - and the ~10s gap before its layers: scipy's HiGHS (linprog, in multiview-stitcher's
  overlap tests) starts ~11 workers per calling thread and tears them down when it exits; from napari's changing
  pooled threads a teardown spun in its TLS callback holding the Windows loader lock, so no thread could start.
  Every HiGHS solve now runs with threads=1 (util.single_threaded_highs, 22fcbd4). Found with cdb native stacks;
  120-preview stress loop clean (hung after 20 before).
- **HPC exit after pre-processing** - Mesa 22.3 llvmpipe crash (see Known issues); Mesa 25 in the image (43ac613).
- **Pairs registering each other's crops** - dask's linear fusion gave two pairs' crop chains the same key (115-char
  prefix + 4 hex digits of hash()); register_pairs computes with optimization.fuse.active off.
- **Pair metrics crash (OpenBLAS)** - OpenBLAS starting its pool from many threads at once: BLAS at 1 thread
  (threadpoolctl).
- **Crash log off on Windows** - faulthandler also reported access violations drivers handle (b8d59ba); tests
  release its file before removing their folder (close_fault_log).
- **Stray napari messages and windows** - private-access warnings muted around the activity dock only; worker-thread
  warnings reach napari's notifications (they were dropped); napari's multiscale label no longer flashes up as its
  own window under xpra.

### xpra / container
- **Clickable buttons when maximised (Chrome 154)** - Xvfb gets an 8192x4096 framebuffer so --resize-display can
  follow the tab, and napari starts maximised (c90d870). xpra.org no longer serves xpra-html5 21, so rebuilds get 19.
- **Apptainer sandbox kept** - the HPC's Apptainer is unprivileged without squashfuse, so a SIF unpacks on every run
  (2m38s vs 10.5s from a sandbox): xpra-pull.sh's sandbox stays.
- **Occasional Xvfb fatal error on startup** - "Cannot open /var/lib/xkb/server-0.xkm", "Failed to activate virtual
  core keyboard". Xvfb uses /var/lib/xkb over /tmp when access() says it is writable, which the user-owned sandbox
  passes, but the container's root is read-only. xpra-run.sh binds a per-job directory there. Why only some runs
  fail (node-dependent mount/NFS behaviour?) is unknown; confirm no further failures on the HPC.

### Sources and metadata
- **Rotated sources' overlap shapes** - on opening a project the overlaps were their tiles' bounding-box
  intersections, wrong for a source_rotation (Ciqtek rotation project); rotated pairs now take the exact 2D clip.
- **Vendor TIFF metadata** - pixel size and stage position from vendor tags (napari-meta-tiff's metadata module,
  ee05aeb); imagecodecs a dependency.
- **Scales with units** - pre_processing scale and preview_scale accept a pixel size ('40nm') as well as a factor.
- **Convert writes the pre-processed sources** - named after the input file.
- **Source table and metadata input** - 3 significant digits shown; an invalid source metadata expression warns.

### Tests
- **Consolidated** (branch tests-consolidation) - 50 -> 35 files, ~530 -> 379 functions, ~890 -> 716 cases:
  obsolete/vacuous tests dropped, per-change files folded into their module's file, near-duplicates tabled,
  shared builders in tests/data_builders.py, accuracy assertions replaced by stubs, test_run 13 -> 4 registrations.
- **No accuracy asserts** - tests never assert a registration's accuracy: stub the registration result and check
  the handling exactly (user decision; RANSAC and platform numerics vary).
