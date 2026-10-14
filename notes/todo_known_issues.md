# Known issues and TODO

Test projects: slides `C:/project/slides` (328 tiffs, 5 sections; the repo's `muvis_align_project.yml`,
12193 data), meatballs `C:/Project/slides/EM04652-02_slice17_spaghettiandmeatballs2` (153 pyramid tiffs,
3 sections, SBEMimage), Ciqtek `C:/Project/slides/Ciqtek`, data_subset (54 tiffs, 6 sections).
HPC: 34k sources in 1081 sections, xpra container (software GL).

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
(fusion_slabs.average_into_zarr_array) does the same per block, 2x2 only. Level 0's chunk comes from the per-worker budget (a quarter
of RAM over the CPUs), so the peak scales with the machine: the HPC's 250GB on ~2TB. A risk only if
_available_memory() reads more than the job may use (no Slurm memory variable or cgroup limit set).

## In progress

Nothing.

## TODO

ESSENTIAL

- [ ] 19:30 CET time close napari - long time to actually close node

MINOR

- [ ] **Faster per-source xarray construction** - not worth a second build path on its own: building the msims is
      CPU 1.9ms opening a source + 3.1ms xarray a source locally, 1.3ms for xarray in one pass (final dims, coords and
      transform to the constructor): ~25-30% of each build. The HPC run of 2026-10-13 built them three times - 15.1
      min pre-processing, 18.4 global registration, 13.4 fusion's make_msims_3d (47 min of ~20h, saving ~12) - so
      building fewer times (see The msims build in global registration) gains more. The preview cap half has gone:
      it only runs where the lazy overview declines.
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

From the HPC run of 2026-10-03 (c73720b; 34k sources, 229725 pairs; registration 7.2h, fusion 9.7h, no errors):
- [ ] **Fusion pyramid levels: 5.5h at '100%'** - 'average' at 0.01um: after level 0,
      ngff_utils.write_sim_to_ome_zarr built the lower levels from it, uncounted by the bar; its 250GB peak was
      expected (see Known issues). Now each level averages the one above in our own blocks (one zarr writer, see
      Done): only blocks some source reaches, z never averaged across sections, each level on the bar. Measure on the
      next HPC run, with the 14 min at 0% before 'Output stack' (timers: fusion: make_msims_3d, output properties,
      export chunk sizes) and 'Fusion output_chunksize'.
- [ ] **Faster refresh after registration** - 48 min on the HPC (2026-10-03). Per source on registered meatballs:
      image shapes 1.3ms, tables 1.8 min; the preview cap and fused preview only where the lazy overview
      declines (a rotation, 2D and 3D sources together). Check on the next HPC run. (Promotion, the second transform
      copy, the cap's estimate and the lazy overview after registration: see Done.)

From the HPC run of 2026-10-12 (4844e9f; 34k sources, 229725 pairs; registration 11.3h, fusion 10.4h, no errors):
- [ ] **Fusion memory climbs through level 0 and stays** - rss 35 -> 288GB during level 0 (0.01um, 2h52), never
      released, peak 510GB at level 6; on 2026-10-09 (direct TIFF reads) level 0 stayed at 25-30GB, peak 248GB.
      Likely the reads through zarr's I/O pool (256 threads, constants.py) leaving freed tiles in Linux's per-thread
      malloc arenas: not reproduced on Windows. Confirmed by the run of 2026-10-13 (7ef399e, direct reads back):
      level 0 rss 24-35GB, peak 39.4GB, in 2h39; level 1 peak 48.9GB. Close once the run's whole-fusion peak is in;
      zarr reads remain for compressed/tiled TIFFs and OME-Zarr, so if it shows there, cap zarr's pool or
      MALLOC_ARENA_MAX. Robust linear is back to 1h18 (msims built to converged; 1h23 on 2026-10-09, 1h36 on
      2026-10-12).
- [ ] **Fusion's fixed cost per level** - every level takes 13 min or more however small: 0.66-10.6um (hundreds of
      pixels) 27, 19, 16, 16, 13 min, 1.5h of the 10.4h. Per level and source: the level choice and sim
      (msim_level_spacings, get_sim_from_msim), sim_views, view_bounds, budget_chunksize and block_sources over 1081
      z blocks, a prepare_block_fusion per source group. Time each on the next run or a meatballs profile.
      2026-10-13: 18 min before level 0 starts (make_msims_3d 13.4, output properties 1.0, export chunk sizes 0.9);
      level 1 (averaged from level 0) 1.5h and counting, shown as 90-99% of the bar.
      The per-block plan's per-source affine checks now run on numpy (see Done); what is left of a small level's
      time locally is spread over xarray/dask per source and block. Compare the levels on the next HPC run.

REFACTORING

- [ ] **After the next HPC run, keep or drop** estimated_pair_cost (pairs largest first: workers busy was 39.7/64)
      and SectionPlanes' prefetch of neighbouring sections (a section's build time when scrolling).


## Done

- **Fusion's per-block plan on numpy** - each block's fusion plan (multiview_stitcher _build_spatial_fusion_plan)
  checked every source's affine through an xarray sel per entry: most of a small level's time (meatballs level 9:
  3 blocks, 5817 sel calls). The three checks are swapped for numpy versions during our fusion only; checks and
  fused output identical. Meatballs fusion 102 -> 80s, small levels ~2 -> ~1.2s each.
- **Fusion bar linear over its levels** - each level was a phase of its own, so level 0 ended at 90% and the rest
  crawled through the last 10. Now one bar over all levels, each its share of the size estimate's bytes (the
  export question's, else computed for the run; level voxels as fallback), the last level filling it; labelled
  'Fusion (2/11 @ 0.02 µm)'. Meatballs: 49% at 30s in level 1 (66% of the bytes), 89% at 1 min. Levels
  past the first few hold a few % of the bytes but take ~25s of 90 locally: see Fusion's fixed cost per level.
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
  sources), one OME-Zarr writer (fusion_slabs.fuse_levels_to_ome_zarr), _paste rewritten, the duplication scan
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
- **One zarr writer** (branch `restructure`) - every OME-Zarr export goes through fusion_slabs.fuse_levels_to_ome_zarr
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
  passes, but the container's root is read-only. xpra-slurm.sh binds a per-job directory there. Why only some runs
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
