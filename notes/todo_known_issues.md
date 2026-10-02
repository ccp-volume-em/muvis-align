# Known issues and TODO

Test projects: slides `C:/project/slides` (328 tiffs, 5 sections; the repo's `muvis_align_project.yml`,
12193 data), meatballs `C:/Project/slides/EM04652-02_slice17_spaghettiandmeatballs2` (153 pyramid tiffs,
3 sections, SBEMimage), Ciqtek `C:/Project/slides/Ciqtek`, data_subset (54 tiffs, 6 sections).
HPC: 34k sources in 1081 sections, xpra container (software GL).

## Known issues

### Refresh bar stops moving in single-step phases
It stalls (without disappearing) in steps that report once, when they finish - capping the preview size (minutes
at 34k) - and while viewer steps run on the Qt thread (adding and refreshing shapes, ~1.5 min at 34k), where
nothing repaints. The bar hidden behind napari's welcome screen is fixed (old layers cleared only once the new view
is ready; welcome screen off while an operation's dock is up). See TODO.

### napari process sometimes not exiting after closing
Seen twice in ~20 scripted runs: every Python thread finished, the interpreter's native shutdown hung (one core
busy). The UI driver now ends itself with TerminateProcess; a user closing napari could see the same lingering
process. Not investigated (needs native stacks: WinDbg's cdb is installed now).

### Unknown root cause behind the HPC draw crash (worked around)
napari segfaulted drawing the pre-processed overview under xpra (Mesa 22.3 llvmpipe: draw_find_shader_output with
no vertex shader bound, from QOpenGLWidget::resizeEvent). Reproduced 100% in the Linux container, never in plain
napari. Fixed by Mesa 25 from bookworm-backports in the image (43ac613), confirmed on the HPC. Which GL program is
invalid at that draw was never found.

### Synthetic test of the transform direction disagrees with real data
On registered pairs multiview-stitcher's affine_matrix takes fixed crop pixels to the moving crop (10/10 meatballs
pairs by brute-force shift search; the pair NCC improves with it). A synthetic scipy-shifted pair suggested the
reverse. The preview grid follows the real data; the synthetic result is unexplained.

## In progress

Nothing.

## TODO

- [ ] From the HPC run of 2026-10-02 (34k sources, 229725 pairs; full run log from the user once it finishes):
      robust_linear runs all 10 rounds though the median residual stopped changing at round 3 (0.104; ~9-13 min
      a round there) - stop once it stops improving; the full per-source msims are rebuilt before global
      registration only to store the transforms (19 min on 1 core); the global registration metrics show no
      progress (one bar step, 75% for ~1h).
- [ ] Keep the refresh bar moving: per-source or per-batch progress for the preview size cap and the Qt-thread
      shape steps.
- [ ] Opening a large project: napari's per-shape Python cost (~10s at 150k shapes, main viewer and overview
      widget alike), ~39s untimed after init_data on the HPC (populating tables?), and init sources over NFS
      (42ms CPU a file on the HPC vs 2.7ms locally: ~31 small reads a file by tifffile).
- [ ] Speed up the remaining slow per-source phases, all xarray object construction: building the msims in
      pre-processing (~16 min at 34k) and the preview size cap (3.4 min on the HPC).
- [ ] Other computes over many similar per-source chains may hit dask's fused-key collision (see Done): fusion and
      the global metrics - check, or switch linear fusion off process-wide. Worth reporting upstream to dask.
- [ ] Lazy overview on single-level sources: each tile is read whole at full res, then strided - read only the
      strided rows (as the direct uncompressed-level reads do), and measure on a single-level set.
- [ ] HPC: check which files name their channel 'channel 0' rather than '#0' (an old single-level export mixed in?).
- [ ] Run a real convert with a pre-processing scale set: check the output's level-0 size and levels down to ~128px.
- [ ] HPC pair registration: rss grew ~1.2MB a pair and slowed after ~100k pairs (19h for 229k pairs, before worker
      processes) - re-measure with worker processes and robust_linear global registration.
- [ ] Upstream to multiview-stitcher: phase correlation's spearman quality for every candidate shift (only the kept
      one is needed), the O(nodes x edges) reference-node search, the HiGHS thread pool from changing threads.
      Details in notes/multiview_stitcher.md.

## Done

### Opening large projects (branch phased-source-init, PR #56)
- Raw sources shown as a lazy overview: one plane per section, built when viewed from each source's coarsest level
  no coarser than the preview scale; planes kept within 1GB, the 4 sections either side prefetched. After
  pre-processing it reads the pre-processed sources the same way.
- The middle section (by the section number in file names, else by folder) is read and shown before the rest, and
  the view stays on it (napari starts on the middle step).
- Shape transforms from a shared template (geometries 6.0 -> 0.7s at 3100 sources); napari's shape labels patched
  to read the shape list once (add_shapes 22.8 -> 10.2s at 150k shapes).
- The project file is copied into the output folder as each action starts.

### HPC memory and speed (pre-processing, refresh, registration)
- Refresh after pre-processing: 47 min / 232GB -> ~18 min / 14GB. No 3D promotion for the overview (2D sources
  placed at their section z); stored pyramid levels read instead of whole tiles; the overview pastes one source per
  synchronous compute.
- Pre-processing builds only each source's finest and coarsest level (registration reads the finest, the preview cap
  the coarsest; convert builds the full pyramid); build threads capped at 32: 29.7 -> 16 min at 34k.
- TIFF level arrays via da.from_array with a cheap name (no tokenize); picklable levels for worker processes.
- Pairing: default pairing gets the bounding-box sweep's candidates; orthogonal pairs, the pair graph (exact AABB
  overlaps for translations) and orthogonal geometry from metadata - ~5h of HPC setup to minutes. Default pairing
  is the wrong choice for a multi-section stack: orthogonal (or split) is.
- Pair registration in worker processes (native pools at 1 thread, replaced every 1000 pairs; threads when a pair
  does not pickle): 153 sources 168.8 -> 111.7s on Windows, 271.8 -> 99.7s in Linux. Phase correlation's quality
  computed for the kept shift only (pair CPU halved).
- Global registration: robust_linear (IRLS around linear_two_pass) - comparable fit to global_optimization,
  ~17x faster locally.
- Metrics per registered pair in worker processes (3.1 min -> 43.5s, 6.5 -> 0.44GB); summary weighted exactly.
- Registration channel resolved per source (HPC files name it 'channel 0' or '#0'), warning once per label set.
- napari's dask cache (up to a quarter of RAM, ~500GB on the HPC) emptied whenever the main view is replaced.

### Split pairing and cancel
- Cancel for long operations: the Process button reads Cancel; operations stop at their next progress step,
  restoring transforms and removing partial output (f580800).
- Split pairing: stage 1 pairs within each z-plane (or channel), stage 2 registers consecutive fused sections with
  the configured method and transform type through multiview-stitcher (one common grid, 5% margin, tile/25
  smoothing, background filled), binned by split_binning (blank = 8). Group pairs saved in pair_mappings.json and
  shown in the metrics table. User decision: no custom rotation search in stage 2 - rotation comes from the method
  and transform type (sift rigid handles the slides project's ~6 deg section rotations).

### Fusion
- Export to zarr fused per z-slab from only the sources reaching it (was ~1.5 cores: fusion re-planned over all
  sources per block); direct reads of uncompressed TIFF levels; the plugin's bar no longer counts a save phase that
  never runs.
- Exclusive fusion: each pixel from the earliest imaged (source order) view, one pass, works on 3D chunks.

### Registration preview
- Transform-only methods (phase correlation, elastix) show a grid of point pairs mapped through the transform,
  30 along the crop's longest side at one spacing (at least 3 a side, 2.5 ring sizes apart), every point kept;
  points and lines sized to the image shown.
- Two images without overlap warn instead of failing; the preview keeps the view until its result is ready,
  shows its metrics first and logs its step timings.
- scikit-image SIFT: keypoints sampled before descriptors, no 2x upsampling (Ciqtek full-res pair 124s / 6GB ->
  22s / 2GB). sift and orb use scikit-image (user decision).

### Crashes, hangs and stalls
- napari hang on a registration preview (and the ~10s gap before its layers): scipy's HiGHS (linprog, in
  multiview-stitcher's overlap tests) starts ~11 workers per calling thread and tears them down when it exits; from
  napari's changing pooled threads a teardown spun in its TLS callback holding the Windows loader lock, so no
  thread could start. Every HiGHS solve now runs with threads=1 (util.single_threaded_highs, 22fcbd4). Found with
  cdb native stacks; 120-preview stress loop clean (hung after 20 before).
- HPC exit after pre-processing: Mesa 22.3 llvmpipe crash (see Known issues); Mesa 25 in the image (43ac613).
- Pair registration mixing up pairs' crops: dask's linear fusion gave two pairs' crop chains the same key (115-char
  prefix + 4 hex digits of hash()); register_pairs computes with optimization.fuse.active off.
- Pair metrics crashed when OpenBLAS started its pool from many threads at once: BLAS at 1 thread (threadpoolctl).
- Windows: the crash log (faulthandler) is off - it also reported access violations drivers handle (b8d59ba);
  tests release its file before removing their folder (close_fault_log).
- napari private-access warnings muted around the activity dock only; worker-thread warnings reach napari's
  notifications (they were dropped); napari's multiscale label no longer flashes up as its own window under xpra.

### xpra / container
- Unclickable buttons with napari maximised under xpra in Chrome 154: Xvfb gets an 8192x4096 framebuffer so
  --resize-display can follow the tab, and napari starts maximised (c90d870). xpra.org no longer serves xpra-html5
  21, so rebuilds get 19.
- The HPC's Apptainer is unprivileged without squashfuse, so a SIF unpacks on every run (2m38s vs 10.5s from a
  sandbox): xpra-pull.sh's sandbox stays.

### Sources and metadata
- Plain TIFFs: pixel size and stage position from vendor tags (napari-meta-tiff's metadata module, ee05aeb);
  imagecodecs a dependency.
- pre_processing scale and preview_scale accept a pixel size with unit ('40nm') as well as a factor.
- Convert writes the pre-processed sources, named after the input file.
- Source table shows 3 significant digits; an invalid source metadata expression warns.

### Tests
- Tests never assert a registration's accuracy: stub the registration result and check the handling exactly
  (user decision; RANSAC and platform numerics vary).
