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

## In progress

Nothing.

## TODO

ESSENTIAL

- [ ] created zarr: starting at 0.01 scale, but dont see 0.25 scale (overview pixel size), could cause expensive resample of overview. The proposed in native_level_spacing should result in: 0.01, 0.02, 0.04, 0.08, 0.16, 0.249, 0.498..
- [ ] occasional fatal error on startup (when trying to connect?):
      2026-10-05 07:12:15,985 Warning: PyGObject 3.42.2 leaks memory with GObject-Introspection 1.74.0
      2026-10-05 07:12:15,985  every GLib callback leaks an executable closure, PyGObject 3.44.2 or later fixes this
      2026-10-05 07:12:15,985  xpra will use a workaround for `GLib.idle_add`, but other callbacks still leak
      2026-10-05 07:12:15,985  see https://github.com/Xpra-org/xpra/issues/5044
      2026-10-05 07:12:16,116 created tcp socket '0.0.0.0:9876'
      _XSERVTransmkdir: ERROR: euid != 0,directory /tmp/.X11-unix will not be created.
      The XKEYBOARD keymap compiler (xkbcomp) reports:
      >   Error:            Cannot open "/var/lib/xkb/server-0.xkm" to write keyboard description
      >                   Exiting
      The XKEYBOARD keymap compiler (xkbcomp) reports:
      >   Error:            Cannot open "/var/lib/xkb/server-0.xkm" to write keyboard description
      >                   Exiting
      XKB: Failed to compile keymap
      Keyboard initialization failed. This could be a missing or incorrect setup of xkeyboard-config.
      (EE) 
      Fatal server error:
      (EE) Failed to activate virtual core keyboard: 2(EE) 
      xpra initialization error:
      Xvfb: did not provide a display number using displayfd
      [Mon  5 Oct 07:12:19 BST 2026] Session ended; password file removed.
- [ ] **HPC channel names** - check which files name their channel 'channel 0' rather than '#0' (an old
      single-level export mixed in?). Fusion now relabels them (see Done), but they should not differ.


PERFORMANCE

- [ ] refresh view after registration - very long time for low res output??
- [ ] 19:30 CET time close napari - long time to acutally close node
- [ ] **Faster per-source xarray construction** - building the msims in pre-processing (~16 min at 34k) and the
      preview size cap (3.4 min on the HPC).

MINOR

- [ ] fusion size accurate for native mode? (reporting: 24.5TB, should be ~350GB - du zarr 260 GB)
- [ ] fusion bar not indicative, not showing which level etc.
- [ ] silence dask warning: the input dask array will be rechunked ... 
- [ ] disabling plugin doesnt remove left had overview window
- [ ] **Keep the refresh bar moving** - per-source or per-batch progress for the preview size cap and the Qt-thread
      shape steps.
- [ ] **OME-Zarr 0.6 from 'mean'/'min'/'max' fusion** - refused before fusing (multiview_stitcher 0.1.62 writes 0.4/0.5
      only); native fusion and convert write 0.6 (ngff-zarr >= 0.48). Ask upstream (see Upstream fixes).
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
- [ ] **The msims build in global registration** - 18 min on one core (GIL-bound) for the full per-source pyramids.
      Not skippable: save_mappings_csv, the view refresh (copy_transforms_to_msims), the metadata table and fusion
      all read them right after. The cost is the build itself (see Faster per-source xarray construction): ~10ms
      CPU a single-level source locally - a third opening the tiff as a zarr store, the rest xarray (assign_coords,
      alignment, DataTree.from_dict, expand_dims); 31ms on the HPC.
- [ ] **Faster refresh after registration** - left: composite overview 12.2 min, preview size cap 6.5 min. Check
      the new timers on the next HPC run (copy transforms to view msims, update_registered: tables).


## Done

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
- **robust_linear stops once converged** - when no residual moved more than 1% of the Cauchy scale in a round; the
  largest change is logged each round (check it on the next HPC run: 7 of 10 rounds were for nothing).
- **Global registration metrics progress** - per registered pair, not one step for ~1h.
- **Fusion on mixed channel labels** - the HPC fusion (2026-10-03) failed with KeyError '#0' after writing 471MB:
  fusion selects every source by the first one's 'c' labels. fuse() now gives a single-channel source named
  otherwise the common label (unify_msim_channels), and stops with an error on differing multichannel labels.

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

### Sources and metadata
- **Vendor TIFF metadata** - pixel size and stage position from vendor tags (napari-meta-tiff's metadata module,
  ee05aeb); imagecodecs a dependency.
- **Scales with units** - pre_processing scale and preview_scale accept a pixel size ('40nm') as well as a factor.
- **Convert writes the pre-processed sources** - named after the input file.
- **Source table and metadata input** - 3 significant digits shown; an invalid source metadata expression warns.

### Tests
- **No accuracy asserts** - tests never assert a registration's accuracy: stub the registration result and check
  the handling exactly (user decision; RANSAC and platform numerics vary).
