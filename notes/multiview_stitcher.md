# multiview-stitcher: what muvis-align relies on, and what to raise upstream

Checked against multiview-stitcher 0.1.62 (the latest release) and its main branch on 2026-10-28: the code behind
1a-1c, 4 and 9 is unchanged there. "Local" is the 3-section test data (153 sources, 831 orthogonal / 1764
overlapping pairs); "HPC" is the full run (34k sources in 1081 sections, 229725 pairs).

## Background

### Groupwise resolution (global registration)

`param_resolution.groupwise_resolution(g_reg, method, **kwargs)` runs a resolver per timepoint and connected
component. Resolvers are registered by name with `register_groupwise_resolution_method`; three ship with it:

- `global_optimization` (the default): iterative per-node point fitting on virtual beads (the overlap bbox corners),
  then removes **one** edge per outer pass (the worst by a residual/quality score, keeping the graph connected)
  until the max residual is below `abs_tol` (default: the voxel diagonal). It does not finish at HPC scale (1d).
- `linear_two_pass`: a linearised sparse least-squares solve (rotations, then translations, with
  `scipy.sparse.linalg.lsqr`), residuals from `utils.compute_edge_residuals`, one pruning step (edges above
  `residual_threshold`, or median + `mad_k` x MAD; `keep_mst` keeps a spanning tree) and a second solve. Only
  'translation' and 'rigid'. Edge weights from `weight_mode` ('quality_overlap', 'quality', 'overlap', 'uniform').
  Fast, but plain least squares: bad pairs pull the fit (1a).
- `shortest_paths`: transforms composed along shortest paths from the reference view; no averaging over redundant
  edges.

There is no reweighting (robust) option; PR #121 added `linear_two_pass` as the "fast yet robust" method, its
robustness the single pruning step.

**muvis-align's `robust_linear`** (`robust_resolution.py`, the plugin's default) is iteratively reweighted least
squares with `linear_two_pass` as the solver: each round sets an edge's quality to its pair quality times a Cauchy
weight of its last residual, 1 / (1 + (residual / scale)^2), and re-solves with pruning off (`mad_k` very large,
`keep_mst=False`, `weight_mode='quality'`). Scale: half the voxel diagonal; it stops once the residuals settle.
With a scale that small the weights act almost as 1 / residual^2, and the rounds can swing between a good fit and
one flinging tiles far off (Supra 5x5, 2026-10-28: median edge residual 1.4 / 300 um alternately), so it returns
the round with the lowest median residual. For a rigid solve a pair's quality is scaled by (overlap's longest side
/ tile's longest side)^2: a corner-only diagonal overlap holds next to no rotation, yet its fit can say 108 deg.
`register_global` then re-solves with `linear_two_pass` if any tile moved over half a tile from its metadata
position, and keeps a tile still out at its metadata position.
Registered as a named method, so `groupwise_resolution` runs it per component. Only 'translation' and 'rigid';
other transform types fall back to `global_optimization`. Its rounds swap numpy versions of `linear_two_pass`'s
parameter building and residuals in (1b), and it picks the reference view itself (1c).

### Pairwise registration

- `dispatch_pairwise_reg_func` reads the registration function's signature to decide what to pass (14).
- A registration function returns `affine_matrix` and `quality`; on registered pairs `affine_matrix` takes fixed crop
  pixels to the moving crop (see "Transform direction" under Known issues in todo_known_issues.md).
- `registration_binning` overrides the automatic binning (9); split pairing passes it for the stitched planes.
- Registering a fused group needs it channel-selected (no 'c' dim), as register_pairs' own msims are: with one,
  `transform_sim` failed ('affine matrix has wrong number of rows').
- muvis-align registers pairs in worker processes (6). Native pools (BLAS, OpenCV) run at one thread there, which
  changes floating-point summation order: 43 of 1764 pairs differ by up to 0.046um from registering with default
  native threading.

### Overlaps, metrics and fusion

- Building the view adjacency graph tests each candidate pair's overlap with a linear program (2, 10), from
  candidates within the largest source's diameter (3). Without a dask scheduler set it computes overlaps with
  spawned processes, so a script needs a `__main__` guard (answered upstream in #144); muvis-align sets a scheduler
  around every call.
- `metrics.tile_pair_image_metrics` in its overlap mode measures every pair overlapping under the base transform (7).
- `fusion.fuse` builds a fusion plan per output block (8a); `simple_average_fusion` is the 'average' rule (8b).
  Writing to zarr (`ngff_utils`) is replaced in muvis-align by `fusion_blocks.fuse_levels_to_ome_zarr` (12).

## Suggestions

Each item: what we measured, what we suggest, muvis-align's workaround, and **Post**: where it goes upstream
(a new issue, or a comment on an existing one, searched open and closed) and its title. Not posted yet. An issue's
text is the item's problem, measurements and suggestion, opened with this paragraph:

> muvis-align (https://github.com/ccp-volume-em/muvis-align) registers and fuses large serial-section EM mosaics
> with multiview-stitcher. Numbers below are from a local dataset (153 tiles in 3 sections) and a full HPC run
> (34k tiles in 1081 sections, 229725 pairs), with multiview-stitcher 0.1.62.

Workaround links point to `https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/<file>`: push
before posting. Ordered by impact at HPC scale: post 1a-1c first.

### 1. Global resolution: a robust linear method that scales

At HPC scale `global_optimization` does not finish and `linear_two_pass` fits poorly where pairs are bad;
reweighting `linear_two_pass` fits better at close to `linear_two_pass`'s cost. Four parts, each useful on its own.

#### 1a. A robust option for `linear_two_pass`

- **Problem:** `linear_two_pass` is plain weighted least squares plus one pruning step (median + 2 x MAD, a spanning
  tree kept), so bad pairs pull the fit; repeating the pruning on its own kept edges prunes down to a spanning tree.
- **Measured (local, rigid, 831 pairs):** `linear_two_pass` 0.9s, median residual 0.111um, p90 0.482um; reweighted
  (9 rounds to converge) 4.3s, median 0.109um, p90 0.284um; `global_optimization` 83.7s. HPC: the median residual
  settled at 0.104um by round 3, ~31s a round with 1b's change.
- **Suggestion:** a `robust_rounds` option (with a residual `scale`, e.g. half the voxel diagonal) on
  `linear_two_pass`: iteratively reweighted least squares, each round re-solving with every edge weighted by its
  quality x 1 / (1 + (residual / scale)^2) - a Cauchy weight of its last residual, so bad pairs stop pulling the fit
  without being cut and the graph stays connected - with pruning off (`residual_threshold=inf`, `keep_mst=False`,
  `weight_mode='quality'`). Stop once the 99th percentile of the residuals' change in a round is within 1% of the
  scale: on a large graph a few edges flip every round, so a stop on the largest change never triggers.
- **muvis-align:** `robust_resolution.groupwise_resolution_robust_linear`, registered as `robust_linear`.
- **Post:** new issue, referencing #101 (global resolution does not scale). Title: *linear_two_pass: an iteratively
  reweighted (robust) option*. Offer a PR.

#### 1b. `linear_two_pass`: the cost is building the parameters, not solving

- **Problem:** per call, `_build_params_from_components` builds an xarray transform per node twice (after pass 1 and
  pass 2), and pass 2 re-solves even when no edge was pruned - giving pass 1's result again.
  `compute_edge_residuals` builds the bead graph each call, with an identity xarray per node it never reads, and
  converts an xarray transform for every point it transforms.
- **Measured (local, profiled, 9 calls with pruning off):** 9.7s in all: the sparse solves 0.15s,
  `_build_params_from_components` 6.5s (18 calls), `compute_edge_residuals` 2.7s. A third of every call repeats
  pass 1 when nothing is pruned; two thirds are xarray construction.
- **Suggestion:** skip pass 2 (and its parameter build) when every edge is kept; keep the per-node components in
  numpy between passes and build the xarray transforms once, for the result returned; residuals on plain matrices.
- **muvis-align:** `robust_resolution._numpy_internals` swaps `_build_params_from_templates` and `_edge_residuals` in
  while its own rounds run. Transforms and residuals identical; a round 8x faster (3600 tiles, 3 rounds
  24.8 -> 3.1s); 34k tiles / 230k edges ~31s a round on one thread.
- **Post:** new issue, referencing #139 (xarray overhead, judged within limits as once per tile: here it is per tile
  per call). Title: *linear_two_pass: most of a call is building xarray transforms, and pass 2 repeats pass 1 when
  nothing is pruned*.

#### 1c. Reference view search: one pass over the edges

- **Problem:** `mv_graph.get_node_with_maximal_edge_weight_sum_from_graph` scans every edge once per node: O(nodes x
  edges). `global_optimization` and `linear_two_pass` call it whenever no `reference_view` is given, or the given one
  is not in the component.
- **Measured:** 1.2us a node-edge on the component subgraph view: ~2.7h at 34k nodes x 229k edges. The HPC spent
  4.8h before `global_optimization`'s first iteration.
- **Suggestion:** accumulate each node's weight sum in a single pass over the edges; same node chosen (3ms locally
  against 57-267ms).
- **muvis-align:** `robust_resolution.find_reference_view`; `register_global` passes it when the graph is connected.
- **Post:** new issue. Title: *get_node_with_maximal_edge_weight_sum_from_graph scans every edge once per node*.
  Include the few lines of `find_reference_view`; offer a PR.

#### 1d. `global_optimization` at scale

- **Problem:** it removes one edge per outer pass, re-running the inner loop (up to 500 iterations) and deep-copying
  the graph for each removal candidate.
- **Measured:** 967 passes locally (354-406s); on the HPC pass 1 took ~2h (323 iterations), then ~3 min a pass, 146
  passes in 10h with the max residual stuck at ~69.6. It would not finish.
- **Suggestion:** remove several edges per pass (e.g. all above a residual threshold, keeping connectivity) and avoid
  the per-candidate deep copies - or point large graphs to the robust linear method (1a).
- **muvis-align:** uses `robust_linear` (1a) for translation and rigid.
- **Post:** comment on #101, linking 1a's issue.

### 2. View adjacency graph: exact box intersections when nothing is rotated

- **Problem:** building the view adjacency graph solves a linear program (`linprog`) per candidate pair.
- **Measured:** 2h16 for 233k pairs on the HPC; locally 4.5s -> 0.3s with the change below, edges identical and
  overlaps within 3e-16.
- **Suggestion:** when all transforms are axis-aligned (every source before registration, usually), take each overlap
  as the bounding boxes' intersection, vectorised over all pairs; keep `linprog` for the rest.
- **muvis-align:** `image.util.build_view_adjacency_graph`, for the pairs it is given.
- **Post:** new issue. Title: *View adjacency graph: skip the linear program per pair when all transforms are
  translations*.

### 3. Default pair search radius

- **Problem:** with no pairs given, candidates come from a cKDTree radius of the largest source's diameter. One large
  source (an overview image) makes nearly every pair a candidate, each a delayed overlap task.
- **Measured:** the HPC run sat at 0% for 34+ min with rss 231 -> 344GB (~1.2 billion candidates). Locally (51
  sources) 2550 candidates with the overview against 436 without, for 179 real overlaps.
- **Suggestion:** a sweep over the sources' bounding boxes (each source's own extent) instead of one global radius.
- **muvis-align:** `image.util.find_candidate_overlap_pairs`, passed in as the pairs: same edges, graph build
  13.2s -> 1.1s.
- **Post:** new issue. Title: *Pair candidates within the largest source's diameter: one overview image makes every
  pair a candidate*.

### 4. Phase correlation: link quality for the kept shift only

- **Problem:** `phase_correlation_registration` computes a spearman link quality for every candidate shift (~11 a
  pair) and keeps one.
- **Measured (51 sources):** spearman was 160s of 238s scoring, in 302s of pair CPU. Deferring it: identical results,
  pair CPU 432s -> 217s, the pair registration 129.5s -> 90.6s.
- **Suggestion:** compute the link quality only for the chosen candidate.
- **muvis-align:** `MVSRegistration.deferred_link_quality` / `resolve_deferred_quality`.
- **Post:** new issue. Title: *phase_correlation_registration computes the link quality of every candidate shift*.

### 5. Phase correlation: disambiguation that prefers zero shift

- **Problem:** the candidate shift is chosen by SSIM over the union (or, with NaNs, the intersection) bounding box,
  with NaN pixels read as 0. When both images share an outline, a background or a fixed pattern at the same place
  (fused serial sections: same tile layout, seams and shading), that box matches best at zero shift, whatever the
  content does. The phase-normalised candidate also locks onto such a pattern.
- **Measured (fused sections, one grid):** skimage's unnormalised phase correlation found (8.9, 5.0)um where
  `phase_correlation_registration` returned exactly 0 on the same images. With the background as NaN, 3 of 5 section
  pairs came out right; the other 2 still chose 0 (NaN corners inside the intersection box). skimage's masked
  normalised cross-correlation found all 5 (NCC 0.52-0.92 at the shift found).
- **Suggestion:** score candidates only where both images have data (a mask, not the bounding box), and offer the
  masked cross-correlation candidate whenever a mask is known, not only when NaNs are present.
- **muvis-align:** split pairing fuses the sections onto one grid, smooths past the tile pattern and fills the
  background before registering; with large shifts relative to the section, a feature method (SIFT) works where
  phase correlation does not.
- **Post:** new issue. Title: *phase_correlation_registration picks zero shift when both images share an outline or
  a fixed pattern*.

### 6. Pairwise registration in processes

- **Problem:** `compute_pairwise_registrations` runs pairs in threads, and pair registration is GIL-bound.
- **Measured:** 64 threads did ~3 cores of pair work on the HPC. In spawned worker processes (native pools at one
  thread): local 1764 pairs 168.8s -> 111.7s (Windows), 271.8s -> 99.7s with peak rss 3.1 -> 0.47GB (Linux).
- **Suggestion:** a process-based option, which needs the source arrays to pickle.
- **muvis-align:** `MVSRegistration.register_pair_in_worker`, with `PicklableTiffLevel` so TIFF levels can be sent.
- **Post:** comment on #102 (pairwise registration in different dask schedulers), with the summation-order side
  effect from the background.

### 7. Registration metrics per registered pair

- **Problem:** `metrics.tile_pair_image_metrics` in its overlap mode (`query_transform_keys`, no `pairs_graph`)
  measures every pair that overlaps under the base transform, not the pairs that were registered.
- **Measured (local):** 1764 pairs where 831 were registered, 3.1 min and 6.5GB peak; per registered pair with only
  its two msims 43.5s, identical values.
- **Suggestion:** accept the pairs (or the registration graph) to measure.
- **muvis-align:** calls it per registered pair, in worker processes when there are more pairs than workers; without
  workers, one pair at a time in the calling thread (10).
- **Post:** new issue. Title: *tile_pair_image_metrics: measure the registered pairs, not every overlapping pair*.

### 8. Fusion

#### 8a. The plan once per fusion, not once per block

- **Problem:** writing to zarr, `_fuse_chunk_to_zarr` calls `fuse()` for every output block with that block as the
  output, so the fusion plan (`_build_spatial_fusion_plan`, `_get_axis_aligned_translation_dims`,
  `_get_grid_aligned_translation_dims`, `sim_sel_coords`) is rebuilt over all sources each time, with xarray `.sel`
  calls on every source's transform - although the plan already maps each source to the chunks it meets.
- **Measured (slides, 328 sources, 1280 blocks of 3840x3840 px, one block at a time):** 2.7s a block, of which
  `sim_sel_coords` ~1.1s (328 calls), the plan ~0.94s, the axis-aligned check ~0.74s, ~4270 `.sel` calls;
  `affine_transform` ~2%. That is Python holding the GIL: 24 threads reach ~1.5 cores. The cost per block grows with
  the total source count, so at 34k sources a block would cost ~100x more.
- **Suggestion:** build the plan (and the per-source transform checks, on plain numpy arrays) once per fusion and hand
  each block only the sources it meets.
- **muvis-align:** `fusion_blocks.fuse_into_zarr_array` fuses each block from only the sources that reach it, by
  multiview-stitcher's own rule, through `prepare_block_fusion(create_output=False)` once per group of sources
  (identical output); `fusion_blocks._numpy_fusion_plan` swaps in numpy versions of the per-source checks. Slides
  export ~31 -> ~9 min (~15-20% of that from direct TIFF reads, see tifffile below); meatballs fusion 102 -> 80s.
- **Post:** new issue. Title: *Writing to zarr rebuilds the fusion plan over all sources for every output block*.

#### 8b. `simple_average_fusion` in one pass

- **Problem:** to count the views reaching each pixel it builds, per view, a stack of the running count and that
  view's mask and takes its `nansum`; then `nansum` over all views copies the whole stack once more to replace NaNs.
- **Measured (meatballs, level-0 blocks, profiled):** 22% of a block's time. In one pass - one validity mask, counts
  summed from it, values summed with NaNs zeroed by `np.where` - the same sums in the same order: bit-identical
  (every level of a meatballs fusion compared), 3.2x faster on a 3-view 4096x4096 block, the whole meatballs fusion
  ~19% faster (116.6 -> ~94s). The same for a `np.nanmean` combine: bit-identical, 2.2x faster.
- **Suggestion:** the one-pass form in `simple_average_fusion`.
- **muvis-align:** `fusion_methods.FusionMethodAverage`; `FusionMethodAdditive` is that average clipped.
- **Post:** new issue or straight to a PR. Title: *simple_average_fusion: one pass, bit-identical, ~3x faster*.

### 9. Registration binning budget for 2D images

- **Problem:** `registration.get_optimal_registration_binning` bins a pair only while it holds more than
  `max_total_pixels_per_stack = 400**3` (64M) pixels. That suits 3D stacks; a 2D image of up to 64M pixels is
  registered unbinned.
- **Measured:** whole fused sections (6051x6801 px, 41M, at 0.032um): phase correlation 90s and 5.9GB a pair, SIFT
  over 10GB a pair. Binned by 8: SIFT 2.3 min for the whole slides registration, peak 2.0GB, the same alignment.
- **Suggestion:** a budget per dimensionality (e.g. 400**ndim, or a 2D default of the order of 1-4M pixels).
- **muvis-align:** passes `registration_binning` for the section pairs, from the `split_binning` setting (default 8).
- **Post:** new issue; mention #183 (merged: projects a singleton spatial dim; the budget is unchanged). Title:
  *get_optimal_registration_binning: the 400\*\*3 pixel budget leaves large 2D images unbinned*.

### 10. HiGHS threads in the overlap tests

- **Problem:** the overlap tests solve with scipy's `linprog` (HiGHS), which starts ~11 workers per calling thread and
  tears them down when that thread exits. Called from changing pool threads (napari's, or a thread pool's), a
  teardown spun in its TLS callback holding the Windows loader lock, so no thread could start: the process hung, or
  raised access violations after a pass or two. Found with native stacks (cdb).
- **Measured:** a 120-preview stress loop hung after ~20 previews; with one HiGHS thread, clean.
- **Suggestion:** solve with `options={'threads': 1}` (each overlap test is tiny), or let callers pass HiGHS options.
- **muvis-align:** `util.single_threaded_highs` runs every HiGHS solve with one thread.
- **Post:** new issue. Title: *Overlap linprog (HiGHS) from changing threads can hang the process (Windows)*.

### 11. Elastix: a failed pair, and the transform type

- **Problem:** `registration_ITKElastix` raises `RuntimeError` on a pair Elastix cannot register (e.g. data/3d's
  13-voxel overlap: 'Too many samples map outside moving image buffer'), ending the whole registration, where the
  feature methods return the identity with quality 0. Its default stages end at rigid, whatever transform type the
  rest of the registration uses.
- **Suggestion:** return the identity with quality 0 (and a warning) for a pair that fails; default the stages from
  the requested transform type.
- **muvis-align:** `MVSRegistration.elastix_registration_or_identity`, with `transform_types` translation then the
  project's type.
- **Post:** new issue - **recheck on main first**: Elastix moved to `multiview_stitcher/elastix.py` (#184). Title:
  *registration_ITKElastix: raises on a pair it cannot register, and default stages end at rigid*.

### 12. OME-Zarr writing

- **0.6:** the writers stop at 0.5; muvis-align writes 0.6 metadata through ngff-zarr
  (`fusion_blocks.write_multiscales_metadata`). Upstream already: PR #189 (open), no issue needed.
- **Level names:** levels are named '0', '1', ... '10', which sort as text past 10 levels ('10' after '1'), so
  napari's own reader takes them in the wrong order. Zero-padded names, or readers taking the multiscales list in
  order.
- **Sequential blocks:** `ngff_utils`' zarr writing runs an export's blocks one after another (`n_batch=1`) unless a
  `batch_func` is given; a parallel default would suit most uses. muvis-align runs the blocks on its own workers
  (`fusion_blocks._run_blocks`).
- **Multiscale input:** writing to zarr fuses level 0 and builds the pyramid from it, where the lazy path can fuse
  each level from the sources' own levels. muvis-align fuses each level itself (`fusion_blocks.fuse_levels_to_ome_zarr`).
- **Post:** one new issue for the last three. Title: *ngff_utils writing: level names past 10, sequential blocks,
  multiscale input*.

### 13. Progress of the global resolution

- **Problem:** `groupwise_resolution` is one long blocking call with nothing to hook (no dask tasks, no tqdm); on the
  HPC `global_optimization` ran for hours with no way to tell converging from stuck.
- **Suggestion:** an optional progress callback (outer pass, inner iteration, current residuals).
- **muvis-align:** `GlobalOptProgress` reads `global_optimization`'s debug log messages, so a rewording upstream
  degrades it to a single phase.
- **Post:** new issue. Title: *A progress callback for groupwise_resolution*.

### 14. Registration function dispatch

- **Problem:** `dispatch_pairwise_reg_func` reads the registration function's signature to decide what to pass (image
  data or points), so a wrapper without `functools.wraps` is silently called without `fixed_data`/`moving_data`.
- **Suggestion:** a line in the docstring for custom registration functions, or an explicit flag.
- **Post:** new issue (docs). Title: *Document that dispatch_pairwise_reg_func reads the function's signature*.

### Related: tifffile and zarr

- **Problem:** tifffile's `ZarrTiffStore.get()` does the file read and decode synchronously inside zarr 3's async
  store API, so it runs on zarr's single event-loop thread: every read through the store in a process is serialised
  there, however many threads ask.
- **Measured:** in the slides export, 52% of the fusion threads' samples were waiting in zarr's `sync()`.
- **Suggestion:** run the blocking part in a thread (`asyncio.to_thread`) so reads overlap.
- **muvis-align:** reads a TIFF level stored as one uncompressed run of rows directly
  (`ome_tiff_helper.PicklableTiffLevel`, a seek and read of the rows asked for), other layouts through the store.
- **Post:** new issue on https://github.com/cgohlke/tifffile (or zarr-python). Title: *ZarrTiffStore.get() blocks
  zarr's event loop: concurrent reads are serialised*.

### Related: dask

- **Problem:** dask's linear fusion (`optimization.fuse.active`, still in 2026.8.0) renames a fused chain to a
  115-char prefix plus 4 hex digits of `hash()`. Two similar chains in one compute (crops of different image pairs
  whose key names share a long prefix) then collide: a chain silently gets another chain's result.
- **Measured:** about 1 in 20 computes at 36 pairs, near certain at 256.
- **muvis-align:** runs pair registration and pair metrics with linear fusion off.
- **Post:** new issue on https://github.com/dask/dask - **first check it reproduces on current dask, search its
  issues, and write a minimal reproducer**. Title: *Linear fusion: renamed keys collide between similar chains in
  one graph*.
