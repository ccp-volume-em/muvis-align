# Suggested multiview-stitcher improvements

Changes to multiview-stitcher that came out of muvis-align's work, each with what we measured, what we suggest,
and how muvis-align works around it today. Checked against multiview-stitcher 0.1.62. "Local" is the 3-section
test data (153 sources, 1764 overlapping / 831 orthogonal pairs); "HPC" is the full run (34k sources, 1081
sections, 229725 pairs). Background on each area: [multiview_stitcher.md](multiview_stitcher.md).

Ordered by impact at HPC scale; global resolution (1) is the main one.

## 1. Global resolution: a robust linear method that scales

The main suggestion. At HPC scale `global_optimization` does not finish and `linear_two_pass` fits poorly where
pairs are bad; reweighting `linear_two_pass` fits as well as `global_optimization` at close to `linear_two_pass`'s
cost. Four parts, each useful on its own.

### 1a. A robust option for `linear_two_pass`

- **Problem:** `linear_two_pass` is plain weighted least squares plus one pruning step (median + 2 x MAD, a
  spanning tree kept), so bad pairs pull the fit; repeating the pruning on its own kept edges prunes down to a
  spanning tree.
- **Measured:** the table below. HPC (34k sources, 229725 pairs): ~10 min a round; the median residual settled
  at 0.104um by round 3.

| method | time | median residual | p90 residual |
|---|---|---|---|
| **local, rigid, 153 sources, 831 orthogonal pairs** | | | |
| `linear_two_pass` | 0.9s | 0.111um | 0.482um |
| `linear_two_pass` reweighted (`robust_linear`, 9 rounds to converge) | 4.3s | 0.109um | 0.284um |
| `global_optimization` | 83.7s | | |
| **local, 1764 overlapping pairs (earlier)** | | | |
| `linear_two_pass` | 2.0s | 0.211um | 0.894um |
| `linear_two_pass` reweighted (10 rounds) | 6.5s | 0.134um | 0.434um |
| `global_optimization` | 354-406s | 0.117um | 0.503um |

- **Suggestion:** a `robust_rounds` option (with a residual `scale`) on `linear_two_pass`: iteratively reweighted
  least squares, each round re-solving with every edge weighted by its quality x 1 / (1 + (residual / scale)^2) -
  a Cauchy weight of its last residual, so bad pairs stop pulling the fit without being cut - and pruning off.
  Stop once the 99th percentile of the residuals' change in a round is within a fraction of the scale: on the HPC
  the largest change never settled (a few of 229k edges flip every round), so a max-based stop ran every round.
- **muvis-align:** `robust_resolution.groupwise_resolution_robust_linear`, registered as `robust_linear`; the
  plugin's default global method. Scale: half the voxel diagonal; tolerance 1% of it.

### 1b. `linear_two_pass`: the cost is building the parameters, not solving

- **Problem:** per call, `_build_params_from_components` builds an xarray transform per node twice (after pass 1
  and pass 2), and pass 2 re-solves even when no edge was pruned - then giving pass 1's result again.
- **Measured (local, profiled, 9 calls with pruning off):** 9.7s in all, of which the sparse solves 0.15s,
  `_build_params_from_components` 6.5s (18 calls), `compute_edge_residuals` 2.7s. So about a third of every call
  repeats pass 1 when nothing is pruned, and two thirds are xarray construction. On the HPC a call is ~10 min.
- **Suggestion:** skip pass 2 (and its parameter build) when every edge is kept; keep the per-node components in
  numpy between passes and build the xarray transforms once, for the result returned.
- **muvis-align:** nothing - the solver is reached only through `groupwise_resolution_linear_two_pass`.

### 1c. Reference view search: one pass over the edges

- **Problem:** `mv_graph.get_node_with_maximal_edge_weight_sum_from_graph` scans every edge once per node, so it
  is O(nodes x edges). `global_optimization` and `linear_two_pass` call it whenever no `reference_view` is given,
  or the given one is not in the component.
- **Measured:** 1.2us a node-edge locally on the component subgraph view: ~2.7h at 34k nodes x 229k edges. The
  HPC spent 4.8h before global_optimization's first iteration.
- **Suggestion:** accumulate each node's weight sum in a single pass over the edges. Same node chosen.
- **muvis-align:** `robust_resolution.find_reference_view` does that (3ms locally against 57-267ms);
  `register_global` passes it when the graph is connected.

### 1d. `global_optimization` at scale

- **Problem:** it removes one edge per outer pass, re-running the inner loop (up to 500 iterations) and
  deep-copying the graph for each removal candidate.
- **Measured:** 967 passes locally (354-406s); on the HPC pass 1 took ~2h, then ~3 min a pass, 146 passes in 10h
  with the max residual stuck at ~69.6. It would not finish.
- **Suggestion:** remove several edges per pass (e.g. all above a residual threshold, keeping connectivity), and
  avoid the per-candidate deep copies - or point large graphs to the robust linear method (1a).
- **muvis-align:** uses `robust_linear` (1a) for translation and rigid.

## 2. View adjacency graph: exact box intersections when nothing is rotated

- **Problem:** building the view adjacency graph solves a linear program (`linprog`) per candidate pair.
- **Measured:** 2h16 for 233k pairs on the HPC; locally 4.5s -> 0.3s with the change below, edges identical and
  overlaps within 3e-16.
- **Suggestion:** when all transforms are translations only (every source before registration), take each
  overlap as the axis-aligned bounding boxes' intersection, all pairs at once; keep `linprog` for the rest.
- **muvis-align:** `image.util.build_view_adjacency_graph` does this for the pairs it is given.

## 3. Default pair search radius

- **Problem:** with no pairs given, candidates come from a cKDTree radius of the largest source's diameter. One
  large source (an overview image) makes nearly every pair a candidate, each a delayed overlap task.
- **Measured:** the HPC run sat at 0% for 34+ min with rss 231 -> 344GB (~1.2 billion candidates at 34k).
  Locally (51 sources) 2550 candidates with the overview against 436 without, for 179 real overlaps.
- **Suggestion:** a sweep over bounding boxes (per-source extents) instead of one global radius.
- **muvis-align:** `image.util.find_candidate_overlap_pairs` hands multiview-stitcher bounding-box sweep
  candidates: same edges, graph build 13.2s -> 1.1s.

## 4. Phase correlation: link quality for the kept shift only

- **Problem:** `phase_correlation_registration` computes a spearman link quality for every candidate shift (~11
  a pair) and keeps one.
- **Measured:** spearman was 160s of 238s scoring in 302s of pair CPU (51 sources). Deferring it: identical
  results, pair CPU 432s -> 217s, register_pairs 129.5s -> 90.6s.
- **Suggestion:** compute the link quality only for the chosen candidate.
- **muvis-align:** `MVSRegistration.deferred_link_quality` / `resolve_deferred_quality`.

## 5. Phase correlation: disambiguation that prefers zero shift

- **Problem:** the candidate shift is chosen by SSIM over the union (or, with NaNs, the intersection) bounding box,
  with NaN pixels read as 0. When both images share an outline, a background or a fixed pattern at the same place
  (fused serial sections: same tile layout, seams and shading), that box matches best at zero shift, whatever the
  content does. The phase-normalised candidate also locks on such a pattern.
- **Measured (fused sections, one grid):** skimage's unnormalised phase correlation found (8.9, 5.0)um where
  `phase_correlation_registration` returned exactly 0 on the same images. With the background as NaN, 3 of 5
  section pairs came out right; the other 2 still chose 0 (NaN corners inside the intersection box). skimage's
  masked normalised cross-correlation found all 5 (NCC 0.52-0.92 at the shift found).
- **Suggestion:** score candidates only where both images have data (a mask, not the bounding box), and offer the
  masked cross-correlation candidate whenever a mask is known, not only when NaNs are present.
- **muvis-align:** split pairing fuses the sections onto one grid, smooths past the tile pattern and fills the
  background before registering; with large shifts relative to the section, a feature method (SIFT) works where
  phase correlation does not.

## 6. Pairwise registration in processes

- **Problem:** `compute_pairwise_registrations` runs pairs in threads, and pair registration is GIL-bound.
- **Measured:** 64 threads did ~3 cores of pair work on the HPC. In spawned worker processes (native pools at one
  thread): local 1764 pairs 168.8s -> 111.7s (Windows), 271.8s -> 99.7s with peak rss 3.1 -> 0.47GB (Linux).
- **Suggestion:** a process-based option, which needs the source arrays to pickle.
- **muvis-align:** `register_pairs` registers each pair in a worker process (`register_pair_in_worker`), with
  `PicklableTiffLevel` so TIFF levels can be sent.

## 7. Registration metrics per registered pair

- **Problem:** `metrics.tile_pair_image_metrics` in its overlap mode measures every pair that overlaps under the
  base transform, not the pairs that were registered; each overlap is a `linprog` (HiGHS) call, which hung or
  crashed with access violations when called from changing pool threads.
- **Measured:** 1764 pairs for 831 registered locally, 3.1 min and 6.5GB peak; per registered pair 43.5s,
  identical values.
- **Suggestion:** accept the pairs (or the registration graph) to measure.
- **muvis-align:** calls it per registered pair with only that pair's two msims, in worker processes when there
  are more pairs than workers; without workers, one pair at a time in the calling thread.

## 8. Fusion: the plan once per fusion, not once per block

- **Problem:** writing to zarr, `_fuse_chunk_to_zarr` calls `fuse()` for every output block with that block as
  the output, so the fusion plan (`_build_spatial_fusion_plan`, `_get_axis_aligned_translation_dims`,
  `_get_grid_aligned_translation_dims`, `sim_sel_coords`) is rebuilt over all sources each time, with xarray
  `.sel` calls on every source's transform. The plan already maps each source to the chunks it meets.
- **Measured (slides, 328 sources, 1280 blocks of 3840x3840 px, one block at a time):** 2.7s a block, of which
  `sim_sel_coords` ~1.1s (328 calls), the plan ~0.94s, the axis-aligned check ~0.74s, ~4270 `.sel` calls;
  `affine_transform` ~2%. That is Python holding the GIL: 24 threads reach ~1.5 cores. The cost per block grows
  with the total source count, so at 34k sources a block would cost ~100x more.
- **Suggestion:** build the plan (and the per-source transform checks, on plain numpy arrays) once per fusion and
  hand each block only the sources it meets.
- **muvis-align:** for a stack, `fusion_slabs.fuse_to_zarr_by_z_slabs` fuses each z-slab of blocks from only the
  sources that reach it, by multiview-stitcher's own rule, through `prepare_block_fusion(create_output=False)`
  (identical output). Slides export ~31 -> ~9 min, ~1.5 -> ~4.3 cores, together with direct TIFF reads (see the
  tifffile item below; ~15-20% of that).

## 9. Registration binning budget for 2D images

- **Problem:** `registration.get_optimal_registration_binning` bins a pair only while it holds more than
  `max_total_pixels_per_stack = 400**3` (64M) pixels. That budget suits 3D stacks; a 2D image of up to 64M pixels
  is registered unbinned.
- **Measured:** split pairing registers whole fused sections (6051x6801 px, 41M, at 0.032um): phase correlation 90s
  and 5.9GB a pair, SIFT over 10GB a pair (a plugin run reached 16.6GB). Binned by 8: SIFT 2.3 min for the whole
  slides registration, peak 2.0GB, the same section alignment.
- **Suggestion:** a budget per dimensionality (e.g. 400**ndim, or a 2D default of the order of 1-4M pixels).
- **muvis-align:** passes `registration_binning` for the section pairs, from the `split_binning` setting (default 8).

## 10. Smaller items

- **Scheduler for the overlap graph:** without a dask scheduler set, building the view adjacency graph computes
  overlaps with spawned processes; a script without a `__main__` guard then re-runs itself in each (>10GB). A
  threads default would avoid it. muvis-align sets a scheduler around every call.
- **Registration function dispatch:** `dispatch_pairwise_reg_func` reads the function's signature to decide what to
  pass, so a wrapper without `functools.wraps` is called without `fixed_data`/`moving_data`. Worth documenting,
  or an explicit flag.

## Related: tifffile and zarr, not multiview-stitcher

tifffile's `ZarrTiffStore.get()` is blocking code (file read and decode) inside zarr 3's async store API, so it
runs on zarr's one event-loop thread: every read through the store in a process is serialised there, however many
threads ask. In the slides export, 52% of the fusion threads' samples were waiting in zarr's `sync()`. muvis-align
reads a TIFF level stored as one uncompressed run of rows directly (`ome_tiff_helper.PicklableTiffLevel`, a seek and
read of the rows asked for), other layouts still through the store. Worth raising with tifffile (e.g. running
`get()` in a thread) or zarr.

## Related: dask, not multiview-stitcher

dask's linear fusion (`optimization.fuse.active`, still in 2026.8.0) renames a fused chain to a 115-char prefix
plus 4 hex digits of `hash()`. Two pairs' crop chains in one compute then collide about 1 in 20 computes at 36
pairs, near certain at 256: a pair silently registers another pair's crop. muvis-align runs pair registration and
pair metrics with it off. Worth reporting to dask.
