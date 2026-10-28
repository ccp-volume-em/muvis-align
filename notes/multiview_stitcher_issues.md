# Draft upstream issues

Drafts of the issues in [multiview_stitcher_suggestions.md](multiview_stitcher_suggestions.md), ready to post on
https://github.com/multiview-stitcher/multiview-stitcher (and, at the end, tifffile and dask). Not posted yet.
Each says where it goes: a new issue, or a comment on an existing one. Checked against 0.1.62 and main on
2026-10-28; existing issues searched (open and closed).

Order to post: 1-3 first (the global resolution, the main one), then the rest as convenient.
Left out: the dask scheduler for the overlap graph (#144 already answered: a `__main__` guard, documented by dask).

Context paragraph used by several drafts:

> muvis-align (https://github.com/ccp-volume-em/muvis-align) registers and fuses large serial-section EM mosaics
> with multiview-stitcher. Numbers below are from a local dataset (153 tiles in 3 sections, 831-1764 pairs) and a
> full HPC run (34k tiles in 1081 sections, 229725 pairs), with multiview-stitcher 0.1.62.

---

## 1. Robust (reweighted) option for `linear_two_pass`

**Where:** new issue, referencing #101.

**Title:** linear_two_pass: an iteratively reweighted (robust) option

**Body:**

> (context paragraph)
>
> Related to #101: at 34k tiles `global_optimization` does not finish (on our HPC run, the first outer pass took
> ~2h, then ~3 min a pass; 146 passes in 10h with the max residual stuck at ~69.6). `linear_two_pass` is fast but is
> plain weighted least squares plus one pruning step, so bad pairs pull the fit; repeating its pruning on its own
> kept edges prunes down to a spanning tree.
>
> Reweighting `linear_two_pass` instead fits as well as `global_optimization` at close to `linear_two_pass`'s cost.
> Each round re-solves with every edge weighted by its quality x 1 / (1 + (residual / scale)^2) - a Cauchy weight of
> its last residual - with pruning off (`residual_threshold=inf`, `keep_mst=False`, `weight_mode='quality'`). Bad
> pairs stop pulling the fit without being cut, so the graph stays connected.
>
> | rigid, 153 tiles, 831 pairs | time | median residual | p90 residual |
> |---|---|---|---|
> | `linear_two_pass` | 0.9s | 0.111um | 0.482um |
> | reweighted, 9 rounds to converge | 4.3s | 0.109um | 0.284um |
> | `global_optimization` | 83.7s | | |
>
> | 153 tiles, 1764 pairs | time | median residual | p90 residual |
> |---|---|---|---|
> | `linear_two_pass` | 2.0s | 0.211um | 0.894um |
> | reweighted, 10 rounds | 6.5s | 0.134um | 0.434um |
> | `global_optimization` | 354-406s | 0.117um | 0.503um |
>
> At 34k tiles / 229k pairs the median residual settled at 0.104um by round 3.
>
> **Suggestion:** a `robust_rounds` option (with a residual `scale`, e.g. defaulting to half the voxel diagonal) on
> `linear_two_pass`. Stopping rule: once the 99th percentile of the residuals' change in a round is within a fraction
> (we use 1%) of the scale - on a large graph a few edges keep flipping every round, so a stop on the largest change
> never triggers.
>
> Our implementation, registered as a groupwise resolution method:
> https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/robust_resolution.py
> (`groupwise_resolution_robust_linear`). Happy to turn it into a PR if the approach suits.

## 2. `linear_two_pass`: skip the second pass when nothing is pruned, build the transforms once

**Where:** new issue, referencing #139.

**Title:** linear_two_pass: most of a call is building xarray transforms, and pass 2 repeats pass 1 when nothing is pruned

**Body:**

> (context paragraph)
>
> Profiling `groupwise_resolution_linear_two_pass` (9 calls, pruning off, 153 tiles): 9.7s in all, of which the sparse
> solves are 0.15s, `_build_params_from_components` 6.5s (18 calls: after pass 1 and after pass 2) and
> `compute_edge_residuals` 2.7s. On the HPC graph (34k tiles) a call took ~10 min.
>
> - Pass 2 re-solves even when no edge was pruned, giving pass 1's result again: about a third of every call.
> - `_build_params_from_components` builds an xarray transform per node, twice a call.
> - `compute_edge_residuals` builds the bead graph with an identity xarray per node it never reads, and converts an
>   xarray transform for every point it transforms.
>
> On #139 the xarray overhead was judged within limits as it is per tile once; here it is per tile per call, and at
> 34k tiles it dominates.
>
> **Suggestion:** skip pass 2 (and its parameter build) when every edge is kept; keep the per-node components as numpy
> between passes and build the xarray transforms once, for the result returned; compute residuals on plain matrices.
>
> We swap numpy versions of both functions in while our own rounds run (`_build_params_from_templates`,
> `_edge_residuals` in
> https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/robust_resolution.py): identical transforms
> and residuals, a round 8x faster (3600 tiles, 3 rounds: 24.8s -> 3.1s), 34k tiles / 230k edges ~31s a round on one
> thread.

## 3. Reference view search is O(nodes x edges)

**Where:** new issue.

**Title:** get_node_with_maximal_edge_weight_sum_from_graph scans every edge once per node

**Body:**

> (context paragraph)
>
> `mv_graph.get_node_with_maximal_edge_weight_sum_from_graph` sums each node's edge weights by scanning all edges once
> per node, so it is O(nodes x edges). `global_optimization` and `linear_two_pass` call it whenever no
> `reference_view` is given, or the given one is not in the component. Measured at 1.2us a node-edge on the component
> subgraph view: ~2.7h at 34k nodes x 229k edges. Our HPC run spent 4.8h before `global_optimization`'s first
> iteration.
>
> **Suggestion:** accumulate each node's weight sum in a single pass over the edges - same node chosen:
>
> ```python
> totals = dict.fromkeys(graph.nodes, 0.0)
> for node1, node2, data in graph.edges(data=True):
>     totals[node1] += data[weight_key]
>     totals[node2] += data[weight_key]
> return max(totals, key=totals.get)
> ```
>
> (3ms locally against 57-267ms.) Ours: `find_reference_view` in
> https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/robust_resolution.py. Can open a PR.

## 4. `global_optimization` at scale

**Where:** comment on #101.

**Body:**

> Some numbers for this from a 34k-tile run (229725 pairs, 0.1.62): `global_optimization`'s first outer pass took ~2h
> (323 inner iterations), then ~3 min a pass; it removes one edge per pass, re-running the inner loop and deep-copying
> the graph per removal candidate. 146 passes in 10h with the max residual stuck at ~69.6: it would not have finished.
> Locally (153 tiles, 1764 pairs) 967 passes, 354-406s.
>
> Two ideas: remove several edges per pass (all above a residual threshold, keeping connectivity) and avoid the
> per-candidate deep copies; or point large graphs to a robust linear method - see #(issue 1), which gets close to
> `global_optimization`'s fit in seconds locally and ~31s a round at 34k tiles.

## 5. View adjacency graph: exact box intersections when nothing is rotated

**Where:** new issue.

**Title:** View adjacency graph: skip the linear program per pair when all transforms are translations

**Body:**

> (context paragraph)
>
> Building the view adjacency graph solves a linear program (`linprog`) per candidate pair to get the overlap. On our
> HPC run that took 2h16 for 233k pairs. Before registration, every source's transform is usually a translation (or
> translation and scale), so each overlap is just the intersection of two axis-aligned boxes.
>
> **Suggestion:** when all transforms are axis-aligned, take each overlap as the bounding boxes' intersection,
> vectorised over all pairs; keep `linprog` for the rest. Locally 4.5s -> 0.3s, edges identical, overlaps within
> 3e-16.
>
> Ours: `build_view_adjacency_graph` in
> https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/image/util.py

## 6. Default pair search radius with one large source

**Where:** new issue.

**Title:** Pair candidates within the largest source's diameter: one overview image makes every pair a candidate

**Body:**

> (context paragraph)
>
> With no pairs given, candidates come from a cKDTree query with a radius of the largest source's diameter. One large
> source (in our case an overview image among the tiles) makes nearly every pair a candidate, each a delayed overlap
> task. Our HPC run (34k sources) sat at 0% for 34+ min with rss growing 231 -> 344GB (~1.2 billion candidates).
> Locally (51 sources) 2550 candidates with the overview against 436 without, for 179 real overlaps.
>
> **Suggestion:** a sweep over the sources' bounding boxes (each source's own extent) instead of one global radius.
> Ours, which we pass in as the pairs: `find_candidate_overlap_pairs` in
> https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/image/util.py - same edges, graph build
> 13.2s -> 1.1s.

## 7. Phase correlation: link quality for the kept shift only

**Where:** new issue.

**Title:** phase_correlation_registration computes the link quality of every candidate shift

**Body:**

> (context paragraph)
>
> `phase_correlation_registration` computes a spearman link quality for every candidate shift (~11 a pair) and keeps
> one. With 51 sources, spearman was 160s of 238s scoring, in 302s of pair CPU. Computing it only for the chosen
> candidate gives identical results: pair CPU 432s -> 217s, the pair registration 129.5s -> 90.6s.
>
> **Suggestion:** compute the link quality once, for the chosen candidate. We defer it with a wrapper
> (`deferred_link_quality` / `resolve_deferred_quality` in
> https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/MVSRegistration.py).

## 8. Phase correlation: disambiguation that prefers zero shift

**Where:** new issue.

**Title:** phase_correlation_registration picks zero shift when both images share an outline or a fixed pattern

**Body:**

> (context paragraph)
>
> The candidate shift is chosen by SSIM over the union (or, with NaNs, the intersection) bounding box, with NaN pixels
> read as 0. When both images share an outline, a background or a fixed pattern at the same place - fused serial
> sections: same tile layout, seams and shading - that box matches best at zero shift, whatever the content does. The
> phase-normalised candidate also locks onto such a pattern.
>
> Measured on fused sections on one grid: skimage's unnormalised phase correlation found (8.9, 5.0)um where
> `phase_correlation_registration` returned exactly 0 on the same images. With the background as NaN, 3 of 5 section
> pairs came out right; the other 2 still chose 0 (NaN corners inside the intersection box). skimage's masked
> normalised cross-correlation found all 5 (NCC 0.52-0.92 at the shift found).
>
> **Suggestion:** score candidates only where both images have data (a mask, not the bounding box), and offer the
> masked cross-correlation candidate whenever a mask is known, not only when NaNs are present.

## 9. Pairwise registration in processes

**Where:** comment on #102.

**Body:**

> Related numbers: in threads, pair registration is GIL-bound - on our HPC run 64 threads did ~3 cores of pair work.
> Registering each pair in a spawned worker process (native pools at one thread there) instead: 1764 pairs
> 168.8s -> 111.7s (Windows), 271.8s -> 99.7s with peak rss 3.1 -> 0.47GB (Linux). It needs the source arrays to
> pickle. Ours: `register_pair_in_worker` in
> https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/MVSRegistration.py.
> One side effect worth knowing: one-thread native pools change floating-point summation order, so 43 of 1764 pairs
> differed by up to 0.046um from the threaded run.

## 10. Registration metrics per registered pair

**Where:** new issue.

**Title:** tile_pair_image_metrics: measure the registered pairs, not every overlapping pair

**Body:**

> (context paragraph)
>
> In its overlap mode (`query_transform_keys`, no `pairs_graph`), `metrics.tile_pair_image_metrics` measures every
> pair that overlaps under the base transform: 1764 pairs where 831 were registered locally, 3.1 min and 6.5GB peak.
> Called per registered pair with only that pair's two msims: 43.5s, identical values.
>
> **Suggestion:** accept the pairs (or the registration graph) to measure.

## 11. Fusion: the plan once per fusion, not once per block

**Where:** new issue.

**Title:** Writing to zarr rebuilds the fusion plan over all sources for every output block

**Body:**

> (context paragraph)
>
> Writing to zarr, `_fuse_chunk_to_zarr` calls `fuse()` for every output block with that block as the output, so the
> fusion plan (`_build_spatial_fusion_plan`, `_get_axis_aligned_translation_dims`,
> `_get_grid_aligned_translation_dims`, `sim_sel_coords`) is rebuilt over all sources each time, with xarray `.sel`
> calls on every source's transform - although the plan already maps each source to the chunks it meets.
>
> Measured (328 sources, 1280 blocks of 3840x3840 px, one block at a time): 2.7s a block, of which `sim_sel_coords`
> ~1.1s (328 calls), the plan ~0.94s, the axis-aligned check ~0.74s (~4270 `.sel` calls); `affine_transform` ~2%. That
> is Python holding the GIL: 24 threads reach ~1.5 cores. The cost per block grows with the total source count, so at
> 34k sources a block would cost ~100x more.
>
> **Suggestion:** build the plan (and the per-source transform checks, on plain numpy arrays) once per fusion and hand
> each block only the sources it meets.
>
> Ours fuses each block from only the sources that reach it, by multiview-stitcher's own rule
> (`fuse_into_zarr_array`, `_numpy_fusion_plan` in
> https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/fusion_blocks.py): identical output, an
> export ~31 -> ~9 min.

## 12. `simple_average_fusion` in one pass

**Where:** new issue (or straight to a PR).

**Title:** simple_average_fusion: one pass, bit-identical, ~3x faster

**Body:**

> To count the views reaching each pixel, `simple_average_fusion` builds, per view, a stack of the running count and
> that view's mask and takes its `nansum`; then `nansum` over all views copies the whole stack once more to replace
> NaNs. Profiled on fused blocks, it was 22% of a block's time.
>
> In one pass - one validity mask, counts summed from it, values summed with NaNs zeroed by `np.where` - the same sums
> in the same order: bit-identical on every level of a whole fusion we compared, 3.2x faster on a 3-view 4096x4096
> block, a whole fusion ~19% faster. The same applies to a `np.nanmean` combine (2.2x).
>
> Ours: https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/fusion_methods/FusionMethodAverage.py.
> Can open a PR.

## 13. Registration binning budget for 2D images

**Where:** new issue.

**Title:** get_optimal_registration_binning: the 400**3 pixel budget leaves large 2D images unbinned

**Body:**

> `registration.get_optimal_registration_binning` bins a pair only while it holds more than
> `max_total_pixels_per_stack = 400**3` (64M) pixels. That suits 3D stacks, but a 2D image of up to 64M pixels is
> registered unbinned. Registering whole fused sections (6051x6801 px, 41M): phase correlation 90s and 5.9GB a pair,
> SIFT over 10GB a pair. Binned by 8: SIFT 2.3 min for the whole set, peak 2.0GB, the same alignment.
>
> **Suggestion:** a budget per dimensionality (e.g. 400**ndim, or a 2D default of the order of 1-4M pixels).
> (#183 projects a singleton spatial dim; the budget is unchanged on main.)

## 14. HiGHS threads in the overlap tests

**Where:** new issue.

**Title:** Overlap linprog (HiGHS) from changing threads can hang the process (Windows)

**Body:**

> The overlap tests solve a linear program with scipy's `linprog` (HiGHS). HiGHS starts ~11 workers per calling
> thread and tears them down when that thread exits. Called from changing pool threads (a GUI's, or a thread pool's),
> a teardown spun in its TLS callback holding the Windows loader lock, so no thread could start: the process hung, or
> raised access violations after a pass or two. Found with native stacks (cdb); a 120-run stress loop hung after ~20
> runs, and with one HiGHS thread was clean.
>
> **Suggestion:** solve with `options={'threads': 1}` - each overlap test is tiny - or let callers pass HiGHS options.
> Ours wraps `linprog` (`single_threaded_highs` in
> https://github.com/ccp-volume-em/muvis-align/blob/main/src/muvis_align/util.py).

## 15. Elastix: a failed pair, and the transform type

**Where:** new issue - **recheck first on main**: Elastix moved to `multiview_stitcher/elastix.py` (#184).

**Title:** registration_ITKElastix: raise on a pair it cannot register, and default stages end at rigid

**Body:**

> - On a pair Elastix cannot register (e.g. a 13-voxel overlap: 'Too many samples map outside moving image buffer')
>   `registration_ITKElastix` raises `RuntimeError`, which ends the whole registration. The feature-based methods
>   return the identity with quality 0 for a pair they cannot register, and the global resolution then leaves it out.
> - Its default stages end at rigid, whatever transform type the rest of the registration uses.
>
> **Suggestion:** return the identity with quality 0 (and a warning) for a failed pair; default the stages from the
> requested transform type.

## 16. OME-Zarr writing

**Where:** new issue. (0.6 itself is PR #189: no issue.)

**Title:** ngff_utils writing: level names past 10, sequential blocks, multiscale input

**Body:**

> A few things we worked around writing large OME-Zarr exports:
>
> - **Level names:** levels are named '0', '1', ... '10', which sort as text past 10 levels ('10' after '1'), so
>   napari's own reader takes them in the wrong order. Zero-padded names, or readers taking the multiscales list in
>   order.
> - **Sequential blocks:** zarr writing runs an export's blocks one after another (`n_batch=1`) unless a `batch_func`
>   is given. A parallel default would suit most users.
> - **Multiscale input:** writing to zarr fuses level 0 and builds the pyramid from it, where the lazy path can fuse
>   each level from the sources' own levels.

## 17. Progress of the global resolution

**Where:** new issue.

**Title:** A progress callback for groupwise_resolution

**Body:**

> `groupwise_resolution` is one long blocking call with nothing to hook (no dask tasks, no tqdm); on a large graph
> `global_optimization` runs for hours with no way to tell converging from stuck. We read progress from its debug
> log messages, which breaks on any rewording.
>
> **Suggestion:** an optional progress callback (outer pass, inner iteration, current residuals).

## 18. Registration function dispatch

**Where:** new issue (docs).

**Title:** Document that dispatch_pairwise_reg_func reads the function's signature

**Body:**

> `dispatch_pairwise_reg_func` reads the registration function's signature to decide what to pass (image data or
> points), so a wrapper without `functools.wraps` is silently called without `fixed_data`/`moving_data`. Worth a line
> in the docstring for custom registration functions, or an explicit flag.

---

## tifffile: blocking reads in `ZarrTiffStore`

**Where:** new issue on https://github.com/cgohlke/tifffile (or zarr-python).

**Title:** ZarrTiffStore.get() blocks zarr's event loop: concurrent reads are serialised

**Body:**

> `ZarrTiffStore.get()` does the file read and decode synchronously inside zarr 3's async store API, so it runs on
> zarr's single event-loop thread: every read through the store in a process is serialised there, however many
> threads ask. Fusing a 328-file TIFF dataset, 52% of the fusion threads' samples were waiting in zarr's `sync()`.
> Running the blocking part in a thread (`asyncio.to_thread`) would let reads overlap.

## dask: linear fusion key collisions

**Where:** new issue on https://github.com/dask/dask - **first check it reproduces on current dask and search
existing issues.**

**Title:** Linear fusion: renamed keys collide between similar chains in one graph

**Body:**

> With `optimization.fuse.active`, a fused linear chain is renamed by `default_fused_keys_renamer` to a 115-character
> prefix plus 4 hex digits of `hash()`. Two similar chains in one compute (in our case, crops of different image pairs
> whose key names share a long prefix) then collide about 1 in 20 computes at 36 chains, near certain at 256: a chain
> silently gets another chain's result. Seen with dask 2026.8.0. A minimal reproducer still to write.
