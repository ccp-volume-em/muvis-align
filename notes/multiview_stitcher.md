# multiview-stitcher: what muvis-align relies on

Background on the parts of multiview-stitcher muvis-align uses, checked against 0.1.62 (the latest release). What
we would change upstream, with measurements and muvis-align's workaround for each, is in
[multiview_stitcher_suggestions.md](multiview_stitcher_suggestions.md) - numbers below refer to its sections.

## Groupwise resolution (global registration)

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
Registered as a named method, so `groupwise_resolution` runs it per component. Only 'translation' and 'rigid';
other transform types fall back to `global_optimization`. Its rounds swap numpy versions of `linear_two_pass`'s
parameter building and residuals in (1b), and it picks the reference view itself (1c).

## Pairwise registration

- `dispatch_pairwise_reg_func` reads the registration function's signature to decide what to pass (image data or
  points): a wrapper has to keep it (`functools.wraps`), or the function is called without
  `fixed_data`/`moving_data` (14).
- A registration function returns `affine_matrix` and `quality`; on registered pairs `affine_matrix` takes fixed crop
  pixels to the moving crop (see "Transform direction" under Known issues in todo_known_issues.md).
- `registration_binning` overrides the automatic binning (9); split pairing passes it for the stitched planes.
- Registering a fused group needs it channel-selected (no 'c' dim), as register_pairs' own msims are: with one,
  `transform_sim` failed ('affine matrix has wrong number of rows').
- muvis-align registers pairs in worker processes (6). Native pools (BLAS, OpenCV) run at one thread there, which
  changes floating-point summation order: 43 of 1764 pairs differ by up to 0.046um from registering with default
  native threading.

## Overlaps and metrics

- Building the view adjacency graph tests each candidate pair's overlap with a linear program (2), from candidates
  within the largest source's diameter (3), and without a dask scheduler set uses spawned processes (14).
- `metrics.tile_pair_image_metrics` in its overlap mode measures every pair overlapping under the base transform (7).

## Fusion

- `fusion.fuse` builds a fusion plan per output block (8a); `simple_average_fusion` is the 'average' rule (8b).
- Writing to zarr (`ngff_utils`) is replaced in muvis-align by `fusion_blocks.fuse_levels_to_ome_zarr` (12).
