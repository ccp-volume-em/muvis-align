# multiview-stitcher: behaviour, workarounds and upstream candidates

What muvis-align relies on, works around or would like changed in multiview-stitcher (checked
against 0.1.62, the latest release; its main branch has the same resolution methods). Measurements
are local (153 sources, 1764 pairs) or from the HPC run (34k sources, 229725 pairs) unless noted.

## Groupwise resolution (global registration)

`param_resolution.groupwise_resolution(g_reg, method, **kwargs)` runs a resolver per timepoint and
connected component. Resolvers are registered by name with `register_groupwise_resolution_method`;
three ship with it:

- `global_optimization` (the default): iterative per-node point fitting on virtual beads (the
  overlap bbox corners), then removes **one** edge per outer pass (the worst by a residual/quality
  score, keeping the graph connected) until the max residual is below `abs_tol` (default: the voxel
  diagonal). Each pass re-runs the inner loop (up to `max_iter`, default 500) and deep-copies the
  graph per removal candidate. Locally: 967 passes, 354-406s. HPC: pass 1 took ~2h (323 iterations
  over 34k nodes), then ~3 min a pass; 146 passes in 10h with the max residual stuck at ~69.6 - it
  would not finish.
- `linear_two_pass`: a linearised sparse least-squares solve (rotations, then translations, with
  `scipy.sparse.linalg.lsqr`), residuals from `utils.compute_edge_residuals`, then one pruning step
  (edges above `residual_threshold`, or median + `mad_k` x MAD; `keep_mst` keeps a spanning tree)
  and a second solve. Only 'translation' and 'rigid'. Edge weights from `weight_mode`
  ('quality_overlap', 'quality', 'overlap', 'uniform'). Locally 2.0s, but a worse fit: median
  residual 0.211um against 0.117um for global_optimization on all edges - plain least squares, so
  bad pairs pull the solution. Tried: repeating it on its own kept edges prunes down to a spanning
  tree (median 0.217-0.229um); an absolute threshold of the voxel diagonal, 0.215um.
- `shortest_paths`: transforms composed along shortest paths from the reference view; no
  averaging over redundant edges.

There is no reweighting (robust) option. PR #121 added `linear_two_pass` as the "fast yet robust"
method; its robustness is the single pruning step.

### Robust linear (muvis-align's wrapper)

Iteratively reweighted least squares with `linear_two_pass` as the solver: each round sets an
edge's quality to its pair quality times a Cauchy weight of its last residual,
1 / (1 + (residual / scale)^2), and re-solves with pruning off (`mad_k` very large,
`keep_mst=False`, `weight_mode='quality'`). Residuals come from `linear_two_pass`'s own `residual`
metric. Registered as a named method, so `groupwise_resolution` runs it per component.

Prototype, 10 rounds, local (median / p90 residual on all edges):

| method | time | median | p90 |
|---|---|---|---|
| global_optimization | 354-406s | 0.117um | 0.503um |
| linear_two_pass | 2.0s | 0.211um | 0.894um |
| robust linear, scale 0.2um | ~23s | 0.090um | 0.564um |
| robust linear, scale 0.35um | ~23s | 0.134um | 0.434um |

(voxel diagonal 0.70um). As implemented (`muvis_align/robust_resolution.py`, method name
`robust_linear`, scale = half the voxel diagonal, residuals reused from `linear_two_pass`): 6.5s,
median 0.134um / p90 0.434um, as the prototype. Select it with the registration parameter
`groupwise_resolution_method: robust_linear`; only 'translation' and 'rigid' (others fall back to
global_optimization). At HPC size (synthetic grid, 34040 tiles, 269374 edges with the local pairs'
attributes, one component): reference view 0.4s, robust_linear 1283s (~2 min a round), peak rss
3.5GB; the median residual settles by round 6-7 (0.809 -> 0.459 -> 0.445). Upstream candidate: a number of robust rounds as a `linear_two_pass`
option, which would make the wrapper unnecessary.

### Reference view search is O(nodes x edges)

`mv_graph.get_node_with_maximal_edge_weight_sum_from_graph` sums each node's edge weights by
scanning every edge once per node. On the component subgraph view `groupwise_resolution` passes
in, 1.2us a node-edge locally: ~2.7h at 34k nodes x 229k edges, plausibly the 4.8h the HPC spent
before global_optimization's first iteration. Both `global_optimization` and `linear_two_pass`
call it when no `reference_view` is given (and when the given one is not in the component).
Workaround (`robust_resolution.find_reference_view`, same node, 3ms locally against 57-267ms): the
robust method picks it per component itself; register_global passes it to any method when the
graph is connected (a reference outside a component sends that component back to the slow search). Upstream candidate: accumulate per node in a single pass over the edges.

## Pairwise registration

- `phase_correlation_registration` computes a spearman link quality for every candidate shift
  (~11 a pair) and keeps one: muvis-align defers it and computes only the kept one's
  (`deferred_link_quality` / `resolve_deferred_quality`) - identical results, pair CPU halved.
  Upstream candidate.
- `dispatch_pairwise_reg_func` looks at the registration function's signature to decide what to
  pass (image data or points): any wrapper has to keep it (`functools.wraps`), or the function is
  called without `fixed_data`/`moving_data`.
- The default pair search (no pairs given) uses a cKDTree radius of the largest source's
  diameter: with overview images among the tiles, nearly every pair of 34k sources is a candidate.
  muvis-align hands it bounding-box sweep candidates (`find_candidate_overlap_pairs`).
- Building the view adjacency graph without a dask scheduler set computes overlaps with spawned
  processes; register_pairs and the metrics set a scheduler.
- In threads, pair registration is GIL-bound (64 threads on the HPC did ~3 cores of pair work).
  muvis-align registers each pair in a spawned worker process instead; native pools (BLAS,
  OpenCV) run at one thread there. That changes floating-point summation order: 43 of 1764 pairs
  differ by up to 0.046um from registering with default native threading.

## Registration metrics

- `metrics.tile_pair_image_metrics` in its overlap mode (`query_transform_keys`, no `pairs_graph`)
  measures every pair that overlaps under the base transform: 1764 pairs for 831 registered locally,
  3.1 min and 6.5GB peak. muvis-align calls it per registered pair with only that pair's two msims
  (43.5s, identical values), in worker processes when there are more pairs than workers.
- The overlap mode computes each overlap with scipy's `linprog` (HiGHS). Called from changing pool
  threads it hung and raised access violations after a pass or two; in the calling thread (or in a
  worker process's own thread) it does not. So without worker processes the global metrics run one
  pair at a time in the calling thread.
- `faulthandler` on Windows also reports access violations that native code handles itself: a
  "Windows fatal exception: access violation" line alone is not the failure - look for the hang or
  exit that follows.

## Split pairing (muvis-align)

Two stages with multiview-stitcher's own pieces: stage 1 is the usual pairwise + groupwise registration
with pairs only within each z-plane or channel - `groupwise_resolution` resolves each connected
component on its own, so each group is stitched independently. Stage 2 fuses each group
(`fusion.fuse`, reduced resolution) and registers consecutive groups with the same pairwise function
(`compute_pairwise_registrations`), then `groupwise_resolution` over the groups. Registration needs the
fused groups channel-selected (no 'c' dim), as register_pairs' own msims are: with it, `transform_sim`
failed ('affine matrix has wrong number of rows').

## dask

- Linear fusion (`optimization.fuse.active`) can give two pairs' fused crop chains the same key
  (dask's `default_fused_keys_renamer` keeps a 115-char prefix plus 4 hash digits), handing a pair
  another pair's crop. Pair registration and the pair metrics run with it off.
