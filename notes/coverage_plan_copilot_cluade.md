Copilot Claude Sonnet 5

# Test Coverage Improvement Plan

## Context

A full local `pytest --cov` run on Windows stalled at ~84% (large clusters of `E`
errors), consistent with napari/Qt widget tests hanging without a display in a
local interactive session. The plan below is based on static analysis of
`src/muvis_align` (lines of code) cross-referenced against existing dedicated
test files in `tests/`. The real coverage baseline should be captured from CI
(GitHub Actions), where `DISPLAY`/`XAUTHORITY` are already passed through to
`tox`.

## Coverage gap map

| File | LOC | Dedicated tests today | Priority |
|---|---|---|---|
| `MVSRegistration.py` | 1300 | Only indirect (`test_copy_transforms_dimension_mismatch.py`, `test_register_pairs_transform_mismatch.py`) | Tier 1 |
| `image/util.py` | 1073 | Partial (`test_image_util_transforms.py`) | Tier 1 |
| `util.py` (top-level) | 707 | None | Tier 1 |
| `ui/Interface.py` | 630 | Partial (`test_napari_interface_registration.py`) | Tier 2 |
| `metrics.py` | 195 | None | Tier 1 |
| `image/TiffDaskSource.py` / `DaskSource.py` / `ZarrDaskSource.py` | 135/127/69 | None | Tier 2 |
| `image/ome_zarr_util.py`, `ome_tiff_helper.py`, `ome_zarr_helper.py`, `ome_helper.py`, `ome_ngff_helper.py` | 124/91/76/58/14 | None | Tier 2 |
| `Pipeline.py` | 121 | Indirect via `test_run.py` | Tier 2 |
| `ui/create_widgets.py` | 111 | `test_create_widgets_integration.py` (partial) | Tier 3 |
| `ui/bilayers_util.py` | 98 | `test_bilayers_util.py` (partial) | Tier 3 |
| `image/flatfield.py`, `color_conversion.py`, `reg_util.py` | 82/15/34 | None | Tier 1 (quick win) |
| `image/source_helper.py` | 68 | None | Tier 2 |
| `fusion_methods/*` | 42/17/5 | None | Tier 1 (quick win) |
| `file/rocrate_utils.py`, `rembi_extension.py`, `project_yaml.py`, `zarr_extension.py`, `resources.py` | 37/31/26/23/19 | None | Tier 2 |
| `Timer.py`, `logging.py`, `ui/_utils.py`, `ui/ParamWidget.py` | 27/27/31/24 | None | Tier 1 (quick win) |
| `_widget.py` | 39 | `test_widget.py` (partial) | Tier 3 |
| `registration_methods/*Features.py`, `CPD.py` | tested | good | - |

## Execution order (batches)

1. **Pure-logic quick wins** (no Qt/GUI, high ROI):
   `Timer.py`, `logging.py`, `fusion_methods/FusionMethodAdditive.py`,
   `FusionMethodExclusive.py`, `image/color_conversion.py`,
   `image/reg_util.py`, `image/flatfield.py`.

2. **Critical math/metrics**:
   `metrics.py` (all metric functions), remaining untested branches in
   `image/util.py` (normalization, foreground map, sim-shape/overlap
   helpers).

3. **Top-level `util.py`**:
   string/path/dict helpers, transform math, dimension utilities not already
   exercised transitively.

4. **`MVSRegistration.py`** (highest complexity):
   isolate pure/deterministic units (parameter resolution, transform key
   handling, pairwise selection logic) with mocked `sims`/`msi_utils`,
   avoiding full pipeline runs.

5. **IO helpers** (`ome_*`, `*DaskSource.py`, `file/*`):
   use `tmp_path` + tiny synthetic Zarr/TIFF fixtures; mock heavy external
   I/O where infeasible.

6. **UI/Interface edge cases**:
   expand `Interface.py`/`_widget.py`/`create_widgets.py` error paths using
   the existing `make_napari_viewer` fixture pattern already in
   `test_napari_interface_registration.py`.

## Tox / coverage config hardening

- `tox.ini` currently uses `--cov=muvis_align` (matches the installed package
  name from `src/muvis_align`), so that's correct as-is.
- Add `--cov-report=xml` for CI artifacts, alongside the existing
  `--cov-report` (term) for visibility.
- Add `--cov-branch` for branch coverage.
- Add a `--cov-fail-under=<N>` floor once a real CI baseline is known, then
  ratchet the threshold upward incrementally per PR.

## Next step

Implement Batch 1 (pure-logic quick-win tests) first, then proceed batch by
batch, validating each with a targeted `pytest` run before moving on.
