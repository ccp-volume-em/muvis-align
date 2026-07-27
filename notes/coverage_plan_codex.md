Codex (GPT 5.6-Sol)

I’ll inspect the referenced test, the tox/coverage configuration, and the corresponding implementation. I’ll return a proposed coverage plan only—no file changes yet.

The current file mixes configuration-schema checks, light state assertions, and heavily mocked workflow tests. I’m now checking what tox actually runs and which `Interface` branches remain unexercised so the plan targets measurable gaps rather than simply adding test count.

One important issue is already visible: several tests patch the lower-level methods but invoke only the confirmation wrappers, so they don’t cover the serialization and result-handling code their comments claim to exercise. I’ll run the file once with branch coverage to establish the baseline and identify any skips/failures across the local tox-equivalent environment.

The cached tox environment is incomplete, so a fresh local baseline cannot run without rebuilding dependencies. I won’t mutate the environment during plan review; instead I’m using the checked-in `coverage.xml` plus source inspection, and I’ll include a clean tox baseline as the first implementation step.

I reviewed the test, `Interface.py`, tox configuration, and existing coverage artifact. No files were changed.

## Findings

- Existing `coverage.xml` reports only **16.36% overall line coverage** and **20.92% for `Interface.py`**, although it appears stale because it references an older installed copy under `.tox`.
- Tox collects line coverage but does not:
  - measure branches,
  - display missing lines,
  - enforce a minimum threshold.
- The two project YAML files parameterize nearly every UI test, duplicating execution without exercising different UI branches.
- Several workflow tests invoke wrappers such as `pair_registration()` while mocking `run_pair_registration()`. Consequently, result processing in `run_pair_registration()` is not covered.
- Assertions such as “did not raise” or “mock was called” are too weak to verify saved mappings, converted bounding boxes, selected transforms, tile-size parsing, or state transitions.
- `test_modify_pair_registration_with_bbox` imports `PyQt5` directly and may skip even though the project uses `qtpy`.
- Confirmation paths currently focus on “Yes”; warning, “No,” rerun, missing-prerequisite, and reversed-pair branches remain uncovered.
- The cached tox environments are incomplete, so the current baseline needs to be regenerated before choosing a realistic coverage gate.

## Proposed plan

1. Establish a trustworthy baseline

   - Rebuild one tox environment and run the full tox pytest command.
   - Capture overall, `Interface.py`, line, and branch coverage.
   - Record failures and platform-specific skips separately.

2. Refactor the referenced test module

   - Create a compact `Interface` fixture with reusable viewer, widget, registration, and context-manager mocks.
   - Parameterize only configuration-validation tests over both YAML files.
   - Run behavioral UI tests once unless configuration values materially change behavior.
   - Replace nested `with patch(...)` blocks with fixtures or `ExitStack`.
   - Remove unused imports and the direct `PyQt5` dependency.

3. Test workflow methods directly

   Add focused tests for:

   - `run_pair_registration()`:
     - registration arguments and metrics,
     - quality extraction,
     - bounding boxes with and without a `t` dimension,
     - exact values passed to `save_pair_mappings()`.
   - `run_global_registration()`:
     - registration arguments,
     - mappings, CSV mappings, and metrics persistence.
   - `pair_registration()`:
     - already-global warning,
     - user cancellation,
     - initial and repeated pair registration.
   - `registration_process()`:
     - no prior pair registration,
     - existing pair registration,
     - existing global registration,
     - cancellation,
     - transform copying, tab enabling, and selected view transform.
   - `fusion_process()`:
     - comma-separated, scalar-string, and already numeric tile sizes,
     - cancellation and rerun prompts,
     - output-name derivation,
     - fused/view state updates.

4. Cover pair-editing behavior

   Add cases for:

   - entering pair-edit mode,
   - reversed edge lookup,
   - missing-pair warning,
   - saving and discarding edits,
   - transforms with and without `t`,
   - exact edge attributes and serialized bounding boxes,
   - widget-state restoration.

5. Cover compact high-value helpers

   Test `populate_metrics_table()`, `update_pair_metrics()`, `calc_mod_pair_transform()`, `preview_registration()`, and `preview_fusion()` with small deterministic arrays and mocks. These methods contain substantial currently uncovered branching without requiring real registration or disk-heavy fusion.

6. Strengthen tox coverage reporting

   - Add `--cov-branch`.
   - Add `--cov-report=term-missing` and retain XML output for CI.
   - Set `--cov-fail-under` only after the clean baseline is known, initially just below the achieved result so coverage cannot regress.
   - Keep the same settings across Python 3.11–3.13 and all platform environments.

7. Verify incrementally

   - Run the referenced test file first.
   - Run the complete tox command for one local Python version.
   - Run or rely on CI for the entire Python/platform matrix.
   - Compare both overall coverage and `Interface.py` coverage against the baseline.

I recommend implementing steps 1–5 first, measuring the gain, and then setting the tox threshold from the verified result rather than choosing an arbitrary percentage.
