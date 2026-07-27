Copilot GPT 5.3 - Codex

here’s a focused plan to raise coverage via tox, and we can review/adjust before implementation.
* Baseline current coverage per module and per tox env
* Prioritize high-impact low-coverage files (core logic before UI-heavy paths)
* Add deterministic unit tests around registration/image utility edge cases
* Add lightweight integration tests for interface flows with mocks/fixtures
* Tighten tox/pytest coverage config and enforce incremental thresholds
* Add CI-friendly reporting so coverage regressions fail fast

Proposed Plan
Baseline & gap map
Run coverage in tox and generate term + XML + HTML reports.
Build a “top uncovered lines/functions” list by file (especially src/muvis_align/registration_methods, src/muvis_align/image, src/muvis_align/ui).
Separate pure logic from UI/IO code so we target quick wins first.
Prioritization strategy
Tier 1: business-critical transform/registration paths (CPD, CvFeatures, SkFeatures, image/util math + coordinate conversions).
Tier 2: interface orchestration (Interface.update_registered, registration process flow, widget wiring behavior via mocks).
Tier 3: hard-to-test external/visual code (napari rendering details), covered with smoke/integration style tests.
Test design upgrades
Reuse shared synthetic data builders (2D for CV/SK, 2D+3D for CPD).
Add edge cases: empty points, singleton dims, dtype conversions, mismatch dims, invalid/partial matches.
Add regression tests for known bugs (axis swaps, transform copy behavior, fallback/no-match paths).
Tox and coverage config hardening
Ensure tox.ini/pytest-cov uses:
branch coverage
stable --cov=src/muvis_align
explicit omit list (generated/docs/site only)
Add env split if needed:
fast unit env
integration env (napari/qt-dependent)
Gate with incremental thresholds (e.g., start current+2%, then ratchet up).
Execution cadence
Implement in small batches (5–10 tests per batch), run targeted tox envs each batch.
After each batch, update the coverage delta summary and next target list.
Stop when we hit agreed target (e.g., overall +8–12 points or module-level minimums).
Recommended initial target (practical)
Raise overall coverage to at least a stable floor (example: >=75%) and critical modules to >=85%.
Then ratchet threshold upward every PR by a small amount.
If you want, next step I’ll do is produce a concrete file-by-file coverage attack plan (top 10 functions to test first, estimated gain per file).
