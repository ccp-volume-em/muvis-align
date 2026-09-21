# Working conventions for this repo

Distilled from recurring feedback across sessions. Follow these when making changes here.

## Comments
- Code comments: 1-2 lines max, even for non-obvious WHY reasoning. No multi-line rationale blocks.
- Never narrate what changed, reference the current fix/task, or mention specific callers/sessions in a comment - that belongs in the commit message or chat response, not the code.
- Comments explain WHY (a hidden constraint, a subtle invariant), never WHAT (identifiers already say that).

## Testing
- For a small/localized fix, run only the targeted test file(s) for the area touched - not the full suite (10+ min).
- Reserve full-suite runs for larger/riskier changes, or once before pushing a batch of combined fixes.
- Don't create a new permanent test file per change - add tests to the existing file matching the module under test (e.g. `tests/test_utils.py` for `util.py`). Only give a feature its own test file when it's substantial/self-contained enough to warrant one.

## Git
- Small, focused commits with a "why" in the message, not a changelog of "what".
