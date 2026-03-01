# Validation

Suggested validation checks for this package:

- Manufactured-solution convergence in steady full-domain and cut-domain settings.
- Robin interface consistency checks against analytical radial solutions.
- Two-phase interface continuity checks (scalar jump and flux jump).
- Unsteady regression checks against known final-time fields.
- Allocation regressions for repeated RHS calls.
- Assembled block-loop checks:
  - one-step equivalence vs direct block linear solve,
  - short-final-step handling,
  - scheduled updater firing at exact times.
- Full-vector volume-weighted error norms (not active-index-only norms), so
  outside regions contribute zero naturally via `V=0`.

The scripts in `examples/` can be adapted into reproducible convergence and regression tests.
