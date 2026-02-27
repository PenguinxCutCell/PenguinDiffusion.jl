# Validation

Suggested validation checks for this package:

- Manufactured-solution convergence in steady full-domain and cut-domain settings.
- Robin interface consistency checks against analytical radial solutions.
- Two-phase interface continuity checks (scalar jump and flux jump).
- Unsteady regression checks against known final-time fields.
- Allocation regressions for repeated RHS calls.

The scripts in `examples/` can be adapted into reproducible convergence and regression tests.
