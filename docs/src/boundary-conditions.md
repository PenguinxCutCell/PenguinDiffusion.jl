# Boundary Conditions

## Box Boundary Conditions

Outer-box boundaries are passed through `CartesianOperators.BoxBC`.

Common choices include:

- `Dirichlet(value)`
- `Neumann(value)`
- `Periodic(...)` where supported by the underlying operator

If omitted, default box conditions come from `BoxBC(Val(N), T)`.

## Embedded Interface Conditions

Monophasic problems use:

- `RobinConstraint(a, b, g)`

Two-phase problems use:

- `FluxJumpConstraint(b1, b2, g_flux)`
- `ScalarJumpConstraint(alpha1, alpha2, g_scalar)`

All three fields can be scalar, full-grid vectors, or runtime-updated via updaters.
