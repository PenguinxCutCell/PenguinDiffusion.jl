# Equations

`PenguinDiffusion.jl` solves diffusion operators on cut-cell geometries.

## Monophasic Form

For active phase `omega`, the semidiscrete system is assembled as:

```math
M \dot u = L_\omega(u; \kappa, \text{BC}, \text{interface}) + V \odot f
```

with steady form:

```math
0 = L_\omega(u; \kappa, \text{BC}, \text{interface}) + V \odot f
```

where `M` is the reduced diagonal mass matrix from active-cell volumes.

## Two-Phase Form

For phases `omega_1` and `omega_2`, the package builds a coupled reduced system with:

- per-phase diffusion coefficients (`kappa1`, `kappa2`),
- per-phase box BCs,
- interface scalar-jump and flux-jump constraints.

## Interface Constraints

The interface is represented through CartesianOperators constraints:

- `RobinConstraint` for monophasic interface closure,
- `FluxJumpConstraint` and `ScalarJumpConstraint` for two-phase coupling.
