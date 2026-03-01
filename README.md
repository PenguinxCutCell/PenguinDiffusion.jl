# PenguinDiffusion.jl

`PenguinDiffusion.jl` is a cut-cell diffusion package built on `CartesianGeometry.jl`, `CartesianOperators.jl`, and `PenguinSolverCore.jl`.

It supports monophasic and two-phase diffusion with reduced-state unknowns on active cells and embedded-interface constraints.

## Features

- Reduced state on active cells via `PenguinSolverCore.DofMap`.
- Monophasic diffusion with `RobinConstraint` on embedded interfaces.
- Two-phase diffusion with `FluxJumpConstraint` and `ScalarJumpConstraint` coupling.
- Steady solve helpers (`steady_linear_problem`, `steady_solve`) based on `LinearSolve.jl`.
- Unsteady integration through `PenguinSolverCore` + SciML (`sciml_odeproblem`, `rhs!`, `mass_matrix`).
- Matrix-free unsteady path (`build_matrixfree_system`, `enable_matrixfree_unsteady!`).
- Assembled block unsteady solvers:
  - Monophasic: `unsteady_block_matrix`, `unsteady_block_solve`
  - Diphasic: `diphasic_unsteady_block_matrix`, `diphasic_unsteady_block_solve`
- Runtime updater hooks for interface data, box Dirichlet values, diffusivity, and sources.

## Quickstart

```julia
using PenguinDiffusion
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore
using SciMLBase
using OrdinaryDiffEq

x = collect(range(0.0, 1.0; length=65))
y = collect(range(0.0, 1.0; length=65))
disk(xp, yp, _t=0.0) = sqrt((xp - 0.5)^2 + (yp - 0.5)^2) - 0.22
moments = geometric_moments(disk, (x, y), Float64, zero; method=:implicitintegration)

bc = BoxBC(Val(2), Float64)
Nd = length(moments.V)
interface = RobinConstraint(ones(Float64, Nd), zeros(Float64, Nd), zeros(Float64, Nd))
prob = DiffusionProblem(1.0, bc, interface, 0.0)
sys = build_system(moments, prob)

u0 = zeros(Float64, length(sys.dof_omega.indices))
odeprob = PenguinSolverCore.sciml_odeproblem(sys, u0, (0.0, 0.05); p=nothing)
sol = SciMLBase.solve(odeprob, Rosenbrock23(autodiff=false); reltol=1e-7, abstol=1e-7)

u_full, gamma_full = full_state(sys, sol.u[end])
println("n_omega=", length(sys.dof_omega.indices), ", n_gamma=", length(sys.dof_gamma.indices))
```

## Problem Forms

For monophasic systems, the semidiscrete form is represented as:

```math
M \dot u = L_\omega(u; \kappa, \text{BC}, \Gamma) + V \odot f
```

with steady form:

```math
0 = L_\omega(u; \kappa, \text{BC}, \Gamma) + V \odot f
```

Two-phase systems build a coupled reduced system over `omega_1` and `omega_2`, with interface scalar/flux jump constraints.

## Supported Boundary and Interface Data

- Outer-box BCs through `CartesianOperators.BoxBC` (`Dirichlet`, `Neumann`, `Periodic` where supported).
- Monophasic interface closure via `RobinConstraint(a, b, g)`.
- Two-phase interface coupling via `FluxJumpConstraint(b1, b2, g_flux)` and `ScalarJumpConstraint(alpha1, alpha2, g_scalar)`.

## Public API Highlights

Core types:

- `DiffusionProblem`, `DiffusionSystem`
- `TwoPhaseDiffusionProblem`, `TwoPhaseDiffusionSystem`

Build/state:

- `build_system`, `build_matrixfree_system`
- `enable_matrixfree_unsteady!`, `full_state`

Updaters:

- Monophasic: `RobinGUpdater`, `RobinABUpdater`, `BoxDirichletUpdater`, `KappaUpdater`, `SourceUpdater`
- Two-phase: `FluxJumpGUpdater`, `FluxJumpBUpdater`, `ScalarJumpGUpdater`, `ScalarJumpAlphaUpdater`, `Kappa1Updater`, `Kappa2Updater`, `BoxDirichletUpdater1`, `BoxDirichletUpdater2`, `Source1Updater`, `Source2Updater`

Steady solve helpers:

- `steady_linear_problem`, `steady_solve`

Assembled block unsteady helpers:

- `unsteady_block_matrix`, `unsteady_block_solve`
- `diphasic_unsteady_block_matrix`, `diphasic_unsteady_block_solve`

## Validation Coverage (tests)

Current tests cover:

- Reduced/full state mapping contracts.
- Runtime update and rebuild behavior.
- Monophasic steady and unsteady manufactured checks.
- Two-phase assembly and manufactured continuity checks.
- Matrix-free unsteady behavior.
- SciML integration and steady solver regression checks.

See `test/runtests.jl` for the executed test set.

## Examples

Run from repository root:

```bash
julia --project examples/<file>.jl
```

Included scripts:

- `examples/robin_disk_steady.jl`
- `examples/robin_disk_unsteady.jl`
- `examples/robin_disk_unsteady_matrixfree.jl`
- `examples/full_domain_manufactured_steady.jl`
- `examples/full_domain_manufactured_unsteady.jl`
- `examples/outside_circle_manufactured_steady.jl`
- `examples/one_d_continuous_body_halfspace_boxbc.jl`
- `examples/twophase_square_circle_continuity_steady.jl`
- `examples/twophase_square_circle_continuity_unsteady.jl`
- `examples/heat_2ph_2d_benchphaseflow.jl`

## Documentation

A local Documenter setup is included in `docs/`:

```bash
julia --project=docs docs/make.jl
```
