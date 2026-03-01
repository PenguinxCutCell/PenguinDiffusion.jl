# Numerics

## Reduced State

Unknowns are stored only on active cells using `PenguinSolverCore.DofMap`.

The package maintains internal full-grid caches for:

- reconstruction of full fields,
- operator application,
- interface coupling terms.

## Build Paths

- `build_system(...)` assembles the standard diffusion system.
- `build_matrixfree_system(...)` builds the matrix-free unsteady path.
- `unsteady_block_*` and `diphasic_unsteady_block_*` provide assembled block
  fixed-step implicit time loops.

## Two-Phase Flux/Scalar Jump Conventions

For diphasic coupling:

- Scalar jump rows enforce `α1*T1 + α2*T2 = g_scalar` at the interface.
- Flux jump rows enforce `b1*q1 + b2*q2 = g_flux` with opposite phase normals.

This means physical flux continuity `D1*∂nT1 = D2*∂nT2` maps to:

- `FluxJumpConstraint(b1=D1, b2=-D2, g=0)`.

## Runtime Updates

The package exposes updater types for time-dependent coefficients/BCs/source terms, including:

- `RobinGUpdater`, `RobinABUpdater`
- `BoxDirichletUpdater`, `KappaUpdater`, `SourceUpdater`
- two-phase updater variants (`...1Updater`, `...2Updater`, jump updaters)

These updaters integrate with `PenguinSolverCore.rebuild!` scheduling.
