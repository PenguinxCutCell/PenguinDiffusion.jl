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

## Runtime Updates

The package exposes updater types for time-dependent coefficients/BCs/source terms, including:

- `RobinGUpdater`, `RobinABUpdater`
- `BoxDirichletUpdater`, `KappaUpdater`, `SourceUpdater`
- two-phase updater variants (`...1Updater`, `...2Updater`, jump updaters)

These updaters integrate with `PenguinSolverCore.rebuild!` scheduling.
