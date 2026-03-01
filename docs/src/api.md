# API Reference

This page lists the package public API.

## Problem and System Types

- `DiffusionProblem`
- `DiffusionSystem`
- `TwoPhaseDiffusionProblem`
- `TwoPhaseDiffusionSystem`

## Build and State

- `build_system`
- `build_matrixfree_system`
- `enable_matrixfree_unsteady!`
- `full_state`

## Monophasic Updaters

- `RobinGUpdater`
- `RobinABUpdater`
- `BoxDirichletUpdater`
- `KappaUpdater`
- `SourceUpdater`

## Two-Phase Updaters

- `FluxJumpGUpdater`
- `FluxJumpBUpdater`
- `ScalarJumpGUpdater`
- `ScalarJumpAlphaUpdater`
- `Kappa1Updater`
- `Kappa2Updater`
- `BoxDirichletUpdater1`
- `BoxDirichletUpdater2`
- `Source1Updater`
- `Source2Updater`

## Steady Solve Helpers

- `steady_linear_problem`
- `steady_solve`

## Assembled Block Unsteady Helpers

- `unsteady_block_matrix`
- `unsteady_block_solve`
- `diphasic_unsteady_block_matrix`
- `diphasic_unsteady_block_solve`
