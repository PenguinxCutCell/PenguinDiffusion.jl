# Design Notes

## Architecture

`PenguinDiffusion.jl` is built around reduced active-cell state vectors and explicit reconstruction to full-grid caches for operator calls.

## Core Contracts

- `PenguinSolverCore.rhs!(du, sys, u, p, t)` for unsteady integration.
- `PenguinSolverCore.mass_matrix(sys)` for semidiscrete mass scaling.
- `PenguinSolverCore.rebuild!(sys, u, p, t)` for scheduled updater application.

## Two-Phase Layout

Two-phase systems keep separate reduced DOF maps per phase and couple them through interface constraints assembled by `CartesianOperators`.

## Matrix-Free Path

For large unsteady runs, matrix-free system construction avoids explicit sparse matrix assembly and applies operators through preallocated work buffers.
