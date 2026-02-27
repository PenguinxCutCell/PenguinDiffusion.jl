# PenguinDiffusion.jl

`PenguinDiffusion.jl` provides cut-cell diffusion solvers on reduced active-cell states.

Current capabilities include:

- Monophasic diffusion systems with embedded Robin interface constraints.
- Two-phase diffusion systems with scalar-jump and flux-jump constraints.
- Steady solves via `LinearSolve`.
- Unsteady solves via `PenguinSolverCore` + SciML ODE integration.
- Matrix-free unsteady RHS path for large problems.

## Documentation Map

- [Equations](equations.md): governing forms solved by this package.
- [Boundary Conditions](boundary-conditions.md): box BCs and interface constraints.
- [Numerics](numerics.md): reduced-state masking, operators, and runtime updates.
- [API Reference](api.md): exported constructors/functions.
- [Examples](examples.md): runnable scripts under `examples/`.
- [Validation](validation.md): recommended verification checks.
- [Design Notes](design.md): implementation-level architecture notes.
