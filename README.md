# PenguinDiffusion.jl

[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PenguinxCutCell.github.io/PenguinDiffusion.jl/dev)
![CI](https://github.com/PenguinxCutCell/PenguinDiffusion.jl/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/PenguinxCutCell/PenguinDiffusion.jl/branch/main/graph/badge.svg)


`PenguinDiffusion.jl` is a cut-cell diffusion package built on `CartesianGeometry.jl`, `CartesianOperators.jl`, and `PenguinSolverCore.jl`.

## Feature Status

Last validated on **2026-03-11** with `julia --project=. test/runtests.jl` and example smoke runs.

| Area | Item | Status | Notes |
|---|---|---|---|
| Models | Monophasic steady | Implemented | `DiffusionModelMono` + `assemble_steady_mono!` |
| Models | Monophasic unsteady | Implemented | Theta-form assembly path via `assemble_unsteady_mono!` |
| Models | Monophasic unsteady moving geometry (space-time slab) | Implemented | `MovingDiffusionModelMono` + `assemble_unsteady_mono_moving!` + `solve_unsteady_moving!` |
| Models | Diphasic steady | Implemented | `DiffusionModelDiph` + `assemble_steady_diph!` |
| Models | Diphasic unsteady | Implemented | Theta-form assembly path + inactive `ω/γ` regularization for cut-cell robustness |
| Models | Diphasic unsteady moving geometry (space-time slab) | Implemented | `MovingDiffusionModelDiph` + `assemble_unsteady_diph_moving!` + `solve_unsteady_moving!` |
| Outer BCs | Dirichlet | Implemented | Face-based contribution in matrix + RHS |
| Outer BCs | Neumann | Implemented | Default behavior is homogeneous Neumann (`0`) |
| Outer BCs | Periodic | Implemented | Paired periodic sides supported |
| Outer BCs | Robin (outer box) | Implemented | Face-based contribution in matrix + RHS |
| Time scheme | Backward Euler (`θ=1`) | Implemented | Covered by regression/order tests |
| Time scheme | Crank–Nicolson (`θ=1/2`) | Implemented | Theta assembly path validated with dedicated temporal-order test |
| Coefficients | Constant diffusion coefficient | Implemented | Scalar coefficient support |
| Coefficients | Space-variable coefficient | Implemented | Face-based assembly (`:harmonic` default, `:arithmetic`, `:face`) |
| Coefficients | Time-variable coefficient | Implemented | `(x..., t)` coefficient support with face-based assembly |
| Embedded interface | Mono interface coupling | Implemented | Monophasic embedded interface uses `Robin(α, β, g)` |
| Embedded interface | Diph interface coupling | Implemented | 4-block diphasic interface assembly |
| Embedded interface | No-interface reduction behavior | Implemented | Regression-tested (`body = -1`) |
| Embedded interface | Cut-cell consistency checks | Implemented | Dedicated 1D cut-interface test |
| Operator/geometry | Halo/node-lattice invariants | Implemented | Tested for buffer/matrix halo convention |
| Operator/geometry | Full-domain `H≈0`, `Γ≈0` sanity | Implemented | Tested on no-interface geometry |
| Solver efficiency | Full identity regularization | Implemented | Most robust path in current code |
| Solver efficiency | Active-DOF pruning | Partial | Works and can reduce allocations; not yet integrated as default solve path |
| Solver efficiency | Schur elimination of interface unknowns | Partial | Promising in CPU for some cases, but currently allocation-heavy and not integrated |
| Diagnostics | Interface exchange metrics | Implemented | `compute_interface_exchange_metrics` + unit tests + `examples/2D/Diffusion/Heat_2ph_disk_transfer_metrics.jl` |
| Examples | 1D/2D/3D diffusion scripts | Implemented | Smoke-tested on 2026-03-11; paths listed in `examples/README.md` |

### Practical Solver Note

- With current identity regularization, old `remove_zero_rows_cols!` style trimming usually does not reduce matrix size.
- If you need better memory efficiency on cut-heavy runs, prefer active-DOF pruning (mask-based reduction) over legacy zero-row trimming.
- Schur elimination can be competitive in CPU time when `γ << ω`, but needs a dedicated optimized implementation to control allocations.

## Diphasic Interface Usage

```julia
ic = InterfaceConditions(
    scalar = RobinJump(α, β, gγ),    # scalar-like row
    flux   = FluxJump(1.0, 1.0, 0.0) # flux-like row (continuity)
)

model = DiffusionModelDiph(cap1, ops1, D1, s1, cap2, ops2, D2, s2;
                           bc_border=bc,
                           ic=ic)
```

Notes:

- `γ1` and `γ2` are separate interface unknown blocks.
- With `scalar = RobinJump(...)`, `uγ1` and `uγ2` are generally not equal.
- Flux continuity is enforced by the second interface row (`flux = FluxJump(...)`).
