# PenguinDiffusion.jl

`PenguinDiffusion.jl` solves cut-cell diffusion problems on Cartesian grids, with:

- monophasic or diphasic models,
- fixed or moving embedded interfaces,
- steady or unsteady (`θ`) time integration.

It sits in the PenguinxCutCell stack:

- `CartesianGeometry.jl` for geometric moments,
- `CartesianOperators.jl` for cut-cell operators,
- `PenguinBCs.jl` for border/interface condition types,
- `PenguinSolverCore.jl` for linear solves.

## Start Here

- [Theory](diffusion.md): PDEs, BCs, interface conditions, callback conventions.
- [Algorithms](algorithms.md): block systems, θ assembly, regularization, moving slabs.
- [API](api.md): public constructors/functions and state layout conventions.
- [Examples](examples.md): curated scripts + verification map.

## Feature Matrix (Summary)

| Area | Support |
|---|---|
| Mono steady/unsteady | Yes |
| Diph steady/unsteady | Yes |
| Moving mono/diph (space-time slab) | Yes |
| Time schemes | `:BE`, `:CN`, numeric `θ ∈ [0,1]` |
| Outer BCs | Dirichlet, Neumann, Robin, Periodic |
| Interface BCs | mono Robin, diph `ScalarJump` / `RobinJump` / `FluxJump` |
| Coeff/source callbacks | constant, space-dependent, time-dependent |

## Quick Start Snippets

### 1) Mono steady

```julia
using CartesianGeometry: geometric_moments, nan
using CartesianOperators
using PenguinBCs
using PenguinDiffusion

grid = (range(0.0, 1.0; length=65),)
moms = geometric_moments((x) -> -1.0, grid, Float64, nan; method=:vofijul)
cap = assembled_capacity(moms; bc=0.0)
bc = BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0))
ops = DiffusionOps(cap; periodic=periodic_flags(bc, 1))
model = DiffusionModelMono(cap, ops, 1.0; source=0.0, bc_border=bc)
sol = solve_steady!(model)
```

### 2) Mono unsteady

```julia
u0 = zeros(cap.ntotal)  # ω-only initial state accepted
sol = solve_unsteady!(model, u0, (0.0, 0.1); dt=0.01, scheme=:CN)
```

### 3) Diph steady

```julia
ic = InterfaceConditions(
    scalar=ScalarJump(1.0, 1.0, 0.0),
    flux=FluxJump(1.0, 1.0, 0.0),
)
model2 = DiffusionModelDiph(cap, ops, 1.0, 1.0; source=(0.0, 0.0), bc_border=bc, ic=ic)
sol2 = solve_steady!(model2)
```

### 4) Moving mono

```julia
using CartesianGrids

grid_m = CartesianGrid((0.0,), (1.0,), (65,))
body(x, t) = x - (0.5 + 0.05 * sin(2pi * t))
mmodel = MovingDiffusionModelMono(grid_m, body, 1.0; source=0.0, bc_border=bc)
solm = solve_unsteady_moving!(mmodel, zeros(prod(grid_m.n)), (0.0, 0.1); dt=0.01, scheme=:BE)
```

## Local Build

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```
