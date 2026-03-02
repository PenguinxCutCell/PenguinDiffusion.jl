# PenguinDiffusion Examples

This folder contains compact, reproducible examples ported from `../Penguin.jl/examples/*/Diffusion/` to the current `PenguinDiffusion.jl` API.

Each example includes:

- an analytical reference solution,
- a numerical solve with `PenguinDiffusion.jl`,
- a volume-weighted error norm check (`L2` over active physical cells),
- and an `@assert` on the expected error level.

Run an example from the repository root with:

```bash
julia --project=. examples/2D/Diffusion/Poisson_robin.jl
```

Available examples:

- `examples/2D/Diffusion/Poisson_robin.jl`:
  2D Poisson inside a disk with embedded Robin condition.
- `examples/2D/Diffusion/Heat_robin.jl`:
  2D unsteady heat diffusion inside a disk with embedded Robin condition (uses `solve_unsteady!`).
- `examples/1D/Diffusion/Poisson_nobody_neumann_mms.jl`:
  1D manufactured Poisson, no embedded body, Neumann box boundaries.
- `examples/1D/Diffusion/Poisson_2ph.jl`:
  1D diphasic Poisson with embedded interface continuity conditions.
- `examples/2D/Diffusion/Heat_2ph.jl`:
  2D diphasic unsteady heat manufactured solution (uses `solve_unsteady!`).
- `examples/3D/Diffusion/Poisson_outside_sphere_embedded_dirichlet.jl`:
  3D manufactured Poisson outside a sphere, embedded Dirichlet interface and analytical Dirichlet on the outer box.
