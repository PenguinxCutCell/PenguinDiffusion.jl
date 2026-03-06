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

- `examples/steady_1d_diph_robinjump.jl`:
  1D steady diphasic Robin-jump + flux-continuity validation using
  `InterfaceConditions(scalar=RobinJump(...), flux=FluxJump(...))`,
  with analytical profile comparison, interface residual checks, and optional plot output.
- `examples/2D/Diffusion/Poisson_robin.jl`:
  2D Poisson inside a disk with embedded Robin condition.
- `examples/2D/Diffusion/Heat_robin.jl`:
  2D unsteady heat diffusion inside a disk with embedded Robin condition (uses `solve_unsteady!`).
- `examples/2D/Diffusion/MovingHeat_robin.jl`:
  2D moving-geometry manufactured transient heat case inside an oscillating disk with embedded Robin condition (uses `solve_unsteady_moving!`).
- `examples/2D/Diffusion/MovingHeat_robin_real.jl`:
  2D moving-geometry manufactured transient heat case with a true embedded Robin condition (`alpha != 0`, `beta != 0`) to exercise the full interface assembly path.
- `examples/1D/Diffusion/Poisson_nobody_neumann_mms.jl`:
  1D manufactured Poisson, no embedded body, Neumann box boundaries.
- `examples/1D/Diffusion/Poisson_2ph.jl`:
  1D diphasic Poisson with embedded interface continuity conditions.
- `examples/2D/Diffusion/Heat_2ph.jl`:
  2D diphasic unsteady heat manufactured solution (uses `solve_unsteady!`).
- `examples/2D/Diffusion/Poisson_2ph_robinjump_mms.jl`:
  2D diphasic steady manufactured Robin-jump case with small x-refinement
  sweep and interface residual checks for `RobinJump + FluxJump`.
- `examples/2D/Diffusion/Heat_2ph_disk_transfer_metrics.jl`:
  2D diphasic unsteady disk-transfer benchmark with post-processing via
  `compute_interface_exchange_metrics` (generic exchange coefficient and
  dimensionless transfer index), plus comparison to the legacy manual
  interface-flux assembly path.
- `examples/2D/Diffusion/MovingHeat_2ph.jl`:
  2D diphasic moving-geometry manufactured transient heat case (uses `solve_unsteady_moving!`).
- `examples/3D/Diffusion/Poisson_outside_sphere_embedded_dirichlet.jl`:
  3D manufactured Poisson outside a sphere, embedded Dirichlet interface and analytical Dirichlet on the outer box.
