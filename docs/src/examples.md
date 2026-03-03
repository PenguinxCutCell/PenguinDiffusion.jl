**Examples**

The repository includes standalone example scripts under `examples/` with analytical references and volume-weighted error checks.

Run from repository root:

```bash
julia --project=. examples/2D/Diffusion/Poisson_robin.jl
```

Available diffusion examples

- `examples/2D/Diffusion/Poisson_robin.jl`
  - 2D steady Poisson inside a disk.
  - Embedded interface condition: `Robin(alpha, beta, g)`.
  - Analytical reference: radial quadratic profile.
  - Check: volume-weighted `L2` error over active physical cells.

- `examples/2D/Diffusion/Heat_robin.jl`
  - 2D unsteady heat diffusion inside a disk.
  - Embedded interface condition: Robin exchange to ambient temperature.
  - Analytical reference: radial Bessel-series solution at final time.
  - Check: volume-weighted `L2` error at final time.
  - Uses `solve_unsteady!(...; scheme=:BE)`.

- `examples/2D/Diffusion/MovingHeat_robin.jl`
  - 2D unsteady heat diffusion on a moving domain (oscillating disk).
  - Embedded interface condition: `Robin(1, 0, g)` (Dirichlet-equivalent on Γ).
  - Analytical reference: manufactured smooth transient solution `u(x,y,t)=exp(λt)sin(πx)sin(πy)`.
  - Check: volume-weighted `L2` error at final time.
  - Uses `solve_unsteady_moving!(...; scheme=:BE)`.

- `examples/1D/Diffusion/Poisson_nobody_neumann_mms.jl`
  - 1D manufactured steady Poisson with no embedded body (`body = -1`).
  - Outer boundary conditions: homogeneous Neumann on both sides.
  - Analytical reference: `u(x)=cos(pi*x)`.
  - Check: volume-weighted `L2` error, with one dof pinned as gauge for the Neumann nullspace.

- `examples/3D/Diffusion/Poisson_outside_sphere_embedded_dirichlet.jl`
  - 3D manufactured steady Poisson outside a sphere (cut-cell geometry).
  - Embedded interface condition: Dirichlet implemented as `Robin(1, 0, g)`.
  - Outer box boundary conditions: Dirichlet from the same analytical solution.
  - Check: volume-weighted `L2` error and active-volume consistency.

- `examples/1D/Diffusion/Poisson_2ph.jl`
  - 1D diphasic steady Poisson with interface continuity constraints.
  - Analytical reference: trivial manufactured solution `u1=u2=0`.
  - Check: phase-wise volume-weighted `L2` errors.

- `examples/2D/Diffusion/Heat_2ph.jl`
  - 2D diphasic unsteady heat manufactured solution.
  - Analytical references:
    - `u1=exp(-2*pi^2*t)*sin(pi*x)sin(pi*y)`
    - `u2=exp(-8*pi^2*t)*sin(2*pi*x)sin(2*pi*y)`
  - Check: phase-wise volume-weighted `L2` errors at final time.
  - Uses `solve_unsteady!(...; scheme=:CN)` and reports operator reuse.

- `examples/2D/Diffusion/MovingHeat_2ph.jl`
  - 2D diphasic unsteady heat on a moving domain (oscillating interface).
  - Interface conditions: zero scalar jump and zero flux jump.
  - Analytical reference: shared manufactured field `u1=u2=exp(-2*pi^2*t)sin(pi*x)sin(pi*y)`.
  - Check: phase-wise volume-weighted `L2` errors at final time.
  - Uses `solve_unsteady_moving!(...; scheme=:BE)`.

Regression tests

For the full package validation suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
