# Examples

Run examples from repository root:

```bash
julia --project examples/<file>.jl
```

Available scripts include:

- `robin_disk_steady.jl`
- `robin_disk_unsteady.jl`
- `robin_disk_unsteady_matrixfree.jl`
- `full_domain_manufactured_steady.jl`
- `full_domain_manufactured_unsteady.jl`
- `outside_circle_manufactured_steady.jl`
- `one_d_continuous_body_halfspace_boxbc.jl`
- `twophase_square_circle_continuity_steady.jl`
- `twophase_square_circle_continuity_unsteady.jl`
- `heat_2ph_2d_benchphaseflow.jl`

Notes:

- `robin_disk_unsteady_blockloop.jl` demonstrates the monophasic assembled block loop.
- `heat_2ph_2d_benchphaseflow.jl` demonstrates the diphasic assembled block loop and
  full-vector volume-weighted error norms.
