# Examples

Instantiate the examples environment:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

Run a script:

```bash
julia --project=examples examples/2D/Diffusion/Poisson_robin.jl
```

## 1D

- `examples/1D/Diffusion/Poisson_nobody_neumann_mms.jl`
  - Physical case: full-domain manufactured Poisson with Neumann boundaries.
  - Key API call: `solve_steady!(DiffusionModelMono(...))`
  - Check: weighted `L2` error and Neumann-gauge consistency.

- `examples/1D/Diffusion/Poisson_2ph.jl`
  - Physical case: diphasic steady Poisson with interface continuity.
  - Key API call: `solve_steady!(DiffusionModelDiph(...))`
  - Check: phase-wise weighted error.

- `examples/1D/Diffusion/steady_1d_diph_robinjump.jl`
  - Physical case: diphasic Robin jump + flux continuity benchmark.
  - Key API call: `InterfaceConditions(scalar=RobinJump(...), flux=FluxJump(...))`
  - Check: piecewise analytical profile + interface residuals.

## 2D Fixed Interface

- `examples/2D/Diffusion/Poisson_robin.jl`
  - Physical case: steady Poisson with embedded Robin interface.
  - Key API call: `solve_steady!`
  - Check: weighted `L2` convergence against manufactured solution.

- `examples/2D/Diffusion/Heat_robin.jl`
  - Physical case: unsteady mono diffusion with embedded Robin interface.
  - Key API call: `solve_unsteady!(...; scheme=:BE)`
  - Check: final-time weighted error.

- `examples/2D/Diffusion/Heat_2ph.jl`
  - Physical case: unsteady diphasic manufactured heat modes.
  - Key API call: `solve_unsteady!(...; scheme=:CN)`
  - Check: phase-wise weighted errors and stable operator reuse path.

- `examples/2D/Diffusion/Heat_2ph_disk_transfer_metrics.jl`
  - Physical case: diphasic transfer benchmark with diagnostics.
  - Key API call: `compute_interface_exchange_metrics`
  - Check: flux/transfer metrics against manual operator-based expressions.

## 2D Moving Interface

- `examples/2D/Diffusion/MovingHeat_robin.jl`
  - Physical case: moving mono geometry with manufactured transient field.
  - Key API call: `solve_unsteady_moving!`
  - Check: final-time weighted error and robustness under motion.

- `examples/2D/Diffusion/MovingHeat_robin_real.jl`
  - Physical case: moving mono geometry with nontrivial Robin interface (`α,β ≠ 0`).
  - Key API call: `MovingDiffusionModelMono(...; bc_interface=Robin(...))`
  - Check: stable interface coupling with real Robin terms.

- `examples/2D/Diffusion/MovingHeat_2ph.jl`
  - Physical case: moving diphasic manufactured case.
  - Key API call: `MovingDiffusionModelDiph` + `solve_unsteady_moving!`
  - Check: phase-wise final-time weighted errors.

## 3D

- `examples/3D/Diffusion/Poisson_dirichlet.jl`
  - Physical case: 3D manufactured Poisson with embedded interface.
  - Key API call: `solve_steady!`
  - Check: weighted `L2` error and active-volume sanity.

## Verification Map (Tests ↔ Examples)

- Moving mono/diph invariance tests ↔ moving examples (`MovingHeat_*`).
- Diphasic jump/Robin tests ↔ `steady_1d_diph_robinjump.jl`.
- Unsteady CN/BE regression tests ↔ `Heat_robin.jl`, `Heat_2ph.jl`.
- Transfer-metric unit tests ↔ `Heat_2ph_disk_transfer_metrics.jl`.
