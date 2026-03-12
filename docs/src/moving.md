# Moving Geometry (Space-Time Slabs)

Moving models:

- `MovingDiffusionModelMono`
- `MovingDiffusionModelDiph`

use per-step space-time slab assembly over `[t^n, t^{n+1}]`.

## Geometry Path

For each step:

1. evaluate phase level-set(s) at `t^n` and `t^{n+1}` for `V^n`, `V^{n+1}`,
2. build a 2-time-node `SpaceTimeCartesianGrid`,
3. compute geometric moments in space-time,
4. reduce slab moments back to space operators/capacity.

## Temporal Treatment

For moving assemblies:

- slab/mass terms use `V^n`, `V^{n+1}` and `Ψ` masks,
- diffusivity-weighted operator blocks are sampled at `t^n + θΔt`,
- source data are end-state for `:BE`,
- source data are `θ`-blended between `t^n` and `t^{n+1}` for `θ < 1`,
- interface scalar/flux coefficients and values follow the same rule.

## Interface Sampling Convention

Spatial interface sampling remains at slab `C_γ`.

When blending in time (`θ < 1`), both `t^n` and `t^{n+1}` evaluations are taken at that same slab `C_γ` sample set.

## API

```julia
sol = solve_unsteady_moving!(model, u0, (t0, tf); dt=Δt, scheme=:CN)
```

`solve_unsteady!(model::MovingDiffusionModel..., ...)` dispatches to the same moving solver.
