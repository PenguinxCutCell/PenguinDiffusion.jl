# Unsteady Solves (`θ`-Schemes)

## Accepted Scheme Values

- `scheme = :BE`  -> `θ = 1`
- `scheme = :CN`  -> `θ = 1/2`
- `scheme = θ::Real`, with `0 ≤ θ ≤ 1`

These apply to both mono and diphasic APIs.

## Fixed Geometry APIs

```julia
sol = solve_unsteady!(model, u0, (t0, tf); dt=Δt, scheme=:CN)
```

Return tuple:

- `times`
- `states`
- `system` (last solved `LinearSystem`)
- `reused_constant_operator` (fixed-geometry optimization flag)

## Initial State Shapes

- Mono:
  - reduced: `length(u0) == ntotal` (`ω` block only), or
  - full: system length (`ω` + `γ`).
- Diph:
  - reduced: `length(u0) == 2*ntotal` (`ω1` + `ω2`), or
  - full: system length (`ω1`,`γ1`,`ω2`,`γ2`).

For reduced initial states, interface blocks are initialized from the corresponding bulk blocks so `θ < 1` explicit terms have a consistent previous-step interface state.

## Constant-Operator Reuse

For fixed geometry, `solve_unsteady!` reuses one factorization when:

- matrix coefficients are time-independent,
- RHS data are time-independent,
- timestep remains at the nominal `dt`.

Otherwise it falls back to per-step assembly.
