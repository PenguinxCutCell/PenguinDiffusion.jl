# Interface Conditions

## Monophasic Embedded Condition

Type: `PenguinBCs.Robin(α, β, g)`

Discrete form (per interface DOF):

```math
\beta q + \alpha u_\gamma = g.
```

`q` is represented by operator blocks coupled to `u_ω` and `u_γ`.

## Diphasic Interface Conditions

Container: `InterfaceConditions(; scalar=..., flux=...)`

### Scalar-like row

- `ScalarJump(α₁, α₂, g)`
- `RobinJump(α, β, g)`

### Flux-like row

- `FluxJump(β₁, β₂, g)`
- `RobinJump(α, β, g)` (when used in the flux slot)

Rows map to unknown blocks as:

- `γ1` row: scalar-like relation,
- `γ2` row: flux-like relation.

## Sampling Convention

All interface coefficients/data are sampled at `C_γ`:

- fixed models: capacity `C_γ`,
- moving models: slab-reduced `C_γ`.

This convention is preserved for both BE and CN/generic `θ`.

## Practical Advice

- If only one relation is needed, set the other component to `nothing`.
- For no embedded interface, pass `bc_interface=nothing` (mono) or `ic=nothing` (diph).
