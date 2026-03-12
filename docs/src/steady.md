# Steady Solves

## Monophasic

Use `DiffusionModelMono` + `solve_steady!`.

```julia
model = DiffusionModelMono(cap, ops, D; source=source, bc_border=bc, bc_interface=ic)
sys = solve_steady!(model)
```

State layout:

- `sys.x[model.layout.offsets.ω]`: bulk unknowns,
- `sys.x[model.layout.offsets.γ]`: interface trace unknowns.

## Diphasic

Use `DiffusionModelDiph` + `solve_steady!`.

```julia
ic = InterfaceConditions(
    scalar=ScalarJump(1.0, 1.0, 0.0),
    flux=FluxJump(1.0, 1.0, 0.0),
)
model = DiffusionModelDiph(cap1, ops1, D1, s1, cap2, ops2, D2, s2; bc_border=bc, ic=ic)
sys = solve_steady!(model)
```

State layout:

- `ω1`, `γ1`, `ω2`, `γ2` blocks in `model.layout.offsets`.

## Notes

- Outer BCs are enforced after block assembly.
- Inactive/halo rows are regularized to identity.
- Interface coefficients/data are sampled at `C_γ`.
