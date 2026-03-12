# API Reference

## Constructor Keywords

### `DiffusionModelMono`

- `source`: constant or callback `(x...)` / `(x..., t)`
- `bc_border`: `BorderConditions`
- `bc_interface`: `Union{Nothing, PenguinBCs.Robin}`
- `layout`: `UnknownLayout` (default `layout_mono(cap.ntotal)`)
- `coeff_mode`: `:harmonic` (default), `:arithmetic`, `:face`, `:cell`

### `DiffusionModelDiph`

- `source`: tuple `(s1, s2)` or callback returning a tuple
- `bc_border`: `BorderConditions`
- `ic` or `bc_interface`: `Union{Nothing, InterfaceConditions}`
- `layout`: `UnknownLayout` (default `layout_diph(cap.ntotal)`)
- `coeff_mode`: `:harmonic` / `:arithmetic` / `:face` / `:cell`

### Moving Constructors

- `MovingDiffusionModelMono(grid, body, D; ...)`
- `MovingDiffusionModelDiph(grid, body1, D1, D2; body2=nothing, ...)`

with additional keyword:

- `geom_method` (default `:vofijul`)

## Time Scheme Values

All unsteady APIs accept:

- `:BE`
- `:CN`
- numeric `θ` with `0 ≤ θ ≤ 1`

## Expected State Shapes

- Mono:
  - reduced: `ntotal` (`ω`)
  - full: `length(model.layout.offsets.γ)` (`ω+γ`)
- Diph:
  - reduced: `2*ntotal` (`ω1+ω2`)
  - full: full stacked system (`ω1,γ1,ω2,γ2`)

Reduced initial states are expanded internally.

## Public Symbols

```@docs
PenguinDiffusion.DiffusionModelMono
PenguinDiffusion.DiffusionModelDiph
PenguinDiffusion.MovingDiffusionModelMono
PenguinDiffusion.MovingDiffusionModelDiph
PenguinDiffusion.assemble_steady_mono!
PenguinDiffusion.assemble_unsteady_mono!
PenguinDiffusion.assemble_steady_diph!
PenguinDiffusion.assemble_unsteady_diph!
PenguinDiffusion.assemble_unsteady_mono_moving!
PenguinDiffusion.assemble_unsteady_diph_moving!
PenguinDiffusion.solve_steady!
PenguinDiffusion.solve_unsteady!
PenguinDiffusion.solve_unsteady_moving!
PenguinDiffusion.compute_interface_exchange_metrics
```
