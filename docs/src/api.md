**API & Types**

This section lists the primary public types and functions you will interact with.

Types

- `DiffusionModelMono{N,T,DT,ST,IT}`
  - Fields: `ops::DiffusionOps`, `cap::AssembledCapacity`, `D::DT` (diffusivity or callback), `source::ST`, `bc_border::BorderConditions`, `bc_interface::IT`, `layout::UnknownLayout`.
  - Construct with `DiffusionModelMono(cap, ops, D; source=..., bc_border=..., bc_interface=..., layout=...)`.

- `MovingDiffusionModelMono{N,T,...}`
  - Fields: `grid::CartesianGrid`, `body` (time-dependent level set), `D`, `source`, `bc_border`, `bc_interface`, `layout`, `coeff_mode`, `geom_method`, and per-step caches (`cap_slab`, `ops_slab`, `Vn`, `Vn1`).
  - Construct with `MovingDiffusionModelMono(grid, body, D; source=..., bc_border=..., bc_interface=..., coeff_mode=..., geom_method=...)`.

- `MovingDiffusionModelDiph{N,T,...}`
  - Fields: `grid::CartesianGrid`, `body1`, optional `body2` (defaults to `-body1`), per-phase `D1/D2`, `source1/source2`, `bc_border`, `ic`, `layout`, `coeff_mode`, `geom_method`, and per-step caches (`cap1_slab/ops1_slab/V1n/V1n1`, `cap2_slab/ops2_slab/V2n/V2n1`).
  - Construct with `MovingDiffusionModelDiph(grid, body1, D1, D2; source=..., body2=nothing, bc_border=..., ic=..., coeff_mode=..., geom_method=...)`.

- `DiffusionModelDiph{N,T,D1T,D2T,ST,IT}`
  - Fields: `D1`, `D2` (per-phase diffusivities), similar other fields.
  - Construct with `DiffusionModelDiph(cap, ops, D1, D2; ...)`.

Key functions

- `assemble_steady_mono!(sys::LinearSystem, model::DiffusionModelMono, t::T)`
- `assemble_unsteady_mono!(sys::LinearSystem, model::DiffusionModelMono, uⁿ, t::T, dt::T, scheme)`
- `assemble_steady_diph!(sys::LinearSystem, model::DiffusionModelDiph, t::T)`
- `assemble_unsteady_diph!(sys::LinearSystem, model::DiffusionModelDiph, uⁿ, t::T, dt::T, scheme)`
- `assemble_unsteady_mono_moving!(sys::LinearSystem, model::MovingDiffusionModelMono, uⁿ, t::T, dt::T; scheme=:CN|:BE)`
- `assemble_unsteady_diph_moving!(sys::LinearSystem, model::MovingDiffusionModelDiph, uⁿ, t::T, dt::T; scheme=:CN|:BE)`
- `solve_steady!(model::DiffusionModelMono; t::T=zero(T), method::Symbol=:direct, kwargs...)`
- `solve_steady!(model::DiffusionModelDiph; t::T=zero(T), method::Symbol=:direct, kwargs...)`
- `solve_unsteady!(model::DiffusionModelMono, u0, tspan; dt, scheme=:BE|:CN|θ, method=:direct, save_history=true, kwargs...)`
- `solve_unsteady!(model::MovingDiffusionModelMono, u0, tspan; dt, scheme=:BE|:CN, method=:direct, save_history=true, kwargs...)`
- `solve_unsteady_moving!(model::MovingDiffusionModelMono, u0, tspan; dt, scheme=:BE|:CN, method=:direct, save_history=true, kwargs...)`
- `solve_unsteady!(model::MovingDiffusionModelDiph, u0, tspan; dt, scheme=:BE|:CN, method=:direct, save_history=true, kwargs...)`
- `solve_unsteady_moving!(model::MovingDiffusionModelDiph, u0, tspan; dt, scheme=:BE|:CN, method=:direct, save_history=true, kwargs...)`
- `solve_unsteady!(model::DiffusionModelDiph, u0, tspan; dt, scheme=:BE|:CN|θ, method=:direct, save_history=true, kwargs...)`
- `compute_interface_exchange_metrics(model, state_or_system; diffusivity_scale=nothing, characteristic_scale=1, reference_value=0)`:
  computes generic interface-transfer outputs (integrated/mean normal gradient, integrated/mean flux, mean interface value, exchange coefficient, and a dimensionless transfer index). For diphasic models, returns per-phase metrics and a flux-balance indicator.

Callbacks and arguments

- Diffusivity (`D`, `D1`, `D2`) can be constants or callables evaluated as `eval_coeff(D, x, t, i)`.
- Variable-coefficient diffusion is assembled with face coefficients (`coeff_mode`): `:harmonic` (default), `:arithmetic`, or direct `:face` sampling.
- `source` may be a numeric constant, a function of `(x... )`, or `(x..., t)`; for diph it may be a function returning a tuple `(s1,s2)` or a tuple of two callbacks/constants.

Layout and offsets

- Use `layout_mono(ntotal)` and `layout_diph(ntotal)` to get `UnknownLayout` giving offsets for ω and γ variables; `model.layout.offsets` contains ranges used when assembling global matrices.
- `solve_unsteady!` accepts either reduced initial vectors (`ω` for mono, `ω1+ω2` for diph) or full system vectors.

Boundary conditions

- `bc_border::BorderConditions` supports `Dirichlet`, `Neumann`, `Robin`, and `Periodic` on domain boundaries.
- Monophasic interface BC uses `PenguinBCs.Robin(α, β, g)`.
- Diphasic interface BC uses `InterfaceConditions` (`ScalarJump`, `FluxJump`, `RobinJump`).

Helpers

- Several internal helpers exist but are documented here because they are useful for extending functionality: `_sample_coeff`, `_source_values_mono`, `_interface_diagonals_mono`, and `_insert_block!`.

See the examples page for common usage patterns and test-driven examples.
