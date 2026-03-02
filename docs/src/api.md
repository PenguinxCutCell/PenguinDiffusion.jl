**API & Types**

This section lists the primary public types and functions you will interact with.

Types

- `DiffusionModelMono{N,T,DT,ST,IT}`
  - Fields: `ops::DiffusionOps`, `cap::AssembledCapacity`, `D::DT` (diffusivity or callback), `source::ST`, `bc_border::BorderConditions`, `bc_interface::IT`, `layout::UnknownLayout`.
  - Construct with `DiffusionModelMono(cap, ops, D; source=..., bc_border=..., bc_interface=..., layout=...)`.

- `DiffusionModelDiph{N,T,D1T,D2T,ST,IT}`
  - Fields: `D1`, `D2` (per-phase diffusivities), similar other fields.
  - Construct with `DiffusionModelDiph(cap, ops, D1, D2; ...)`.

Key functions

- `assemble_steady_mono!(sys::LinearSystem, model::DiffusionModelMono, t::T)`
- `assemble_unsteady_mono!(sys::LinearSystem, model::DiffusionModelMono, uⁿ, t::T, dt::T, scheme)`
- `assemble_steady_diph!(sys::LinearSystem, model::DiffusionModelDiph, t::T)`
- `assemble_unsteady_diph!(sys::LinearSystem, model::DiffusionModelDiph, uⁿ, t::T, dt::T, scheme)`
- `solve_steady!(model::DiffusionModelMono; t::T=zero(T), method::Symbol=:direct, kwargs...)`
- `solve_steady!(model::DiffusionModelDiph; t::T=zero(T), method::Symbol=:direct, kwargs...)`

Callbacks and arguments

- Diffusivity (`D`, `D1`, `D2`) can be constants or callables evaluated as `eval_coeff(D, x, t, i)` by the sampling helper.
- `source` may be a numeric constant, a function of `(x... )`, or `(x..., t)`; for diph it may be a function returning a tuple `(s1,s2)` or a tuple of two callbacks/constants.

Layout and offsets

- Use `layout_mono(ntotal)` and `layout_diph(ntotal)` to get `UnknownLayout` giving offsets for ω and γ variables; `model.layout.offsets` contains ranges used when assembling global matrices.

Boundary conditions

- `bc_border::BorderConditions` supports `Dirichlet`, `Neumann`, `Robin`, and `Periodic` on domain boundaries.
- `bc_interface::InterfaceConditions` supports `ScalarJump`, `FluxJump`, and `RobinJump` types to represent embedded-interface constraints.

Helpers

- Several internal helpers exist but are documented here because they are useful for extending functionality: `_sample_coeff`, `_source_values_mono`, `_interface_diagonals_mono`, and `_insert_block!`.

See the examples page for common usage patterns and test-driven examples.