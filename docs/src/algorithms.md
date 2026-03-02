**Numerical Algorithms & Routines**

This page summarizes high-level algorithms implemented by the core routines.

1) Assembly of steady mono-phase system (`assemble_steady_mono!`)

- Inputs: `sys::LinearSystem`, `model::DiffusionModelMono`, time `t`.
- Steps:
  - Compute core operators: $K,C,J,L$ from `ops` via `K = G' * Winv * G`, etc.
  - Sample diffusivity: $D_\omega$ at cell centroids (or callback) producing a diagonal `ID`.
  - Sample source: build vector $f_\omega` by evaluating provided source callbacks.
  - Build interface diagonals: compute `α,β,g_γ` from `bc_interface` over the interface mask.
  - Construct block matrices $A_{11}, A_{12}, A_{21}, A_{22}$ as described in Theory.
  - For canonical layouts (`ω,γ` stacked by blocks), assemble global `A` using sparse block concatenation.
  - For non-canonical layouts, fall back to offset-based sparse insertion.
  - Apply box boundary conditions with `apply_box_bc_mono!`.
  - Enforce halo and inactive-cell identity rows so halo/inactive do not pollute solves.

2) Assembly of unsteady mono-phase (`assemble_unsteady_mono!`)

- Builds the steady system at time $t+\theta\Delta t$ (where `scheme` gives $\theta$),
- Adds mass diagonal $M = V/\Delta t$ into the ω-ω block and the RHS $M u^n$.

3) Assembly for two-phase / diph systems (`assemble_steady_diph!` / `assemble_unsteady_diph!`)

- Similar structure but with two ω/γ blocks (phase 1 and phase 2).
- Each phase uses its own sampled $D_1, D_2$ and interface diagonal contributions `α1,α2,β1,β2`.
- Box BCs are applied per-phase by temporarily constructing an `UnknownLayout` restricted to the ω-block for each phase.

4) Solving

- `solve_steady!` helper constructs a `LinearSystem`, calls the appropriate assemble routine, then invokes `solve!` from `PenguinSolverCore` with the selected method (direct or iterative).
- `solve_unsteady!` advances in time with `:BE` / `:CN` (or numeric `θ`) and supports a constant-operator fast path:
  - if matrix and RHS coefficients are time-invariant, it assembles once and reuses the factorization across timesteps,
  - if not, it falls back to per-step unsteady assembly.

Performance / numerical notes

- Canonical mono/diph systems use direct sparse block concatenation for lower assembly overhead.
- Non-canonical custom layouts still use sparse offset insertion via `_insert_block!`.
- Halo/inactive constraints are enforced through sparse masking (`M*A*M + P`) to avoid expensive row-by-row CSC edits.
- Interface sampling is only active where `cap.buf.Γ` (interface measure) is finite and positive; the code computes a boolean mask for these locations.

Reference to core symbols in the implementation: `K,C,J,L`, `G,H,Winv`, `cap.V`, `cap.Γ`, `apply_box_bc_mono!`, and the assembly helpers `_insert_block!`, `_insert_vec!`, `_set_row_identity!`.
