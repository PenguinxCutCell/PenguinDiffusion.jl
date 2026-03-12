# Developer Notes

## Layout Conventions

- Mono: `ω, γ`
- Diph: `ω1, γ1, ω2, γ2`

Canonical contiguous layouts are fast-pathed; arbitrary `UnknownLayout` is supported through block insertion helpers.

## Why Identity Regularization

Halo/inactive rows are set to identity to keep assembled systems robust without changing index mappings.

Benefits:

- stable direct/iterative solves,
- simpler API (state layout is fixed),
- robust fresh/dead-cell handling in moving runs.

## Inactive Row Handling

- activity is derived from capacity volume/interface measure and halo masks,
- inactive rows are projected to identity with zero RHS.

## Adding a New BC / Interface Condition

1. Extend coefficient/value extraction in the relevant helper:
   - mono: `_interface_diagonals_mono`
   - diph: `_interface_coupling_diph`
2. Add/adjust assembly block terms.
3. Add fixed and moving tests.
4. Update `docs/src/interface_conditions.md` and `docs/src/api.md`.

## Time-Dependent Sampling Rules

- Fixed geometry unsteady: operator sampled at `t + θΔt`.
- Moving geometry unsteady:
  - slab geometry from `[t^n, t^{n+1}]`,
  - source/interface data: end-state for BE, `θ`-blend for `θ < 1`,
  - interface spatial sampling at slab `C_γ`.

## Moving Geometry Build

`_build_moving_slab!` performs:

1. space moments at `t^n` and `t^{n+1}`,
2. space-time moments on a 2-node slab,
3. reduction back to spatial operators/capacity for assembly.
