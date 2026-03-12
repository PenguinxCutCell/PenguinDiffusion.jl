# Algorithms

## Core Discrete Operators

From `ops = DiffusionOps(cap)`:

```math
K = G^\top W^{-1} G,\quad
C = G^\top W^{-1} H,\quad
J = H^\top W^{-1} G,\quad
L = H^\top W^{-1} H.
```

With coefficient weighting, assembly effectively uses weighted variants of these blocks.

## Unknown Layouts

### Mono

Unknown blocks are `ω` (cell values) and `γ` (interface traces):

```text
x = [ uω
      uγ ]
```

Canonical mono layout uses contiguous blocks of length `ntotal`.

### Diph

Unknown blocks are `ω1, γ1, ω2, γ2`:

```text
x = [ uω1
      uγ1
      uω2
      uγ2 ]
```

Canonical diphasic layout uses four contiguous blocks of length `ntotal`.

## Steady Mono Assembly

At time `t`:

```math
\begin{bmatrix}
A_{11} & A_{12}\\
A_{21} & A_{22}
\end{bmatrix}
\begin{bmatrix}
u_\omega\\
u_\gamma
\end{bmatrix}
=
\begin{bmatrix}
b_\omega\\
b_\gamma
\end{bmatrix}
```

with

```math
A_{11}=K,\;
A_{12}=C,\;
A_{21}=\operatorname{diag}(\beta)J,\;
A_{22}=\operatorname{diag}(\beta)L+\operatorname{diag}(\alpha)\Gamma,
```

```math
b_\omega = V f_\omega,\quad b_\gamma = \Gamma g_\gamma.
```

## Unsteady Mono (`θ`)

Fixed geometry:

1. Assemble steady operator at `t + θΔt`.
2. Scale previous-step contribution on `ω` rows by `(1-θ)`.
3. Add mass diagonal `M = V/Δt` on `ω` rows.

Moving geometry:

- build slab (`V^n`, `V^{n+1}`, slab `cap/ops`),
- use `Ψ+`, `Ψ-` weights for birth/death handling,
- sample diffusivity-weighted operator blocks at `t^n + θΔt`,
- sample/interface data at slab `C_γ`,
- for `θ<1`, blend interface/source data between `t^n` and `t^{n+1}`.

## Steady Diph Assembly

Two per-phase diffusion rows plus two interface rows:

```text
[ ω1-row ]  diffusion phase 1
[ γ1-row ]  scalar-like interface relation
[ ω2-row ]  diffusion phase 2
[ γ2-row ]  flux-like interface relation
```

Supported interface combinations:

- scalar row: `ScalarJump` or `RobinJump`,
- flux row: `FluxJump` or `RobinJump`,
- either row can be omitted (`nothing`), defaulting to identity regularization on that block.

## Unsteady Diph (`θ`)

Fixed geometry follows the same pattern as mono, applied to both `ω1` and `ω2`.

Moving diphasic:

- slab geometry and per-phase volumes are rebuilt every step,
- `Ψ` weights control active/inactive transitions,
- source data are `θ`-blended in time,
- scalar and flux interface coefficients/data are `θ`-blended in time on slab `C_γ`.

## Inactive/Halo Regularization

Inactive or halo rows are transformed to identity constraints:

- keeps systems non-singular in cut/dead-cell situations,
- stabilizes moving-interface steps with fresh/dead cells,
- avoids removing rows/cols and preserves consistent layouts.

## Constant-Operator Reuse in `solve_unsteady!`

For fixed-geometry models:

- detect matrix and RHS time dependence from callbacks,
- if both are time-invariant and `dt` is unchanged:
  assemble once, reuse factorization, update RHS only,
- otherwise assemble every step.

Moving models currently rebuild each step (no constant-operator reuse).
