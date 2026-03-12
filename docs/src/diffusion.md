# Diffusion Models and PDEs

## Monophasic Model

In a domain `Ω(t)` with outer boundary `∂Ω`, solve

```math
\partial_t u = \nabla \cdot (D(x,t)\nabla u) + s(x,t)
```

with outer boundary conditions (Dirichlet/Neumann/Robin/Periodic depending on side).

For fixed geometry (`DiffusionModelMono`), `Ω` is time-independent.
For moving geometry (`MovingDiffusionModelMono`), assembly uses a space-time slab.

## Diphasic Model

For two phases separated by an embedded interface `Γ`:

```math
\partial_t u_k = \nabla \cdot (D_k(x,t)\nabla u_k) + s_k(x,t), \quad k=1,2.
```

Interface coupling is represented by `InterfaceConditions` with two optional rows:

- scalar-like row (`ScalarJump` or `RobinJump`),
- flux-like row (`FluxJump` or `RobinJump`).

Typical forms:

```math
\alpha_1 u_1 - \alpha_2 u_2 = g_s,
```

```math
\beta_1 q_1 + \beta_2 q_2 = g_f,
```

where `q_k` are discrete normal-flux traces.

## Outer Boundary Conditions

`BorderConditions` supports:

- `Dirichlet(value)`,
- `Neumann(value)`,
- `Robin(α, β, value)`,
- `Periodic()`.

Default side condition is homogeneous Neumann (`0`).

## Embedded Interface Conditions

- Mono embedded condition: `PenguinBCs.Robin(α, β, g)`.
- Diph embedded condition: `InterfaceConditions(scalar=..., flux=...)`.

Important convention:

- Interface/jump coefficients and values are sampled at `C_γ`.
- This includes fixed and moving assembly paths.
- In moving slab assembly there is one natural slab `C_γ` sample set per step; temporal blending uses that same spatial sample set.

## Callback Conventions

Diffusivity/source/interface coefficients can be:

- constants,
- space-dependent callbacks `(x...)`,
- time-dependent callbacks `(x..., t)`.

Examples:

```julia
D(x, y, t) = 1 + 0.2sin(2pi*t)
s(x, t) = x*(1 - x)
gγ(x, t) = exp(-t)
```

## Geometry Assumptions

- Fixed models use a fixed cut-cell geometry (`assembled_capacity`).
- Moving models rebuild per-step slab geometry from `SpaceTimeCartesianGrid`.
- Halo and inactive rows are regularized to identity to keep the global linear system robust.

See [Algorithms](algorithms.md) for discrete block systems and θ assembly.
