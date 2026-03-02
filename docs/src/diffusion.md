**Theory and PDE**

This package discretizes diffusion problems on cut-cell grids (possibly with embedded interfaces). The continuous mono-species diffusion PDE solved is

$$
\frac{\partial u}{\partial t} = \nabla \cdot (D(x,t) \nabla u) + s(x,t),
$$

with appropriate boundary conditions on the domain boundary and jump/interface conditions on embedded interfaces. For two-phase (diph) models we solve a system for $(u_1,u_2)$ with possibly distinct diffusivities $D_1,D_2$ and coupled interface conditions.

Interface conditions are expressed in two components: scalar jumps and flux jumps. For a scalar jump we may specify

$$\alpha_1 u_1 - \alpha_2 u_2 = g, $$

and for flux jumps we may specify relations involving normal fluxes (Robin/Flux types).

Discrete operators

The code builds discrete difference operators via per-direction gradient and weighting matrices. Using the internal operator names:

- $G$ and $H$ are discrete gradient and interface sampling operators (stacked per direction).
- $W^{-1}$ (exposed as `Winv`) is the block-diagonal inverse weighting matrix.

From these the following assembled matrices appear in the algebraic system:

$$
K = G^{T} W^{-1} G,\qquad C = G^{T} W^{-1} H,\qquad J = H^{T} W^{-1} G,\qquad L = H^{T} W^{-1} H.
$$

These are used to form per-phase block matrices. For a mono model with diagonal diffusivity $D_\omega$ the continuous-to-discrete mapping produces block entries (using diagonal operators $\mathrm{diag}(\cdot)$):

- $A_{11} = \mathrm{diag}(D_\omega) \, K$
- $A_{12} = \mathrm{diag}(D_\omega) \, C$
- $A_{21} = \mathrm{diag}(\beta) \, J$
- $A_{22} = \mathrm{diag}(\beta) \, L + \mathrm{diag}(\alpha) \, \Gamma$

where $\Gamma$ contains the interface surface measures and $\alpha,\beta$ are interface-diagonal contributions coming from the interface conditions. When no interface is present the interface blocks are reduced to identity constraints.

Treatment of sources and capacity

Cellwise source terms are sampled to a vector $f_\omega$ and multiplied by cell capacity `V` via

$$ b_1 = V f_\omega, \qquad b_2 = \Gamma g. $$

Unsteady formulation

For backward Euler / θ-methods the mass-term contribution is added to the diagonal of $A_{11}$ (or both phase ω blocks for diph) via

$$ M = V / \Delta t, $$

and the right-hand side gets the corresponding $M u^{n}$ contribution.

Boundary and halo handling

- Box boundary conditions are applied with `apply_box_bc_mono!`, which modifies rows of the global matrix and RHS and enforces Dirichlet/Neumann/Periodic behavior on the halo cells.
- Halo rows and inactive (zero-volume) cells are set to identity to keep the linear system well-posed and numerically stable.
