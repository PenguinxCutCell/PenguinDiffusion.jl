using LinearAlgebra
using Printf

using CartesianGeometry
using CartesianOperators
using OrdinaryDiffEq
using PenguinDiffusion
using PenguinSolverCore
using SciMLBase

"""
Unsteady diffusion inside a disk using matrix-free RHS kernels.

Model:
    M * du/dt = div(kappa * grad(u)) + V .* source
with Robin interface condition and default box Neumann.
"""

function main()
    # Geometry
    n = 28
    x = collect(range(0.0, 1.0; length=n + 1))
    y = collect(range(0.0, 1.0; length=n + 1))
    radius = 0.23
    center = (0.5, 0.5)
    disk_phi(xp, yp, _t=0.0) = sqrt((xp - center[1])^2 + (yp - center[2])^2) - radius
    moments = geometric_moments(disk_phi, (x, y), Float64, zero; method=:implicitintegration)

    # PDE data
    bc = BoxBC(Val(2), Float64)
    Nd = length(moments.V)
    a = ones(Float64, Nd)
    b = fill(0.8, Nd)
    g = fill(1.5, Nd)
    interface = RobinConstraint(a, b, g)

    # Build full spatially varying kappa on Nd nodes.
    dims = ntuple(d -> length(moments.xyz[d]), 2)
    li = LinearIndices(dims)
    kappa_full = zeros(Float64, Nd)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        xx = moments.xyz[1][I[1]]
        yy = moments.xyz[2][I[2]]
        kappa_full[idx] = 1.0 + 0.3 * xx + 0.1 * yy
    end

    source = (sys, u, p, t) -> 0.15 * exp(-t)
    prob = DiffusionProblem(kappa_full, bc, interface, source)
    sys = build_matrixfree_system(moments, prob)

    @printf("Built matrix-free system: n_omega=%d, n_gamma=%d\n", length(sys.dof_omega.indices), length(sys.dof_gamma.indices))

    # Initial condition
    u0 = fill(0.25, length(sys.dof_omega.indices))
    tspan = (0.0, 0.6)
    odeprob = sciml_odeproblem(sys, u0, tspan; p=nothing)
    sol = SciMLBase.solve(
        odeprob,
        OrdinaryDiffEq.Rosenbrock23(autodiff=false);
        reltol=1e-8,
        abstol=1e-8,
        saveat=range(tspan[1], tspan[2]; length=7),
    )
    SciMLBase.successful_retcode(sol) || error("time integration failed with retcode=$(sol.retcode)")

    uf, _ = full_state(sys, sol.u[end])
    active = sys.dof_omega.indices
    @printf("t_final = %.3f\n", tspan[2])
    @printf("u_min(active)=%.6f, u_max(active)=%.6f\n", minimum(uf[active]), maximum(uf[active]))
end

main()
