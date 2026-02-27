using LinearAlgebra
using Printf

using CartesianGeometry
using CartesianOperators
using OrdinaryDiffEq
using PenguinDiffusion
using PenguinSolverCore
using SciMLBase

"""
Full-domain continuous manufactured unsteady diffusion with spatially and time-varying Dirichlet BC.

Analytic field:
    u(x,y,t) = exp(t) * (sin(pi*x)*sin(pi*y) + x + y)

Then:
    u_t = exp(t) * (sin(pi*x)*sin(pi*y) + x + y)
    Δu  = -2*pi^2 * exp(t) * sin(pi*x)*sin(pi*y)

For semidiscrete model M*du/dt = kappa*L(u) + V*f, use:
    f(x,y,t) = u_t - kappa*Δu

Dirichlet values on all box sides are updated from the exact solution each step.
"""

@inline shape(xx, yy) = sin(pi * xx) * sin(pi * yy) + xx + yy
@inline u_exact(xx, yy, t) = exp(t) * shape(xx, yy)
@inline source_density(xx, yy, t, kappa) = exp(t) * (shape(xx, yy) + 2.0 * kappa * pi^2 * sin(pi * xx) * sin(pi * yy))

function full_field!(out::Vector{Float64}, moments, t)
    dims = ntuple(d -> length(moments.xyz[d]), 2)
    li = LinearIndices(dims)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        out[idx] = u_exact(moments.xyz[1][I[1]], moments.xyz[2][I[2]], t)
    end
    return out
end

function source_field!(out::Vector{Float64}, moments, t, kappa)
    dims = ntuple(d -> length(moments.xyz[d]), 2)
    li = LinearIndices(dims)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        out[idx] = source_density(moments.xyz[1][I[1]], moments.xyz[2][I[2]], t, kappa)
    end
    return out
end

function main()
    nx, ny = 12, 12
    x = collect(range(0.0, 1.0; length=nx + 1))
    y = collect(range(0.0, 1.0; length=ny + 1))
    full_domain(_x, _y, _t=0.0) = -1.0
    moments = geometric_moments(full_domain, (x, y), Float64, zero; method=:implicitintegration)

    Nd = prod(ntuple(d -> length(moments.xyz[d]), 2))
    u_buf = zeros(Float64, Nd)
    f_buf = zeros(Float64, Nd)

    kappa = 1.1
    t0 = 0.0
    tf = 0.12
    full_field!(u_buf, moments, t0)

    bc = BoxBC(
        (Dirichlet(copy(u_buf)), Dirichlet(copy(u_buf))),
        (Dirichlet(copy(u_buf)), Dirichlet(copy(u_buf))),
    )

    ops = assembled_ops(moments; bc=bc)
    interface = RobinConstraint(ones(Float64, ops.Nd), zeros(Float64, ops.Nd), zeros(Float64, ops.Nd))
    sourcefun = (sys, _u, _p, t) -> source_field!(f_buf, moments, t, sys.kappa)
    prob = DiffusionProblem(kappa, bc, interface, sourcefun)
    sys = build_system(moments, prob)

    dirichlet_updater = BoxDirichletUpdater((sys, _u, _p, t) -> begin
        full_field!(u_buf, moments, t)
        (lo=(u_buf, u_buf), hi=(u_buf, u_buf))
    end)
    add_update!(sys, EveryStep(), dirichlet_updater)

    u0 = copy(u_buf[sys.dof_omega.indices])
    odeprob = sciml_odeproblem(sys, u0, (t0, tf); p=nothing, include_every_step=true)
    sol = SciMLBase.solve(
        odeprob,
        OrdinaryDiffEq.Rosenbrock23(autodiff=false);
        reltol=1e-8,
        abstol=1e-8,
        saveat=range(t0, tf; length=11),
    )
    SciMLBase.successful_retcode(sol) || error("time solve failed with retcode=$(sol.retcode)")

    full_field!(u_buf, moments, tf)
    u_num_full, _ = full_state(sys, sol.u[end])
    active = sys.dof_omega.indices
    V_active = Float64.(moments.V[active])
    err = u_num_full[active] .- u_buf[active]
    ref = u_buf[active]
    rel_l2 = sqrt(sum(V_active .* (err .^ 2)) / sum(V_active .* (ref .^ 2)))
    max_abs = maximum(abs, err)

    @printf("Full-domain unsteady continuous manufactured test\n")
    @printf("n_omega=%d, n_gamma=%d\n", length(active), length(sys.dof_gamma.indices))
    @printf("t_final=%.3f\n", tf)
    @printf("relative L2 error = %.6e\n", rel_l2)
    @printf("max abs error     = %.6e\n", max_abs)
end

main()
