using LinearAlgebra
using LinearSolve
using Printf
using SciMLBase

using CartesianGeometry
using CartesianOperators
using PenguinDiffusion

"""
Continuous manufactured steady diffusion on the outside of a circle
with spatially varying kappa.

Domain:
    Omega = box minus disk

Model:
    div(kappa * grad(u)) + f = 0 in Omega
    a*u + b*dn(u) = g on interface (circle)
    u = u_exact on box boundary

This example uses:
    a = 1, b = 0  (so interface condition is u = g)
    g = u_exact sampled on the full grid (then restricted by Iγ internally)
    spatially varying box Dirichlet values from u_exact
    spatially varying kappa(x,y)
"""

@inline function u_exact(x, y)
    return sin(pi * x) * cos(pi * y) + 0.10 * x + 0.20 * y
end

@inline function grad_u_exact(x, y)
    ux = pi * cos(pi * x) * cos(pi * y) + 0.10
    uy = -pi * sin(pi * x) * sin(pi * y) + 0.20
    return ux, uy
end

@inline function lap_u_exact(x, y)
    return -2.0 * pi^2 * sin(pi * x) * cos(pi * y)
end

@inline function kappa_xy(x, y)
    return 1.0 + 0.30 * x + 0.20 * y
end

function solve_case(n::Int; radius::Float64=0.22, center=(0.5, 0.5))
    x = collect(range(0.0, 1.0; length=n + 1))
    y = collect(range(0.0, 1.0; length=n + 1))

    # Negative outside the circle -> outside region is active material.
    outside_circle(xp, yp, _t=0.0) = sqrt((xp - center[1])^2 + (yp - center[2])^2) - radius
    moments = geometric_moments(outside_circle, (x, y), Float64, zero; method=:vofi)

    dims = ntuple(d -> length(moments.xyz[d]), 2)
    Nd = prod(dims)
    li = LinearIndices(dims)

    u_full = zeros(Float64, Nd)
    f_full = zeros(Float64, Nd)
    g_full = zeros(Float64, Nd)
    kappa_full = zeros(Float64, Nd)
    kx = 0.30
    ky = 0.20
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        xx = moments.xyz[1][I[1]]
        yy = moments.xyz[2][I[2]]
        ue = u_exact(xx, yy)
        kap = kappa_xy(xx, yy)
        ux, uy = grad_u_exact(xx, yy)
        lapu = lap_u_exact(xx, yy)
        div_k_grad_u = kap * lapu + kx * ux + ky * uy

        u_full[idx] = ue
        g_full[idx] = ue
        kappa_full[idx] = kap
        f_full[idx] = -div_k_grad_u
    end

    # Space-varying Dirichlet on all box boundaries from the manufactured solution.
    bc = BoxBC(
        (Dirichlet(copy(u_full)), Dirichlet(copy(u_full))),
        (Dirichlet(copy(u_full)), Dirichlet(copy(u_full))),
    )

    # Interface Robin with a=1, b=0 -> u = g at the interface.
    interface = RobinConstraint(ones(Float64, Nd), zeros(Float64, Nd), g_full)
    prob = DiffusionProblem(kappa_full, bc, interface, f_full)
    sys = build_system(moments, prob)

    sol = steady_solve(
        sys;
        alg=LinearSolve.SimpleGMRES(),
        reltol=1e-11,
        abstol=1e-11,
        maxiters=80_000,
    )
    SciMLBase.successful_retcode(sol) || error("steady solve failed with retcode=$(sol.retcode)")

    u_num_full, _ = full_state(sys, sol.u)
    active = sys.dof_omega.indices
    V_active = Float64.(moments.V[active])

    err = u_num_full[active] .- u_full[active]
    ref = u_full[active]
    rel_l2 = sqrt(sum(V_active .* (err .^ 2)) / sum(V_active .* (ref .^ 2)))
    max_abs = maximum(abs, err)
    h = max(x[2] - x[1], y[2] - y[1])
    return h, rel_l2, max_abs, length(active), length(sys.dof_gamma.indices)
end

function main()
    ns = (8, 16, 32, 64, 128, 256)
    hs = Float64[]
    errs = Float64[]

    @printf("Outside-circle steady continuous manufactured test (Robin a=1,b=0 + box Dirichlet)\n")
    @printf("%6s  %12s  %14s  %14s  %8s  %8s\n", "n", "h", "relL2", "maxAbs", "nω", "nγ")
    for n in ns
        h, rel, maxabs, nω, nγ = solve_case(n)
        push!(hs, h)
        push!(errs, rel)
        @printf("%6d  %12.5e  %14.6e  %14.6e  %8d  %8d\n", n, h, rel, maxabs, nω, nγ)
    end

    if length(errs) > 1
        @printf("observed rates (relL2):\n")
        for i in 2:length(errs)
            r = log(errs[i - 1] / errs[i]) / log(hs[i - 1] / hs[i])
            @printf("  n=%d -> n=%d : %.3f\n", ns[i - 1], ns[i], r)
        end
    end
end

main()
