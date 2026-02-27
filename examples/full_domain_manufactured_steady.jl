using LinearAlgebra
using LinearSolve
using Printf
using SciMLBase

using CartesianGeometry
using CartesianOperators
using PenguinDiffusion

"""
Full-domain continuous manufactured steady diffusion with spatially varying Dirichlet BC.

Model used by PenguinDiffusion:
    0 = kappa * L(u) + b
with b = V .* f, where f is a source density.

Choose exact solution:
    u(x,y) = sin(pi*x)*sin(pi*y) + x + y

Then:
    Δu = -2*pi^2*sin(pi*x)*sin(pi*y)
    f(x,y) = -kappa * Δu = 2*kappa*pi^2*sin(pi*x)*sin(pi*y)

Dirichlet values on all box sides are set from the exact solution.
"""

function solve_case(nx::Int, ny::Int; kappa::Float64=1.4)
    x = collect(range(0.0, 1.0; length=nx + 1))
    y = collect(range(0.0, 1.0; length=ny + 1))
    full_domain(_x, _y, _t=0.0) = -1.0
    moments = geometric_moments(full_domain, (x, y), Float64, zero; method=:implicitintegration)

    dims = ntuple(d -> length(moments.xyz[d]), 2)
    Nd = prod(dims)
    li = LinearIndices(dims)

    u_exact_full = zeros(Float64, Nd)
    f_full = zeros(Float64, Nd)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        xx = moments.xyz[1][I[1]]
        yy = moments.xyz[2][I[2]]
        s = sin(pi * xx) * sin(pi * yy)
        u_exact_full[idx] = s + xx + yy
        f_full[idx] = 2.0 * kappa * pi^2 * s
    end

    bc = BoxBC(
        (Dirichlet(copy(u_exact_full)), Dirichlet(copy(u_exact_full))),
        (Dirichlet(copy(u_exact_full)), Dirichlet(copy(u_exact_full))),
    )

    ops = assembled_ops(moments; bc=bc)
    interface = RobinConstraint(ones(Float64, ops.Nd), zeros(Float64, ops.Nd), zeros(Float64, ops.Nd))
    prob = DiffusionProblem(kappa, bc, interface, f_full)
    sys = build_system(moments, prob)

    sol = steady_solve(
        sys;
        alg=LinearSolve.SimpleGMRES(),
        reltol=1e-12,
        abstol=1e-12,
        maxiters=50_000,
    )
    SciMLBase.successful_retcode(sol) || error("steady solve failed with retcode=$(sol.retcode)")

    u_num_full, _ = full_state(sys, sol.u)
    active = sys.dof_omega.indices

    V_active = Float64.(moments.V[active])
    err = u_num_full[active] .- u_exact_full[active]
    ref = u_exact_full[active]
    rel_l2 = sqrt(sum(V_active .* (err .^ 2)) / sum(V_active .* (ref .^ 2)))
    max_abs = maximum(abs, err)
    h = max(x[2] - x[1], y[2] - y[1])
    return h, rel_l2, max_abs, length(active)
end

function main()
    ns = (8, 16, 32, 64, 128, 256)
    hs = Float64[]
    errs = Float64[]

    @printf("Full-domain steady continuous manufactured test\n")
    @printf("%6s  %12s  %14s  %14s  %8s\n", "n", "h", "relL2", "maxAbs", "nω")
    for n in ns
        h, rel, maxabs, nω = solve_case(n, n)
        push!(hs, h)
        push!(errs, rel)
        @printf("%6d  %12.5e  %14.6e  %14.6e  %8d\n", n, h, rel, maxabs, nω)
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
