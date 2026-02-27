using LinearAlgebra
using LinearSolve
using Printf
using SciMLBase

using CartesianGeometry
using CartesianOperators
using PenguinDiffusion

"""
1D continuous manufactured steady diffusion.

Body:
    x - x0

Active region is x < x0 (single interface at x0), so one outer box boundary
condition is required at x=0. We impose spatial Dirichlet there from u_exact.
"""

@inline u_exact(x) = x^2 + 0.35 * x + 0.8
@inline d2u_exact(x) = 2.0

function solve_case(n::Int; x0::Float64=0.8501, kappa::Float64=1.0)
    x = collect(range(0.0, 1.0; length=n + 1))
    body(xp, _t=0.0) = xp - x0
    moments = geometric_moments(body, (x,), Float64, zero; method=:implicitintegration)

    dims = (length(moments.xyz[1]),)
    Nd = prod(dims)
    li = LinearIndices(dims)

    u_full = zeros(Float64, Nd)
    f_full = zeros(Float64, Nd)
    g_full = zeros(Float64, Nd)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        xx = moments.xyz[1][I[1]]
        ue = u_exact(xx)
        u_full[idx] = ue
        f_full[idx] = -kappa * d2u_exact(xx)
        g_full[idx] = ue
    end

    # One outer boundary needed (x=0 side). x=1 side is inactive for this body.
    bc = BoxBC((Dirichlet(copy(u_full)),), (Neumann(0.0),))
    interface = RobinConstraint(ones(Float64, Nd), zeros(Float64, Nd), g_full) # a=1, b=0
    prob = DiffusionProblem(kappa, bc, interface, f_full)
    sys = build_system(moments, prob)

    sol = steady_solve(
        sys;
        alg=LinearSolve.SimpleGMRES(),
        reltol=1e-12,
        abstol=1e-12,
        maxiters=40_000,
    )
    SciMLBase.successful_retcode(sol) || error("steady solve failed with retcode=$(sol.retcode)")

    u_num_full, _ = full_state(sys, sol.u)
    active = sys.dof_omega.indices
    V_active = Float64.(moments.V[active])
    err = u_num_full[active] .- u_full[active]
    ref = u_full[active]
    rel_l2 = sqrt(sum(V_active .* (err .^ 2)) / sum(V_active .* (ref .^ 2)))
    max_abs = maximum(abs, err)
    h = x[2] - x[1]
    return h, rel_l2, max_abs, length(active), length(sys.dof_gamma.indices)
end

function main()
    ns = (8, 16, 32, 64, 128, 256)
    hs = Float64[]
    errs = Float64[]

    @printf("1D manufactured (body=x-x0), one box Dirichlet side\n")
    @printf("%6s  %12s  %14s  %14s  %8s  %8s\n", "n", "h", "relL2", "maxAbs", "nω", "nγ")
    for n in ns
        h, rel, maxabs, nω, nγ = solve_case(n)
        push!(hs, h)
        push!(errs, rel)
        @printf("%6d  %12.5e  %14.6e  %14.6e  %8d  %8d\n", n, h, rel, maxabs, nω, nγ)
    end

    @printf("observed rates (relL2):\n")
    for i in 2:length(errs)
        r = log(errs[i - 1] / errs[i]) / log(hs[i - 1] / hs[i])
        @printf("  n=%d -> n=%d : %.3f\n", ns[i - 1], ns[i], r)
    end
end

main()
