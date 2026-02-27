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
    sqrt((x-x0)^2) - r = abs(x-x0) - r

Active region is the inside interval [x0-r, x0+r], bounded by two interfaces.
No outer box Dirichlet boundary is required in this setup.
"""

@inline u_exact(x) = x^2 + 0.20 * x + 1.0
@inline d2u_exact(x) = 2.0

function solve_case(n::Int; x0::Float64=0.5, r::Float64=0.3, kappa::Float64=1.0)
    x = collect(range(0.0, 1.0; length=n + 1))
    body(xp, _t=0.0) = sqrt((xp - x0)^2) - r
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

    bc = BoxBC(Val(1), Float64) # default Neumann(0) on box boundaries
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

    @printf("1D manufactured (body=abs(x-x0)-r), no box Dirichlet\n")
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
