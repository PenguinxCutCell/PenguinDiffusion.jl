using LinearAlgebra
using Printf

using CartesianGeometry
using CartesianOperators
using OrdinaryDiffEq
using PenguinDiffusion
using PenguinSolverCore
using SciMLBase

"""
Two-phase unsteady manufactured verification on a square with a circular interface.

Reference field:
    u(x,y,t) = exp(-t) * v(x,y),   where v solves -Δv = 1 with v|∂Ω = 0.
Hence:
    u_t = Δu + f,  with  f(x,y,t) = exp(-t) * (1 - v(x,y)).

Interface constraints enforce value and flux continuity between phases.
"""

function v_analytical(x, y, lx, ly; max_mode::Int=121)
    s = 0.0
    for m in 1:2:max_mode
        sx = sin(m * pi * x / lx)
        for n in 1:2:max_mode
            s += 16 * lx^2 / (pi^4 * m * n * (m^2 + n^2)) * sx * sin(n * pi * y / ly)
        end
    end
    return s
end

function solve_twophase_unsteady(n::Int; radius::Float64=0.23, tf::Float64=0.1, max_mode::Int=121)
    lx = 1.0
    ly = 1.0
    x = collect(range(0.0, lx; length=n + 1))
    y = collect(range(0.0, ly; length=n + 1))
    center = (0.5, 0.5)

    body(x, y, _=0.0) = sqrt((x - center[1])^2 + (y - center[2])^2) - radius
    moments1 = geometric_moments(body, (x, y), Float64, zero; method=:implicitintegration)
    moments2 = geometric_moments((x, y, t=0.0) -> -body(x, y, t), (x, y), Float64, zero; method=:implicitintegration)

    Nd = length(moments1.V)
    dims = ntuple(d -> length(moments1.xyz[d]), 2)
    li = LinearIndices(dims)

    vfull = zeros(Float64, Nd)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        xx = moments1.xyz[1][I[1]]
        yy = moments1.xyz[2][I[2]]
        vfull[idx] = v_analytical(xx, yy, lx, ly; max_mode=max_mode)
    end

    bc1 = BoxBC(Val(2), Float64)
    bc2 = BoxBC((Dirichlet(0.0), Dirichlet(0.0)), (Dirichlet(0.0), Dirichlet(0.0)))

    fluxjump = FluxJumpConstraint(ones(Float64, Nd), -ones(Float64, Nd), zeros(Float64, Nd))
    scalarjump = ScalarJumpConstraint(ones(Float64, Nd), ones(Float64, Nd), zeros(Float64, Nd))

    sourcefun = (_sys, _u, _p, t) -> @. exp(-t) * (1 - vfull)
    prob = TwoPhaseDiffusionProblem(1.0, 1.0, bc1, bc2, fluxjump, scalarjump, sourcefun, sourcefun)
    sys = build_system(moments1, moments2, prob)

    idx1 = sys.dof_omega1.indices
    idx2 = sys.dof_omega2.indices
    u0 = vcat(vfull[idx1], vfull[idx2])

    odeprob = sciml_odeproblem(sys, copy(u0), (0.0, tf); p=nothing)
    sol = SciMLBase.solve(
        odeprob,
        OrdinaryDiffEq.Rosenbrock23(autodiff=false);
        reltol=1e-7,
        abstol=1e-7,
        save_everystep=false,
    )
    SciMLBase.successful_retcode(sol) || error("unsteady solve failed with retcode=$(sol.retcode)")

    uω1_full, _, uω2_full, _ = full_state(sys, sol.u[end])
    u_ref = exp(-tf) .* vfull

    V1 = Float64.(moments1.V[idx1])
    V2 = Float64.(moments2.V[idx2])

    e1 = uω1_full[idx1] .- u_ref[idx1]
    e2 = uω2_full[idx2] .- u_ref[idx2]
    rel1 = sqrt(sum(V1 .* (e1 .^ 2)) / max(sum(V1 .* (u_ref[idx1] .^ 2)), eps(Float64)))
    rel2 = sqrt(sum(V2 .* (e2 .^ 2)) / max(sum(V2 .* (u_ref[idx2] .^ 2)), eps(Float64)))
    rel = sqrt(0.5 * (rel1^2 + rel2^2))
    h = max(x[2] - x[1], y[2] - y[1])

    return h, rel1, rel2, rel, length(idx1), length(idx2), length(sys.dof_gamma.indices)
end

function main()
    ns = (12, 16)
    hs = Float64[]
    errs = Float64[]

    @printf("Two-phase unsteady manufactured (square + circular interface)\n")
    @printf("%6s  %12s  %12s  %12s  %12s  %8s  %8s  %6s\n", "n", "h", "rel1", "rel2", "rel", "nω1", "nω2", "nγ")
    for n in ns
        h, rel1, rel2, rel, nω1, nω2, nγ = solve_twophase_unsteady(n)
        push!(hs, h)
        push!(errs, rel)
        @printf("%6d  %12.5e  %12.5e  %12.5e  %12.5e  %8d  %8d  %6d\n", n, h, rel1, rel2, rel, nω1, nω2, nγ)
    end

    @printf("Observed rates (combined rel):\n")
    for i in 2:length(ns)
        rate = log(errs[i - 1] / errs[i]) / log(hs[i - 1] / hs[i])
        @printf("  n=%d -> n=%d : %.3f\n", ns[i - 1], ns[i], rate)
    end
end

main()
