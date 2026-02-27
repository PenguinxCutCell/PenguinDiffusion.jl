using LinearAlgebra
using LinearSolve
using Printf

using CartesianGeometry
using CartesianOperators
using PenguinDiffusion
using PenguinSolverCore

"""
Two-phase steady manufactured verification on a square with a circular interface.

Manufactured field:
    u(x,y) = x^2 + y^2
with source:
    f = -4
for the semidiscrete model solved by PenguinDiffusion:
    0 = L(u) + V .* f

Interface constraints are set to enforce continuity:
    scalar jump: u1 - u2 = 0  via (α1, α2) = (1, 1)
    flux jump:   q1 + q2 = 0  via (b1, b2) = (1, -1)
"""

@inline u_analytical(x, y) = x^2 + y^2

function solve_twophase_steady(n::Int; radius::Float64=0.23)
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
    u_ref = zeros(Float64, Nd)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        xx = moments1.xyz[1][I[1]]
        yy = moments1.xyz[2][I[2]]
        u_ref[idx] = u_analytical(xx, yy)
    end

    bc1 = BoxBC(Val(2), Float64)
    bc2 = BoxBC((Dirichlet(copy(u_ref)), Dirichlet(copy(u_ref))), (Dirichlet(copy(u_ref)), Dirichlet(copy(u_ref))))

    # With current operators sign conventions, these choices enforce continuity.
    fluxjump = FluxJumpConstraint(ones(Float64, Nd), -ones(Float64, Nd), zeros(Float64, Nd))
    scalarjump = ScalarJumpConstraint(ones(Float64, Nd), ones(Float64, Nd), zeros(Float64, Nd))

    prob = TwoPhaseDiffusionProblem(1.0, 1.0, bc1, bc2, fluxjump, scalarjump, -4.0, -4.0)
    sys = build_system(moments1, moments2, prob)

    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nω = nω1 + nω2

    rhs0 = zeros(Float64, nω)
    PenguinSolverCore.rhs!(rhs0, sys, zeros(Float64, nω), nothing, 0.0)
    b = -rhs0

    tmp = zeros(Float64, nω)
    op! = (out, u, _x, _p, _t) -> PenguinDiffusion.apply_L!(out, sys, u)
    Aop = LinearSolve.FunctionOperator(op!, zeros(Float64, nω), tmp; isinplace=true, T=Float64, isconstant=true)
    lprob = LinearProblem(Aop, b; u0=zeros(Float64, nω))
    lsol = LinearSolve.solve(lprob, LinearSolve.SimpleGMRES(); reltol=1e-11, abstol=1e-11, maxiters=300_000)

    uω1_full, _, uω2_full, _ = full_state(sys, lsol.u)

    idx1 = sys.dof_omega1.indices
    idx2 = sys.dof_omega2.indices
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
    ns = (12, 16, 24, 32, 48, 64, 96, 128)
    hs = Float64[]
    errs = Float64[]

    @printf("Two-phase steady manufactured (square + circular interface)\n")
    @printf("%6s  %12s  %12s  %12s  %12s  %8s  %8s  %6s\n", "n", "h", "rel1", "rel2", "rel", "nω1", "nω2", "nγ")
    for n in ns
        h, rel1, rel2, rel, nω1, nω2, nγ = solve_twophase_steady(n)
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
