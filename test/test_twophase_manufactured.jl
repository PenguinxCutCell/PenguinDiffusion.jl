using LinearAlgebra
using LinearSolve
using SciMLBase
using OrdinaryDiffEq

function _series_poisson_square(x, y, lx, ly; max_mode::Int=81)
    s = 0.0
    for m in 1:2:max_mode
        sx = sin(m * pi * x / lx)
        for n in 1:2:max_mode
            s += 16 * lx^2 / (pi^4 * m * n * (m^2 + n^2)) * sx * sin(n * pi * y / ly)
        end
    end
    return s
end

function _build_twophase_square_circle(n::Int)
    lx = 1.0
    ly = 1.0
    x = collect(range(0.0, lx; length=n + 1))
    y = collect(range(0.0, ly; length=n + 1))

    radius = 0.23
    center = (0.5, 0.5)
    body(x, y, _=0.0) = sqrt((x - center[1])^2 + (y - center[2])^2) - radius

    moments1 = geometric_moments(body, (x, y), Float64, zero; method=:implicitintegration)
    moments2 = geometric_moments((x, y, t=0.0) -> -body(x, y, t), (x, y), Float64, zero; method=:implicitintegration)

    Nd = length(moments1.V)
    bc1 = BoxBC(Val(2), Float64)
    bc2 = BoxBC((Dirichlet(0.0), Dirichlet(0.0)), (Dirichlet(0.0), Dirichlet(0.0)))

    fluxjump = FluxJumpConstraint(ones(Float64, Nd), -ones(Float64, Nd), zeros(Float64, Nd))
    scalarjump = ScalarJumpConstraint(ones(Float64, Nd), ones(Float64, Nd), zeros(Float64, Nd))

    return moments1, moments2, TwoPhaseDiffusionProblem(1.0, 1.0, bc1, bc2, fluxjump, scalarjump, nothing, nothing)
end

function _quadratic_field(moments)
    dims = ntuple(d -> length(moments.xyz[d]), 2)
    Nd = prod(dims)
    li = LinearIndices(dims)
    out = zeros(Float64, Nd)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        xx = moments.xyz[1][I[1]]
        yy = moments.xyz[2][I[2]]
        out[idx] = xx^2 + yy^2
    end
    return out
end

function _reference_field(moments; max_mode::Int=81)
    dims = ntuple(d -> length(moments.xyz[d]), 2)
    Nd = prod(dims)
    li = LinearIndices(dims)
    out = zeros(Float64, Nd)
    @inbounds for I in CartesianIndices(dims)
        idx = li[I]
        out[idx] = _series_poisson_square(moments.xyz[1][I[1]], moments.xyz[2][I[2]], 1.0, 1.0; max_mode=max_mode)
    end
    return out
end

function _combined_rel_error(sys, moments1, moments2, uω1_full, uω2_full, uref_full)
    idx1 = sys.dof_omega1.indices
    idx2 = sys.dof_omega2.indices
    V1 = Float64.(moments1.V[idx1])
    V2 = Float64.(moments2.V[idx2])

    e1 = uω1_full[idx1] .- uref_full[idx1]
    e2 = uω2_full[idx2] .- uref_full[idx2]
    rel1 = sqrt(sum(V1 .* (e1 .^ 2)) / max(sum(V1 .* (uref_full[idx1] .^ 2)), eps(Float64)))
    rel2 = sqrt(sum(V2 .* (e2 .^ 2)) / max(sum(V2 .* (uref_full[idx2] .^ 2)), eps(Float64)))
    return sqrt(0.5 * (rel1^2 + rel2^2))
end

function _solve_twophase_steady_quadratic(n::Int)
    moments1, moments2, prob0 = _build_twophase_square_circle(n)
    uref = _quadratic_field(moments1)

    bc2 = BoxBC((Dirichlet(copy(uref)), Dirichlet(copy(uref))), (Dirichlet(copy(uref)), Dirichlet(copy(uref))))
    prob = TwoPhaseDiffusionProblem(
        prob0.kappa1,
        prob0.kappa2,
        prob0.bc1,
        bc2,
        prob0.fluxjump,
        prob0.scalarjump,
        -4.0,
        -4.0,
    )
    sys = build_system(moments1, moments2, prob)

    nω = length(sys.dof_omega1.indices) + length(sys.dof_omega2.indices)
    rhs0 = zeros(Float64, nω)
    PenguinSolverCore.rhs!(rhs0, sys, zeros(Float64, nω), nothing, 0.0)
    b = -rhs0

    tmp = zeros(Float64, nω)
    op! = (out, x, _u, _p, _t) -> PenguinDiffusion.apply_L!(out, sys, x)
    Aop = LinearSolve.FunctionOperator(op!, zeros(Float64, nω), tmp; isinplace=true, T=Float64, isconstant=true)
    lprob = LinearProblem(Aop, b; u0=zeros(Float64, nω))
    lsol = LinearSolve.solve(lprob, LinearSolve.SimpleGMRES(); reltol=1e-10, abstol=1e-10, maxiters=250_000)
    SciMLBase.successful_retcode(lsol) || error("steady two-phase solve failed with retcode=$(lsol.retcode)")

    uω1_full, _, uω2_full, _ = full_state(sys, lsol.u)
    rel = _combined_rel_error(sys, moments1, moments2, uω1_full, uω2_full, uref)
    return rel
end

@testset "Two-phase manufactured steady square-circle continuity" begin
    e24 = _solve_twophase_steady_quadratic(24)
    e48 = _solve_twophase_steady_quadratic(48)
    rate = log(e24 / e48) / log(2.0)

    @test e48 < e24
    @test rate > 1.3
end

@testset "Two-phase manufactured unsteady square-circle continuity" begin
    moments1, moments2, prob0 = _build_twophase_square_circle(16)
    uref = _reference_field(moments1)

    sourcefun = (_sys, _u, _p, t) -> @. exp(-t) * (1 - uref)
    prob = TwoPhaseDiffusionProblem(prob0.kappa1, prob0.kappa2, prob0.bc1, prob0.bc2, prob0.fluxjump, prob0.scalarjump, sourcefun, sourcefun)
    sys = build_system(moments1, moments2, prob)

    idx1 = sys.dof_omega1.indices
    idx2 = sys.dof_omega2.indices
    u0 = vcat(uref[idx1], uref[idx2])

    tf = 0.08
    odeprob = sciml_odeproblem(sys, copy(u0), (0.0, tf); p=nothing)
    sol = SciMLBase.solve(
        odeprob,
        OrdinaryDiffEq.Rosenbrock23(autodiff=false);
        reltol=1e-7,
        abstol=1e-7,
        save_everystep=false,
    )
    @test SciMLBase.successful_retcode(sol)

    uω1_full, _, uω2_full, _ = full_state(sys, sol.u[end])
    uref_t = exp(-tf) .* uref
    rel = _combined_rel_error(sys, moments1, moments2, uω1_full, uω2_full, uref_t)
    @test rel < 0.23
end
