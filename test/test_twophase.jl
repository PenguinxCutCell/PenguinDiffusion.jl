function build_two_phase_moments()
    x = collect(range(0.0, 1.0; length=12))
    y = collect(range(0.0, 1.0; length=11))
    levelset(x, y, _=0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.24
    moments1 = CartesianGeometry.geometric_moments(levelset, (x, y), Float64, zero; method=:implicitintegration)
    moments2 = CartesianGeometry.geometric_moments((x, y, t=0.0) -> -levelset(x, y, t), (x, y), Float64, zero; method=:implicitintegration)
    return moments1, moments2
end

function build_two_phase_system(; kappa1=1.0, kappa2=1.4, source1=nothing, source2=nothing)
    moments1, moments2 = build_two_phase_moments()
    bc1 = CartesianOperators.BoxBC(Val(2), Float64)
    bc2 = CartesianOperators.BoxBC(Val(2), Float64)
    Nd = length(moments1.V)

    fluxjump = CartesianOperators.FluxJumpConstraint(ones(Float64, Nd), -ones(Float64, Nd), zeros(Float64, Nd))
    scalarjump = CartesianOperators.ScalarJumpConstraint(ones(Float64, Nd), -ones(Float64, Nd), zeros(Float64, Nd))
    prob = PenguinDiffusion.TwoPhaseDiffusionProblem(kappa1, kappa2, bc1, bc2, fluxjump, scalarjump, source1, source2)
    return PenguinDiffusion.build_system(moments1, moments2, prob)
end

@testset "Two-phase masking and constraint block sizes" begin
    sys = build_two_phase_system()
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)

    @test nω1 > 0
    @test nω2 > 0
    @test nγ > 0
    @test size(sys.M, 1) == nω1 + nω2
    @test size(sys.M, 2) == nω1 + nω2
    @test size(sys.Cγ) == (2 * nγ, 2 * nγ)
    @test size(sys.Cω) == (2 * nγ, nω1 + nω2)
    @test length(sys.r) == 2 * nγ
    @test sys.Cγ_fact !== nothing
end

@testset "Two-phase reduction satisfies stacked constraints" begin
    sys = build_two_phase_system()
    nω = length(sys.dof_omega1.indices) + length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)

    u = randn(nω)
    uγ = zeros(Float64, 2 * nγ)
    PenguinDiffusion.solve_uγ!(uγ, sys, u)

    resid = sys.Cω * u + sys.Cγ * uγ - sys.r
    @test isapprox(norm(resid), 0.0; atol=1e-10, rtol=1e-10)
end

@testset "Two-phase rhs and updater contract" begin
    sys = build_two_phase_system(; source1=0.2, source2=0.1)
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nω = nω1 + nω2
    nγ = length(sys.dof_gamma.indices)

    u = zeros(Float64, nω)
    du = zeros(Float64, nω)
    PenguinSolverCore.rhs!(du, sys, u, nothing, 0.0)
    @test all(isfinite, du)
    @test norm(du) > 0.0

    old_r = copy(sys.r)
    gupd = PenguinDiffusion.FluxJumpGUpdater((sys, u, p, t) -> fill(t, nγ))
    PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.5]), gupd)
    PenguinSolverCore.apply_scheduled_updates!(sys, u, nothing, 0.5; step=0)
    @test sys.rebuild_calls == 0
    @test norm(sys.r[1:nγ] - old_r[1:nγ]) > 0.0

    bupd = PenguinDiffusion.FluxJumpBUpdater((sys, u, p, t) -> (b1=2.0, b2=-0.5))
    PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.75]), bupd)
    PenguinSolverCore.apply_scheduled_updates!(sys, u, nothing, 0.75; step=1)
    @test sys.rebuild_calls == 1
    @test sys.constraints_dirty == false
end
