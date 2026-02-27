using LinearSolve
using SciMLBase

function build_two_phase_steady_system(; source1=0.12, source2=-0.07, matrixfree_unsteady=false)
    n = 18
    x = collect(range(0.0, 1.0; length=n + 1))
    y = collect(range(0.0, 1.0; length=n + 1))
    body(x, y, _=0.0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.23

    moments1 = geometric_moments(body, (x, y), Float64, zero; method=:implicitintegration)
    moments2 = geometric_moments((x, y, t=0.0) -> -body(x, y, t), (x, y), Float64, zero; method=:implicitintegration)

    Nd = length(moments1.V)
    bc1 = BoxBC(Val(2), Float64)
    bc2 = BoxBC((Dirichlet(0.0), Dirichlet(0.0)), (Dirichlet(0.0), Dirichlet(0.0)))
    fluxjump = FluxJumpConstraint(ones(Float64, Nd), -ones(Float64, Nd), zeros(Float64, Nd))
    scalarjump = ScalarJumpConstraint(ones(Float64, Nd), ones(Float64, Nd), zeros(Float64, Nd))
    prob = TwoPhaseDiffusionProblem(1.2, 1.8, bc1, bc2, fluxjump, scalarjump, source1, source2)

    if matrixfree_unsteady
        return build_matrixfree_system(moments1, moments2, prob)
    end
    return build_system(moments1, moments2, prob)
end

@testset "Two-phase unsteady matrix-free RHS contract" begin
    src1 = (_sys, _u, _p, t) -> 0.2 + 0.1 * t
    src2 = (_sys, _u, _p, t) -> -0.1 + 0.05 * t
    sys_a = build_two_phase_system(; kappa1=1.3, kappa2=0.7, source1=src1, source2=src2, matrixfree_unsteady=false)
    sys_mf = build_two_phase_system(; kappa1=1.3, kappa2=0.7, source1=src1, source2=src2, matrixfree_unsteady=true)

    nω = length(sys_a.dof_omega1.indices) + length(sys_a.dof_omega2.indices)
    u = randn(nω)
    du_a = zeros(Float64, nω)
    du_mf = zeros(Float64, nω)

    PenguinSolverCore.rhs!(du_a, sys_a, u, nothing, 0.37)
    PenguinSolverCore.rhs!(du_mf, sys_mf, u, nothing, 0.37)
    @test isapprox(du_mf, du_a; atol=1e-11, rtol=1e-11)
end

@testset "Two-phase steady solve matrix-free contract" begin
    source1 = 0.12
    source2 = -0.07
    sys_a = build_two_phase_steady_system(; source1=source1, source2=source2, matrixfree_unsteady=false)
    sys_mf = build_two_phase_steady_system(; source1=source1, source2=source2, matrixfree_unsteady=true)

    nω = length(sys_a.dof_omega1.indices) + length(sys_a.dof_omega2.indices)
    rhs0 = zeros(Float64, nω)
    PenguinSolverCore.rhs!(rhs0, sys_a, zeros(Float64, nω), nothing, 0.0)
    b = -rhs0

    op! = (out, x, _u, _p, _t) -> PenguinDiffusion.apply_L!(out, sys_a, x)
    Aop = LinearSolve.FunctionOperator(op!, zeros(Float64, nω), zeros(Float64, nω); isinplace=true, T=Float64, isconstant=true)
    lprob = LinearSolve.LinearProblem(Aop, b; u0=zeros(Float64, nω))
    lsol_ref = LinearSolve.solve(
        lprob,
        LinearSolve.SimpleGMRES();
        reltol=1e-10,
        abstol=1e-10,
        maxiters=250_000,
    )

    fill!(sys_mf.Loo1.nzval, 0.0)
    fill!(sys_mf.Log1.nzval, 0.0)
    fill!(sys_mf.Loo2.nzval, 0.0)
    fill!(sys_mf.Log2.nzval, 0.0)
    fill!(sys_mf.dir_aff1, 3.0)
    fill!(sys_mf.dir_aff2, -2.0)

    lsol_mf = PenguinDiffusion.steady_solve(
        sys_mf;
        alg=LinearSolve.SimpleGMRES(),
        reltol=1e-10,
        abstol=1e-10,
        maxiters=250_000,
    )

    @test SciMLBase.successful_retcode(lsol_ref)
    @test SciMLBase.successful_retcode(lsol_mf)
    @test isapprox(lsol_mf.u, lsol_ref.u; atol=1e-8, rtol=1e-8)
end

@testset "Two-phase steady solver accepts time-dependent source callbacks" begin
    source1 = (_sys, _u, p, t) -> p.s1 + t
    source2 = (_sys, _u, p, t) -> p.s2 - 0.5 * t
    p = (s1=0.1, s2=-0.2)
    sys = build_two_phase_steady_system(; source1=source1, source2=source2, matrixfree_unsteady=true)

    sol_t0 = PenguinDiffusion.steady_solve(
        sys;
        p=p,
        t=0.0,
        alg=LinearSolve.SimpleGMRES(),
        reltol=1e-10,
        abstol=1e-10,
        maxiters=250_000,
    )
    sol_t1 = PenguinDiffusion.steady_solve(
        sys;
        p=p,
        t=1.0,
        alg=LinearSolve.SimpleGMRES(),
        reltol=1e-10,
        abstol=1e-10,
        maxiters=250_000,
    )

    @test SciMLBase.successful_retcode(sol_t0)
    @test SciMLBase.successful_retcode(sol_t1)
    @test norm(sol_t0.u - sol_t1.u) > 1e-8
end
