using LinearSolve
using SciMLBase

function build_steady_test_system(; kappa=2.3, a=1.7, b=0.9, g=0.4, source=1.25)
    moments = build_cut_moments()
    bc = CartesianOperators.BoxBC(Val(2), Float64)
    Nd = length(moments.V)
    interface = CartesianOperators.RobinConstraint(fill(a, Nd), fill(b, Nd), fill(g, Nd))
    prob = PenguinDiffusion.DiffusionProblem(kappa, bc, interface, source)
    return PenguinDiffusion.build_system(moments, prob)
end

function explicit_reduced_steady_system(sys::PenguinDiffusion.DiffusionSystem, source_scalar::Float64)
    n_omega = length(sys.dof_omega.indices)
    n_gamma = length(sys.dof_gamma.indices)

    A = Matrix(sys.L_oo)
    c = zeros(Float64, n_omega)
    if n_gamma > 0
        S = sys.C_gamma_fact \ Matrix(sys.C_omega)
        q = sys.C_gamma_fact \ sys.r_gamma
        A .-= Matrix(sys.L_og) * S
        c .= Matrix(sys.L_og) * q
    end
    A .*= sys.kappa

    V_active = Float64.(sys.moments.V[sys.dof_omega.indices])
    src = V_active .* source_scalar
    rhs = .-(sys.kappa .* (c .+ sys.dirichlet_affine) .+ src)
    return A, rhs
end

@testset "steady_solve matches explicit reduced solve" begin
    source_value = 1.25
    sys = build_steady_test_system(; source=source_value)

    A_ref, rhs_ref = explicit_reduced_steady_system(sys, source_value)
    u_ref = A_ref \ rhs_ref

    sol = PenguinDiffusion.steady_solve(
        sys;
        alg=LinearSolve.SimpleGMRES(),
        abstol=1e-12,
        reltol=1e-12,
        maxiters=10_000,
    )

    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u, u_ref; atol=1e-9, rtol=1e-9)
end

@testset "steady_linear_problem accepts time-dependent source callback" begin
    source_fun = (sys, u, p, t) -> p.shift + t
    sys = build_steady_test_system(; source=source_fun)
    p = (shift=0.2,)

    sol_t0 = PenguinDiffusion.steady_solve(
        sys;
        p=p,
        t=0.0,
        alg=LinearSolve.SimpleGMRES(),
        abstol=1e-12,
        reltol=1e-12,
        maxiters=10_000,
    )
    sol_t1 = PenguinDiffusion.steady_solve(
        sys;
        p=p,
        t=1.0,
        alg=LinearSolve.SimpleGMRES(),
        abstol=1e-12,
        reltol=1e-12,
        maxiters=10_000,
    )

    @test SciMLBase.successful_retcode(sol_t0)
    @test SciMLBase.successful_retcode(sol_t1)
    @test norm(sol_t0.u - sol_t1.u) > 1e-8
end

@testset "steady solver rejects zero kappa" begin
    sys = build_steady_test_system(; kappa=0.0, source=1.0)
    @test_throws ArgumentError PenguinDiffusion.steady_solve(sys)
end
