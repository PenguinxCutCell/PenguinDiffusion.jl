using LinearSolve

@testset "Unsteady block matrix assembly" begin
    sys = build_test_system(; kappa=1.4, source=0.0)
    dt = 0.07

    nω = length(sys.dof_omega.indices)
    nγ = length(sys.dof_gamma.indices)

    A_be = unsteady_block_matrix(sys, dt; scheme=:BE)
    A11_be = sparse(sys.M - dt * sys.L_oo)
    A12_be = sparse(-dt * sys.L_og)
    Aexp_be = sparse(vcat(hcat(A11_be, A12_be), hcat(sys.C_omega, sys.C_gamma)))
    @test size(A_be) == (nω + nγ, nω + nγ)
    @test isapprox(norm(A_be - Aexp_be), 0.0; atol=1e-12, rtol=1e-12)

    A_cn = unsteady_block_matrix(sys, dt; scheme=:CN)
    halfdt = dt / 2
    A11_cn = sparse(sys.M - halfdt * sys.L_oo)
    A12_cn = sparse(-halfdt * sys.L_og)
    A21_cn = sparse(sys.C_omega)
    A22_cn = sparse(sys.C_gamma)
    Aexp_cn = sparse(vcat(hcat(A11_cn, A12_cn), hcat(A21_cn, A22_cn)))
    @test isapprox(norm(A_cn - Aexp_cn), 0.0; atol=1e-12, rtol=1e-12)
end

@testset "Unsteady block BE one-step matches explicit block solve" begin
    sys = build_test_system(; kappa=1.2, source=0.25)
    nω = length(sys.dof_omega.indices)
    nγ = length(sys.dof_gamma.indices)
    dt = 0.05

    u0 = randn(nω)
    uγ0 = zeros(Float64, nγ)
    PenguinDiffusion.solve_x_gamma!(uγ0, sys, u0)

    A = unsteady_block_matrix(sys, dt; scheme=:BE)

    src_mass = zeros(Float64, nω)
    PenguinDiffusion._source_to_reduced!(src_mass, sys, 0.25)
    bω = sys.M * u0 + dt .* src_mass
    b = vcat(bω, sys.r_gamma)

    x0 = vcat(u0, uγ0)
    lprob = LinearSolve.LinearProblem(A, b; u0=x0)
    lsol = LinearSolve.solve(lprob, LinearSolve.KLUFactorization())
    x_ref = Vector{Float64}(lsol.u)

    sol = unsteady_block_solve(
        sys,
        u0,
        (0.0, dt);
        dt=dt,
        scheme=:BE,
        alg=LinearSolve.KLUFactorization(),
    )
    x_num = sol.states[end]

    @test isapprox(x_num, x_ref; atol=1e-11, rtol=1e-11)
end

@testset "Unsteady block loop advances with short final step and scheduled updates" begin
    sys = build_test_system(; kappa=1.0, source=0.0)
    nω = length(sys.dof_omega.indices)
    nγ = length(sys.dof_gamma.indices)

    gupd = RobinGUpdater((sys, u, p, t) -> fill(t, nγ))
    PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.2]), gupd)

    u0 = zeros(Float64, nω)
    sol = unsteady_block_solve(
        sys,
        u0,
        (0.0, 0.25);
        dt=0.1,
        scheme="BE",
        alg=LinearSolve.KLUFactorization(),
        save_everystep=true,
    )

    @test length(sol.t) == 4
    @test isapprox(sol.t[1], 0.0; atol=1e-14)
    @test isapprox(sol.t[2], 0.1; atol=1e-14)
    @test isapprox(sol.t[3], 0.2; atol=1e-14)
    @test isapprox(sol.t[4], 0.25; atol=1e-14)
    @test sys.rebuild_calls == 0

    expected = 0.2 .* sys.ops.Iγ[sys.dof_gamma.indices]
    @test isapprox(sys.r_gamma, expected; atol=1e-14, rtol=1e-14)
end
