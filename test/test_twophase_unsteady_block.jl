using LinearSolve

@testset "Two-phase unsteady block matrix assembly" begin
    sys = build_two_phase_system(; kappa1=1.1, kappa2=0.9, source1=0.0, source2=0.0)
    dt = 0.04

    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)
    nω = nω1 + nω2
    nfull = nω1 + nγ + nω2 + nγ

    M1 = sys.M[1:nω1, 1:nω1]
    M2 = sys.M[(nω1 + 1):nω, (nω1 + 1):nω]

    Cω_flux = sys.Cω[1:nγ, :]
    Cω_flux1 = Cω_flux[:, 1:nω1]
    Cω_flux2 = Cω_flux[:, (nω1 + 1):(nω1 + nω2)]
    Cγ_flux1 = sys.Cγ[1:nγ, 1:nγ]
    Cγ_flux2 = sys.Cγ[1:nγ, (nγ + 1):(2 * nγ)]
    Cγ_scal1 = sys.Cγ[(nγ + 1):(2 * nγ), 1:nγ]
    Cγ_scal2 = sys.Cγ[(nγ + 1):(2 * nγ), (nγ + 1):(2 * nγ)]

    Zω1ω2 = spzeros(Float64, nω1, nω2)
    Zω1γ2 = spzeros(Float64, nω1, nγ)
    Zγω1 = spzeros(Float64, nγ, nω1)
    Zγω2 = spzeros(Float64, nγ, nω2)
    Zω2ω1 = spzeros(Float64, nω2, nω1)
    Zω2γ1 = spzeros(Float64, nω2, nγ)

    A_be = diphasic_unsteady_block_matrix(sys, dt; scheme=:BE)
    A11_be = M1 - dt * sys.Loo1
    A12_be = -dt * sys.Log1
    A33_be = M2 - dt * sys.Loo2
    A34_be = -dt * sys.Log2
    Aexp_be = vcat(
        hcat(A11_be, A12_be, Zω1ω2, Zω1γ2),
        hcat(Zγω1, Cγ_scal1, Zγω2, Cγ_scal2),
        hcat(Zω2ω1, Zω2γ1, A33_be, A34_be),
        hcat(Cω_flux1, Cγ_flux1, Cω_flux2, Cγ_flux2),
    )
    @test size(A_be) == (nfull, nfull)
    @test isapprox(norm(A_be - Aexp_be), 0.0; atol=1e-11, rtol=1e-11)

    halfdt = dt / 2
    A_cn = diphasic_unsteady_block_matrix(sys, dt; scheme=:CN)
    A11_cn = M1 - halfdt * sys.Loo1
    A12_cn = -halfdt * sys.Log1
    A33_cn = M2 - halfdt * sys.Loo2
    A34_cn = -halfdt * sys.Log2
    Aexp_cn = vcat(
        hcat(A11_cn, A12_cn, Zω1ω2, Zω1γ2),
        hcat(Zγω1, Cγ_scal1, Zγω2, Cγ_scal2),
        hcat(Zω2ω1, Zω2γ1, A33_cn, A34_cn),
        hcat(Cω_flux1, Cγ_flux1, Cω_flux2, Cγ_flux2),
    )
    @test isapprox(norm(A_cn - Aexp_cn), 0.0; atol=1e-11, rtol=1e-11)
end

@testset "Two-phase unsteady block BE one-step matches explicit block solve" begin
    sys = build_two_phase_system(; kappa1=1.2, kappa2=0.8, source1=0.0, source2=0.0)
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)
    nω = nω1 + nω2
    nfull = nω1 + nγ + nω2 + nγ
    dt = 0.03

    u0ω = randn(nω)
    state0 = zeros(Float64, nfull)
    @inbounds for i in 1:nω1
        state0[i] = u0ω[i]
    end
    @inbounds for i in 1:nω2
        state0[nω1 + nγ + i] = u0ω[nω1 + i]
    end
    uγ0 = zeros(Float64, 2 * nγ)
    PenguinDiffusion.solve_uγ!(uγ0, sys, u0ω)
    @inbounds for i in 1:nγ
        state0[nω1 + i] = uγ0[i]
        state0[nω1 + nγ + nω2 + i] = uγ0[nγ + i]
    end

    A = diphasic_unsteady_block_matrix(sys, dt; scheme=:BE)
    b = zeros(Float64, nfull)
    b[1:nω1] .= (sys.M[1:nω1, 1:nω1] * state0[1:nω1])
    b[(nω1 + 1):(nω1 + nγ)] .= sys.r[(nγ + 1):(2 * nγ)]
    b[(nω1 + nγ + 1):(nω1 + nγ + nω2)] .=
        (sys.M[(nω1 + 1):(nω1 + nω2), (nω1 + 1):(nω1 + nω2)] * state0[(nω1 + nγ + 1):(nω1 + nγ + nω2)])
    b[(nω1 + nγ + nω2 + 1):nfull] .= sys.r[1:nγ]

    lprob = LinearSolve.LinearProblem(A, b; u0=state0)
    lsol = LinearSolve.solve(lprob, LinearSolve.KLUFactorization())
    x_ref = Vector{Float64}(lsol.u)

    sol = diphasic_unsteady_block_solve(
        sys,
        u0ω,
        (0.0, dt);
        dt=dt,
        scheme=:BE,
        alg=LinearSolve.KLUFactorization(),
    )
    x_num = sol.states[end]
    @test isapprox(x_num, x_ref; atol=1e-10, rtol=1e-10)
end

@testset "Two-phase unsteady block loop advances with short final step and scheduled updates" begin
    sys = build_two_phase_system(; kappa1=1.0, kappa2=1.1, source1=0.0, source2=0.0)
    nω = length(sys.dof_omega1.indices) + length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)

    gupd = FluxJumpGUpdater((sys, u, p, t) -> fill(t, nγ))
    PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.2]), gupd)

    u0 = zeros(Float64, nω)
    sol = diphasic_unsteady_block_solve(
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

    expected_flux = 0.2 .* sys.ops1.Iγ[sys.dof_gamma.indices]
    @test isapprox(sys.r[1:nγ], expected_flux; atol=1e-14, rtol=1e-14)
end
