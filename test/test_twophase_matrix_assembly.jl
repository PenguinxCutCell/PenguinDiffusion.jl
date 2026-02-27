using LinearAlgebra
using SparseArrays

@testset "Two-phase matrix assembly and sign conventions" begin
    n = 14
    x = collect(range(0.0, 1.0; length=n + 1))
    y = collect(range(0.0, 1.0; length=n + 1))
    body(x, y, _=0.0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.23

    moments1 = geometric_moments(body, (x, y), Float64, zero; method=:implicitintegration)
    moments2 = geometric_moments((x, y, t=0.0) -> -body(x, y, t), (x, y), Float64, zero; method=:implicitintegration)

    bc1 = BoxBC(Val(2), Float64)
    bc2 = BoxBC(Val(2), Float64)

    Nd = length(moments1.V)
    fluxjump = FluxJumpConstraint(fill(1.3, Nd), fill(-0.9, Nd), fill(0.2, Nd))
    scalarjump = ScalarJumpConstraint(fill(1.1, Nd), fill(0.7, Nd), fill(-0.4, Nd))
    sys = build_system(moments1, moments2, TwoPhaseDiffusionProblem(1.0, 1.0, bc1, bc2, fluxjump, scalarjump, 0.0, 0.0))

    idxω1 = sys.dof_omega1.indices
    idxω2 = sys.dof_omega2.indices
    idxγ = sys.dof_gamma.indices
    nω1 = length(idxω1)
    nω2 = length(idxω2)
    nγ = length(idxγ)

    @test nγ > 0

    KWinv1 = spdiagm(0 => sys.kappa_face1) * sys.ops1.Winv
    KWinv2 = spdiagm(0 => sys.kappa_face2) * sys.ops2.Winv
    Loo1_exp = sparse(-sys.ops1.G' * KWinv1 * sys.ops1.G)[idxω1, idxω1]
    Log1_exp = sparse(-sys.ops1.G' * KWinv1 * sys.ops1.H)[idxω1, idxγ]
    Loo2_exp = sparse(-sys.ops2.G' * KWinv2 * sys.ops2.G)[idxω2, idxω2]
    Log2_exp = sparse(-sys.ops2.G' * KWinv2 * sys.ops2.H)[idxω2, idxγ]

    @test isapprox(norm(sys.Loo1 - Loo1_exp), 0.0; atol=1e-11, rtol=1e-11)
    @test isapprox(norm(sys.Log1 - Log1_exp), 0.0; atol=1e-11, rtol=1e-11)
    @test isapprox(norm(sys.Loo2 - Loo2_exp), 0.0; atol=1e-11, rtol=1e-11)
    @test isapprox(norm(sys.Log2 - Log2_exp), 0.0; atol=1e-11, rtol=1e-11)

    Cω1_f, Cγ1_f, Cω2_f, Cγ2_f, r_f = fluxjump_constraint_matrices(sys.ops1, sys.ops2, sys.fluxjump)
    Cγ1_s, Cγ2_s, r_s = scalarjump_constraint_matrices(sys.ops1, sys.ops2, sys.scalarjump)

    Cω_flux_exp = hcat(Cω1_f[idxγ, idxω1], Cω2_f[idxγ, idxω2])
    Cγ_flux_exp = hcat(Cγ1_f[idxγ, idxγ], Cγ2_f[idxγ, idxγ])
    Cγ_scal_exp = hcat(Cγ1_s[idxγ, idxγ], Cγ2_s[idxγ, idxγ])
    r_flux_exp = r_f[idxγ]
    r_scal_exp = r_s[idxγ]

    @test isapprox(norm(sys.Cω[1:nγ, :] - Cω_flux_exp), 0.0; atol=1e-11, rtol=1e-11)
    @test isapprox(norm(sys.Cγ[1:nγ, :] - Cγ_flux_exp), 0.0; atol=1e-11, rtol=1e-11)
    @test isapprox(norm(sys.Cγ[(nγ + 1):(2 * nγ), :] - Cγ_scal_exp), 0.0; atol=1e-11, rtol=1e-11)
    @test nnz(sys.Cω[(nγ + 1):(2 * nγ), :]) == 0
    @test isapprox(norm(sys.r[1:nγ] - r_flux_exp), 0.0; atol=1e-12, rtol=1e-12)
    @test isapprox(norm(sys.r[(nγ + 1):(2 * nγ)] - r_scal_exp), 0.0; atol=1e-12, rtol=1e-12)

    # Explicit scalar-jump sign convention in current operators:
    #   -Iγ*α1*uγ1 + Iγ*α2*uγ2 = Iγ*g
    diag_s1 = diag(sys.Cγ[(nγ + 1):(2 * nγ), 1:nγ])
    diag_s2 = diag(sys.Cγ[(nγ + 1):(2 * nγ), (nγ + 1):(2 * nγ)])
    @test isapprox(diag_s1, -sys.ops1.Iγ[idxγ] .* sys.scalarjump.α1[idxγ]; atol=1e-12, rtol=1e-12)
    @test isapprox(diag_s2,  sys.ops1.Iγ[idxγ] .* sys.scalarjump.α2[idxγ]; atol=1e-12, rtol=1e-12)

    Cω_flux = sys.Cω[1:nγ, :]
    Cγ_flux_1 = sys.Cγ[1:nγ, 1:nγ]
    Cγ_flux_2 = sys.Cγ[1:nγ, (nγ + 1):(2 * nγ)]
    Cγ_scal_1 = sys.Cγ[(nγ + 1):(2 * nγ), 1:nγ]
    Cγ_scal_2 = sys.Cγ[(nγ + 1):(2 * nγ), (nγ + 1):(2 * nγ)]

    Cω_flux_1 = Cω_flux[:, 1:nω1]
    Cω_flux_2 = Cω_flux[:, (nω1 + 1):(nω1 + nω2)]

    A = sparse(vcat(
        hcat(sys.Loo1, sys.Log1, spzeros(nω1, nω2), spzeros(nω1, nγ)),
        hcat(spzeros(nγ, nω1), Cγ_scal_1, spzeros(nγ, nω2), Cγ_scal_2),
        hcat(spzeros(nω2, nω1), spzeros(nω2, nγ), sys.Loo2, sys.Log2),
        hcat(Cω_flux_1, Cγ_flux_1, Cω_flux_2, Cγ_flux_2),
    ))

    uω = randn(nω1 + nω2)
    uγ = zeros(Float64, 2 * nγ)
    PenguinDiffusion.solve_uγ!(uγ, sys, uω)

    xfull = vcat(uω[1:nω1], uγ[1:nγ], uω[(nω1 + 1):(nω1 + nω2)], uγ[(nγ + 1):(2 * nγ)])
    yfull = A * xfull

    yω = zeros(Float64, nω1 + nω2)
    PenguinDiffusion.apply_L!(yω, sys, uω)

    rows_ω1 = 1:nω1
    rows_s = (nω1 + 1):(nω1 + nγ)
    rows_ω2 = (nω1 + nγ + 1):(nω1 + nγ + nω2)
    rows_f = (nω1 + nγ + nω2 + 1):(nω1 + 2 * nγ + nω2)

    @test isapprox(yfull[rows_ω1], yω[1:nω1]; atol=1e-10, rtol=1e-10)
    @test isapprox(yfull[rows_ω2], yω[(nω1 + 1):(nω1 + nω2)]; atol=1e-10, rtol=1e-10)
    @test isapprox(yfull[rows_s], sys.r[(nγ + 1):(2 * nγ)]; atol=1e-10, rtol=1e-10)
    @test isapprox(yfull[rows_f], sys.r[1:nγ]; atol=1e-10, rtol=1e-10)
end
