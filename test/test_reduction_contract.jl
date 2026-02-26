using Random

@testset "Reduction contract" begin
    Random.seed!(0xD1FF)

    n_omega = 7
    n_gamma = 3

    L_oo = sprandn(Float64, n_omega, n_omega, 0.75)
    L_og = sprandn(Float64, n_omega, n_gamma, 0.85)
    C_omega = sprandn(Float64, n_gamma, n_omega, 0.85)
    C_gamma = sprandn(Float64, n_gamma, n_gamma, 1.00) + spdiagm(0 => fill(4.0, n_gamma))
    C_gamma_fact = lu(C_gamma)
    r_gamma = randn(n_gamma)
    u = randn(n_omega)

    tmp_gamma = zeros(Float64, n_gamma)
    tmp_omega = zeros(Float64, n_omega)
    out = zeros(Float64, n_omega)

    PenguinDiffusion.apply_L!(out, L_oo, L_og, C_omega, C_gamma_fact, r_gamma, tmp_gamma, tmp_omega, u)

    x_gamma = C_gamma_fact \ (r_gamma - C_omega * u)
    reference = L_oo * u + L_og * x_gamma
    @test isapprox(out, reference; atol=1e-12, rtol=1e-12)
end

@testset "Reduction contract (no gamma)" begin
    Random.seed!(0xCAFE)

    n_omega = 6
    L_oo = sprandn(Float64, n_omega, n_omega, 0.7)
    L_og = spzeros(Float64, n_omega, 0)
    C_omega = spzeros(Float64, 0, n_omega)
    r_gamma = zeros(Float64, 0)
    u = randn(n_omega)

    tmp_gamma = zeros(Float64, 0)
    tmp_omega = zeros(Float64, n_omega)
    out = zeros(Float64, n_omega)

    PenguinDiffusion.apply_L!(out, L_oo, L_og, C_omega, nothing, r_gamma, tmp_gamma, tmp_omega, u)
    @test isapprox(out, L_oo * u; atol=1e-12, rtol=1e-12)
end
