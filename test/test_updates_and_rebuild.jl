@testset "Active-cell masking" begin
    moments = build_cut_moments()
    bc = CartesianOperators.BoxBC(Val(2), Float64)
    ops = CartesianOperators.assembled_ops(moments; bc=bc)

    interface = CartesianOperators.RobinConstraint(ones(Float64, ops.Nd), zeros(Float64, ops.Nd), zeros(Float64, ops.Nd))
    prob = PenguinDiffusion.DiffusionProblem(1.0, bc, interface, nothing)
    sys = PenguinDiffusion.build_system(moments, prob)

    V = Float64.(moments.V)
    vtol = sqrt(eps(Float64)) * maximum(abs, V; init=0.0)
    material_mask = (moments.cell_type .!= 0) .& (V .> vtol)
    pad_mask = PenguinDiffusion.padded_mask(ops.dims)
    expected_omega = findall(material_mask .& .!pad_mask)

    ig_tol = sqrt(eps(Float64)) * maximum(abs, ops.Iγ; init=0.0)
    expected_gamma = findall((ops.Iγ .> ig_tol) .& material_mask .& .!pad_mask)

    @test sys.dof_omega.indices == expected_omega
    @test sys.dof_gamma.indices == expected_gamma
    @test all(diag(sys.M) .> 0.0)
    !isempty(sys.dof_gamma.indices) && @test sys.C_gamma_fact !== nothing
end

@testset "Update + rebuild contract" begin
    @testset "KappaUpdater triggers one rebuild" begin
        sys = build_test_system()
        u = zeros(Float64, length(sys.dof_omega.indices))

        kupd = PenguinDiffusion.KappaUpdater((sys, u, p, t) -> 2.5)
        PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.5]), kupd)

        PenguinSolverCore.apply_scheduled_updates!(sys, u, nothing, 0.5; step=0)

        @test sys.rebuild_calls == 1
        @test sys.kappa == 2.5
    end

    @testset "RobinGUpdater is rhs_only (no rebuild)" begin
        sys = build_test_system()
        u = zeros(Float64, length(sys.dof_omega.indices))
        n_gamma = length(sys.r_gamma)
        @test n_gamma > 0

        gupd = PenguinDiffusion.RobinGUpdater((sys, u, p, t) -> fill(t, n_gamma))
        PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.5]), gupd)

        PenguinSolverCore.apply_scheduled_updates!(sys, u, nothing, 0.5; step=0)

        @test sys.rebuild_calls == 0
        @test isapprox(norm(sys.r_gamma .- 0.5), 0.0; atol=0.0, rtol=0.0)
    end
end
