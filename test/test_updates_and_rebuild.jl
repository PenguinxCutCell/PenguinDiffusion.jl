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
    @test length(sys.dirichlet_affine) == length(sys.dof_omega.indices)
    @test maximum(abs, sys.dirichlet_affine; init=0.0) == 0.0
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
        expected = 0.5 .* sys.ops.Iγ[sys.dof_gamma.indices]
        @test isapprox(norm(sys.r_gamma .- expected), 0.0; atol=0.0, rtol=0.0)
    end

    @testset "RobinABUpdater refreshes constraint blocks (matrix rebuild)" begin
        sys = build_test_system()
        u = randn(length(sys.dof_omega.indices))

        y_before = zeros(Float64, length(u))
        PenguinDiffusion.apply_L!(y_before, sys, u)
        Cw_before = copy(sys.C_omega)
        Cg_before = copy(sys.C_gamma)

        n_gamma = length(sys.r_gamma)
        @test n_gamma > 0
        abupd = PenguinDiffusion.RobinABUpdater((sys, u, p, t) -> (a=2.0, b=1.0, g=fill(0.3, n_gamma)))
        PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.5]), abupd)

        PenguinSolverCore.apply_scheduled_updates!(sys, u, nothing, 0.5; step=0)

        @test sys.rebuild_calls == 1
        @test sys.constraints_dirty == false
        @test !isapprox(norm(Matrix(sys.C_omega - Cw_before)), 0.0; atol=1e-12, rtol=0.0)
        @test !isapprox(norm(Matrix(sys.C_gamma - Cg_before)), 0.0; atol=1e-12, rtol=0.0)

        y_after = zeros(Float64, length(u))
        PenguinDiffusion.apply_L!(y_after, sys, u)
        @test norm(y_after - y_before) > 1e-10
    end
end

@testset "Dirichlet affine reduced contract" begin
    sys, _, _ = build_dirichlet_test_system(; source=nothing, ulo=1.25, uhi=2.75)
    n_omega = length(sys.dof_omega.indices)
    u = randn(n_omega)

    lhs_reduced = zeros(Float64, n_omega)
    PenguinDiffusion.apply_L!(lhs_reduced, sys, u)
    lhs_reduced .+= sys.dirichlet_affine

    x_omega_full, x_gamma_full = PenguinDiffusion.full_state(sys, u)
    full_eval = CartesianOperators.laplacian_matrix(sys.ops, x_omega_full, x_gamma_full)
    ref_reduced = full_eval[sys.dof_omega.indices]

    @test isapprox(lhs_reduced, ref_reduced; atol=1e-11, rtol=1e-11)
end

@testset "BoxDirichletUpdater updates affine RHS without rebuild" begin
    sys, _, _ = build_dirichlet_test_system(; source=nothing, ulo=1.0, uhi=1.0)
    u = zeros(Float64, length(sys.dof_omega.indices))

    old_aff = copy(sys.dirichlet_affine)
    old_vals = copy(sys.dirichlet_values)
    upd = PenguinDiffusion.BoxDirichletUpdater((sys, u, p, t) -> (lo=(4.0, nothing), hi=(-1.0, nothing)))
    PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([0.25]), upd)
    PenguinSolverCore.apply_scheduled_updates!(sys, u, nothing, 0.25; step=0)

    @test sys.rebuild_calls == 0
    @test any(abs.(sys.dirichlet_affine .- old_aff) .> 1e-12)
    @test any(abs.(sys.dirichlet_values .- old_vals) .> 0.0)

    du = zeros(Float64, length(u))
    PenguinSolverCore.rhs!(du, sys, u, nothing, 0.25)
    @test norm(du) > 1e-12
end
