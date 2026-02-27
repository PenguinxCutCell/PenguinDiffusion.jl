@testset "Unsteady matrix-free RHS contract" begin
    sys_a = build_test_system(; kappa=1.7, source=(sys, u, p, t) -> 0.2 + 0.1 * t, matrixfree_unsteady=false)
    sys_mf = build_test_system(; kappa=1.7, source=(sys, u, p, t) -> 0.2 + 0.1 * t, matrixfree_unsteady=true)

    u = randn(length(sys_a.dof_omega.indices))
    du_a = zeros(Float64, length(u))
    du_mf = zeros(Float64, length(u))

    PenguinSolverCore.rhs!(du_a, sys_a, u, nothing, 0.37)
    PenguinSolverCore.rhs!(du_mf, sys_mf, u, nothing, 0.37)
    @test isapprox(du_mf, du_a; atol=1e-11, rtol=1e-11)
end

@testset "Unsteady matrix-free RHS contract (Dirichlet)" begin
    sys_a, _, _ = build_dirichlet_test_system(; kappa=1.3, source=0.15, ulo=1.2, uhi=2.1, matrixfree_unsteady=false)
    sys_mf, _, _ = build_dirichlet_test_system(; kappa=1.3, source=0.15, ulo=1.2, uhi=2.1, matrixfree_unsteady=true)

    u = randn(length(sys_a.dof_omega.indices))
    du_a = zeros(Float64, length(u))
    du_mf = zeros(Float64, length(u))

    PenguinSolverCore.rhs!(du_a, sys_a, u, nothing, 0.0)
    PenguinSolverCore.rhs!(du_mf, sys_mf, u, nothing, 0.0)
    @test isapprox(du_mf, du_a; atol=1e-11, rtol=1e-11)
end

if Base.find_package("SciMLBase") !== nothing && Base.find_package("OrdinaryDiffEq") !== nothing
    using SciMLBase
    using OrdinaryDiffEq

    @testset "Unsteady matrix-free SciML integration (guarded)" begin
        sys_a = build_test_system(; kappa=1.2, source=0.0, matrixfree_unsteady=false)
        sys_mf = build_test_system(; kappa=1.2, source=0.0, matrixfree_unsteady=true)
        u0 = zeros(Float64, length(sys_a.dof_omega.indices))
        tspan = (0.0, 0.35)

        prob_a = PenguinSolverCore.sciml_odeproblem(sys_a, u0, tspan; p=nothing)
        prob_mf = PenguinSolverCore.sciml_odeproblem(sys_mf, u0, tspan; p=nothing)
        alg = OrdinaryDiffEq.Rosenbrock23(autodiff=false)

        sol_a = SciMLBase.solve(prob_a, alg; reltol=1e-9, abstol=1e-9)
        sol_mf = SciMLBase.solve(prob_mf, alg; reltol=1e-9, abstol=1e-9)
        @test SciMLBase.successful_retcode(sol_a)
        @test SciMLBase.successful_retcode(sol_mf)
        @test isapprox(sol_mf.u[end], sol_a.u[end]; atol=5e-10, rtol=5e-10)
    end
end
