if Base.find_package("SciMLBase") === nothing || Base.find_package("OrdinaryDiffEq") === nothing
    @testset "SciML integration (guarded)" begin
        @test true
    end
else
    using SciMLBase
    using OrdinaryDiffEq

    supports_sciml = let ok = true
        try
            sys_probe = build_test_system()
            u0_probe = zeros(Float64, length(sys_probe.dof_omega.indices))
            PenguinSolverCore.sciml_odeproblem(sys_probe, u0_probe, (0.0, 0.1); p=nothing)
        catch err
            if err isa ArgumentError && occursin("requires SciMLBase", sprint(showerror, err))
                ok = false
            else
                rethrow(err)
            end
        end
        ok
    end

    if !supports_sciml
        @testset "SciML integration (guarded)" begin
            @test true
        end
    else
        @testset "SciML integration (guarded)" begin
            tstar = 0.37

            sys = build_test_system()
            n_gamma = length(sys.r_gamma)
            u0 = zeros(Float64, length(sys.dof_omega.indices))

            gupd = PenguinDiffusion.RobinGUpdater((sys, u, p, t) -> fill(0.75, n_gamma))
            PenguinSolverCore.add_update!(sys, PenguinSolverCore.AtTimes([tstar]), gupd)

            prob = PenguinSolverCore.sciml_odeproblem(sys, u0, (0.0, 1.0); p=nothing)
            alg = OrdinaryDiffEq.Rosenbrock23(autodiff=false)
            sol = SciMLBase.solve(prob, alg;
                reltol=1e-9, abstol=1e-9, save_everystep=true,
            )

            @test any(t -> isapprox(t, tstar; atol=1000eps(tstar)), sol.t)
            @test sys.rebuild_calls == 0
            expected = 0.75 .* sys.ops.Iγ[sys.dof_gamma.indices]
            @test isapprox(norm(sys.r_gamma .- expected), 0.0; atol=0.0, rtol=0.0)

            sys_ref = build_test_system()
            prob_ref = PenguinSolverCore.sciml_odeproblem(sys_ref, u0, (0.0, 1.0); p=nothing)
            sol_ref = SciMLBase.solve(prob_ref, alg;
                reltol=1e-9, abstol=1e-9,
            )

            @test norm(sol.u[end] - sol_ref.u[end]) > 1e-9
        end
    end
end
