# --- Manufactured steady manufactured test ---
@testset "manufactured_full_domain_steady" begin
    using LinearSolve
    using SciMLBase

    function manufactured_full_domain_steady()
        # Full-domain geometry.
        nx, ny = 32, 28
        x = collect(range(0.0, 1.0; length=nx + 1))
        y = collect(range(0.0, 1.0; length=ny + 1))
        full_domain(_x, _y, _t=0.0) = -1.0
        moments = geometric_moments(full_domain, (x, y), Float64, zero; method=:implicitintegration)

        # Manufactured exact field (spatially varying).
        dims = ntuple(d -> length(moments.xyz[d]), 2)
        li = LinearIndices(dims)
        Nd = prod(dims)
        u_exact_full = zeros(Float64, Nd)
        @inbounds for I in CartesianIndices(dims)
            idx = li[I]
            xx = moments.xyz[1][I[1]]
            yy = moments.xyz[2][I[2]]
            u_exact_full[idx] = 2.0 + 0.4 * xx - 0.3 * yy + sin(pi * xx) * cos(2pi * yy)
        end

        # Spatially varying Dirichlet BC from the manufactured field.
        bc = BoxBC(
            (Dirichlet(copy(u_exact_full)), Dirichlet(copy(u_exact_full))),
            (Dirichlet(copy(u_exact_full)), Dirichlet(copy(u_exact_full))),
        )

        # Interface is unused for full domain but still required by v0 system builder.
        ops = assembled_ops(moments; bc=bc)
        interface = RobinConstraint(ones(Float64, ops.Nd), zeros(Float64, ops.Nd), zeros(Float64, ops.Nd))

        kappa = 1.7
        # Placeholder source at build; replaced after discrete manufactured source is constructed.
        prob = DiffusionProblem(kappa, bc, interface, 0.0)
        sys = build_system(moments, prob)

        # Build discrete manufactured source so u_exact is exact for the reduced system.
        u_exact_reduced = u_exact_full[sys.dof_omega.indices]
        Lu = zeros(Float64, length(u_exact_reduced))
        PenguinDiffusion.apply_L!(Lu, sys, u_exact_reduced)
        Lu .+= sys.dirichlet_affine

        V_active = Float64.(moments.V[sys.dof_omega.indices])
        source_density_reduced = -(kappa .* Lu) ./ V_active
        sys.sourcefun = (_sys, _u, _p, _t) -> source_density_reduced

        # Steady solve with LinearSolve backend.
        sol = steady_solve(
            sys;
            alg=LinearSolve.SimpleGMRES(),
            reltol=1e-12,
            abstol=1e-12,
            maxiters=50_000,
        )
        SciMLBase.successful_retcode(sol) || error("steady solve failed with retcode=$(sol.retcode)")

        u_num_full, _ = full_state(sys, sol.u)
        err = u_num_full[sys.dof_omega.indices] .- u_exact_reduced

        rel = norm(err) / max(norm(u_exact_reduced), eps(Float64))
        maxabs = maximum(abs, err)
        return rel, maxabs
    end

    rel, maxabs = manufactured_full_domain_steady()
    println("Full-domain steady manufactured test: rel=%.6e, maxabs=%.6e\n", rel, maxabs)
    @test rel < 1e-8
end


# --- Manufactured unsteady manufactured test ---
@testset "manufactured_full_domain_unsteady" begin
    using OrdinaryDiffEq
    using SciMLBase
    using PenguinSolverCore

    function manufactured_full_domain_unsteady()
        # Full-domain geometry.
        nx, ny = 10, 10
        x = collect(range(0.0, 1.0; length=nx + 1))
        y = collect(range(0.0, 1.0; length=ny + 1))
        full_domain(_x, _y, _t=0.0) = -1.0
        moments = geometric_moments(full_domain, (x, y), Float64, zero; method=:implicitintegration)

        dims = ntuple(d -> length(moments.xyz[d]), 2)
        Nd = prod(dims)
        li = LinearIndices(dims)

        # Spatial basis fields for manufactured solution.
        base = zeros(Float64, Nd)
        psi = zeros(Float64, Nd)
        @inbounds for I in CartesianIndices(dims)
            idx = li[I]
            xx = moments.xyz[1][I[1]]
            yy = moments.xyz[2][I[2]]
            base[idx] = 1.0 + 0.2 * xx - 0.15 * yy + 0.1 * xx * yy
            psi[idx] = sin(2pi * xx) + 0.35 * cos(pi * yy)
        end

        amp(t) = 0.8 + 0.3 * sin(3.0 * t) + 0.2 * t
        damp(t) = 0.9 * cos(3.0 * t) + 0.2
        u_exact_full(t) = base .+ amp(t) .* psi

        # Spatial+time-varying Dirichlet BC; initialize with t=0 values.
        u0_full = u_exact_full(0.0)
        bc = BoxBC(
            (Dirichlet(copy(u0_full)), Dirichlet(copy(u0_full))),
            (Dirichlet(copy(u0_full)), Dirichlet(copy(u0_full))),
        )

        ops = assembled_ops(moments; bc=bc)
        interface = RobinConstraint(ones(Float64, ops.Nd), zeros(Float64, ops.Nd), zeros(Float64, ops.Nd))
        kappa = 1.3
        prob = DiffusionProblem(kappa, bc, interface, nothing)
        sys = build_system(moments, prob)

        idx_omega = sys.dof_omega.indices
        V_active = Float64.(moments.V[idx_omega])
        psi_reduced = psi[idx_omega]

        # Time-varying source manufactured to make u_exact(t) exact for the semi-discrete system.
        sys.sourcefun = function (_sys, _u, _p, t)
            u_ex_r = u_exact_full(t)[idx_omega]
            dudt_r = damp(t) .* psi_reduced

            Lu = zeros(Float64, length(idx_omega))
            PenguinDiffusion.apply_L!(Lu, sys, u_ex_r)
            Lu .+= sys.dirichlet_affine

            mass_dudt = V_active .* dudt_r
            src_mass = mass_dudt .- kappa .* Lu
            return src_mass ./ V_active
        end

        # Update Dirichlet vectors at each solver step.
        bupd = BoxDirichletUpdater((sys, _u, _p, t) -> begin
            uf = u_exact_full(t)
            (lo=(uf, uf), hi=(uf, uf))
        end)
        add_update!(sys, EveryStep(), bupd)

        u0 = u0_full[idx_omega]
        tspan = (0.0, 0.12)
        odeprob = sciml_odeproblem(sys, u0, tspan; p=nothing, include_every_step=true)
        sol = SciMLBase.solve(
            odeprob,
            OrdinaryDiffEq.Rosenbrock23(autodiff=false);
            reltol=1e-7,
            abstol=1e-7,
            saveat=range(tspan[1], tspan[2]; length=13),
        )
        SciMLBase.successful_retcode(sol) || error("time solve failed with retcode=$(sol.retcode)")

        u_num_full, _ = full_state(sys, sol.u[end])
        u_ex_final = u_exact_full(tspan[2])
        err = u_num_full[idx_omega] .- u_ex_final[idx_omega]
        rel = norm(err) / max(norm(u_ex_final[idx_omega]), eps(Float64))
        maxabs = maximum(abs, err)
        return rel, maxabs
    end

    rel, maxabs = manufactured_full_domain_unsteady()
    println("Full-domain unsteady manufactured test: rel=%.6e, maxabs=%.6e\n", rel, maxabs)
    @test rel < 1e-6
end
