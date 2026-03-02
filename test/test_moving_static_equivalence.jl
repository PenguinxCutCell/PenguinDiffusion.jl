using Test

@testset "Moving slab solver matches static solver for static geometry" begin
    x = collect(range(0.0, 1.0; length=9))
    y = collect(range(0.0, 1.0; length=8))
    phi_full(_x, _y, _t=0.0) = -1.0

    moments = CartesianGeometry.geometric_moments(phi_full, (x, y), Float64, zero; method=:implicitintegration)
    bc = CartesianOperators.BoxBC(Val(2), Float64)
    Nd = prod(length.(moments.xyz))
    robin = CartesianOperators.RobinConstraint(ones(Float64, Nd), zeros(Float64, Nd), zeros(Float64, Nd))
    prob = PenguinDiffusion.DiffusionProblem(1.2, bc, robin, 0.0)

    sys_static = PenguinDiffusion.build_system(moments, prob)
    sys_moving = PenguinDiffusion.build_moving_system(phi_full, (x, y), prob; t0=0.0, t1=0.05)

    dims = ntuple(d -> length(moments.xyz[d]), 2)
    li = LinearIndices(dims)
    u0_full = zeros(Float64, Nd)
    u0_static = u0_full[sys_static.dof_omega.indices]

    tspan = (0.0, 0.15)
    dt = 0.05
    sol_static = PenguinDiffusion.unsteady_block_solve(
        sys_static,
        u0_static,
        tspan;
        dt=dt,
        scheme=:BE,
        save_everystep=true,
    )
    sol_moving = PenguinDiffusion.moving_unsteady_block_solve(
        sys_moving,
        u0_full,
        tspan;
        dt=dt,
        scheme=:BE,
        save_everystep=true,
    )

    @test length(sol_static.t) == length(sol_moving.t)
    @test all(i -> isapprox(sol_static.t[i], sol_moving.t[i]; atol=1e-12, rtol=0.0), eachindex(sol_static.t))

    idx = sys_static.dof_omega.indices
    for k in eachindex(sol_static.t)
        ustat_full, _ = PenguinDiffusion.full_state(sys_static, sol_static.omega[k])
        umov_full = zeros(Float64, Nd)
        umov_full[sys_moving.omega_idx] .= sol_moving.omega[k]
        @test isapprox(umov_full[idx], ustat_full[idx]; atol=1e-10, rtol=1e-9)
    end
end
