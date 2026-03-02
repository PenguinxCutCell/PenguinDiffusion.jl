using Test
using LinearAlgebra
using SparseArrays

using CartesianGeometry: geometric_moments, nan
using CartesianOperators
using PenguinBCs
using PenguinDiffusion
using PenguinSolverCore

full_moments(grid) = geometric_moments((args...) -> -1.0, grid, Float64, nan; method=:vofijul)

function physical_indices(nnodes::NTuple{N,Int}) where {N}
    LI = LinearIndices(nnodes)
    idx = Int[]
    for I in CartesianIndices(nnodes)
        if all(d -> I[d] < nnodes[d], 1:N)
            push!(idx, LI[I])
        end
    end
    return idx
end

function solve_mono(grid, bc_border::BorderConditions, source)
    moms = full_moments(grid)
    cap = assembled_capacity(moms; bc=0.0)
    pflags = periodic_flags(bc_border, length(grid))
    ops = DiffusionOps(cap; periodic=pflags)
    model = DiffusionModelMono(cap, ops, 1.0; source=source, bc_border=bc_border)
    sys = solve_steady!(model)
    return cap, model, sys
end

function l2_error(cap, u, u_exact)
    phys = physical_indices(cap.nnodes)
    e2 = 0.0
    for i in phys
        x = cap.C_ω[i]
        ue = u_exact(x...)
        e2 += (u[i] - ue)^2
    end
    return sqrt(e2 / length(phys))
end

@testset "Dirichlet manufactured convergence 1D" begin
    u_exact(x) = sin(pi * x)
    f(x, t) = pi^2 * sin(pi * x)

    errs = Float64[]
    hs = Float64[]
    for n in (17, 33)
        grid = (range(0.0, 1.0; length=n),)
        bc = BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0))
        cap, model, sys = solve_mono(grid, bc, f)
        u = sys.x[model.layout.offsets.ω]
        push!(errs, l2_error(cap, u, x -> u_exact(x)))
        push!(hs, step(grid[1]))
    end

    order = log(errs[1] / errs[2]) / log(hs[1] / hs[2])
    @test order > 1.5
end

@testset "Dirichlet manufactured convergence 2D" begin
    u_exact(x, y) = sin(pi * x) * sin(pi * y)
    f(x, y, t) = 2pi^2 * sin(pi * x) * sin(pi * y)

    errs = Float64[]
    hs = Float64[]
    for n in (9, 17)
        grid = (range(0.0, 1.0; length=n), range(0.0, 1.0; length=n))
        bc = BorderConditions(
            ; left=Dirichlet(0.0), right=Dirichlet(0.0),
            bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        )
        cap, model, sys = solve_mono(grid, bc, f)
        u = sys.x[model.layout.offsets.ω]
        push!(errs, l2_error(cap, u, (x, y) -> u_exact(x, y)))
        push!(hs, step(grid[1]))
    end

    order = log(errs[1] / errs[2]) / log(hs[1] / hs[2])
    @test order > 1.5
end

@testset "Neumann sign test 1D" begin
    grid = (range(0.0, 1.0; length=21),)
    moms = full_moments(grid)
    cap = assembled_capacity(moms; bc=0.0)
    bc = BorderConditions(; left=Neumann(-1.0), right=Neumann(1.0))
    ops = DiffusionOps(cap; periodic=periodic_flags(bc, 1))
    model = DiffusionModelMono(cap, ops, 1.0; source=(x, t) -> 0.0, bc_border=bc)

    nsys = last(model.layout.offsets.γ)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady_mono!(sys, model, 0.0)

    # Gauge fix for pure Neumann: pin first physical cell to exact linear profile.
    x0 = cap.C_ω[1][1]
    for j in 1:size(sys.A, 2)
        sys.A[1, j] = 0.0
    end
    sys.A[1, 1] = 1.0
    sys.b[1] = x0
    solve!(sys; method=:direct)

    u = sys.x[model.layout.offsets.ω]
    phys = physical_indices(cap.nnodes)
    maxerr = maximum(abs.(u[phys] .- map(i -> cap.C_ω[i][1], phys)))
    @test maxerr < 2e-2
end

@testset "Halo invariants for BC applier" begin
    grid = (range(0.0, 1.0; length=9), range(0.0, 1.0; length=7))
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    bc = BorderConditions(
        ; left=Dirichlet(1.0), right=Neumann(2.0),
        bottom=Neumann(0.0), top=Dirichlet(0.0),
    )
    ops = DiffusionOps(cap; periodic=periodic_flags(bc, 2))
    layout = layout_mono(cap.ntotal)
    A = spdiagm(0 => ones(Float64, 2 * cap.ntotal))
    b = zeros(Float64, 2 * cap.ntotal)

    apply_box_bc_mono!(A, b, cap, ops, 1.0, bc; t=0.0, layout=layout)

    LI = LinearIndices(cap.nnodes)
    touched = Int[]
    for i in layout.offsets.ω
        if A[i, i] != 1.0 || b[i] != 0.0
            push!(touched, i - first(layout.offsets.ω) + 1)
        end
    end

    expected = Set{Int}()
    for side in (:left, :right, :bottom, :top)
        for I in each_boundary_cell(cap.nnodes, side)
            push!(expected, LI[I])
        end
    end

    for i in touched
        @test i in expected
        I = CartesianIndices(cap.nnodes)[i]
        @test all(d -> I[d] < cap.nnodes[d], 1:2)
    end
end

function assemble_mono_system(model; t=0.0)
    nsys = last(model.layout.offsets.γ)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady_mono!(sys, model, t)
    return sys
end

function active_physical_indices(cap)
    LI = LinearIndices(cap.nnodes)
    idx = Int[]
    for I in CartesianIndices(cap.nnodes)
        i = LI[I]
        if all(d -> I[d] < cap.nnodes[d], 1:length(cap.nnodes)) && isfinite(cap.buf.V[i]) && cap.buf.V[i] > 0.0
            push!(idx, i)
        end
    end
    return idx
end

function weighted_energy(cap, uω, idx)
    e = 0.0
    for i in idx
        e += cap.buf.V[i] * uω[i]^2
    end
    return e
end

@testset "Node-lattice halo slab convention" begin
    cases = (
        (range(0.0, 1.0; length=7),),
        (range(0.0, 1.0; length=5), range(0.0, 1.0; length=4)),
    )

    bcfill = -3.25
    for grid in cases
        moms = geometric_moments((args...) -> -1.0, grid, Float64, nan; method=:vofijul)
        cap = assembled_capacity(moms; bc=bcfill)
        @test cap.ntotal == prod(cap.nnodes)

        LI = LinearIndices(cap.nnodes)
        N = length(grid)
        for I in CartesianIndices(cap.nnodes)
            i = LI[I]
            halo = any(d -> I[d] == cap.nnodes[d], 1:N)
            if halo
                @test cap.buf.V[i] == bcfill
                @test cap.buf.Γ[i] == bcfill
                @test cap.V[i, i] == bcfill
                @test cap.Γ[i, i] == bcfill
                for d in 1:N
                    @test cap.buf.A[d][i] == bcfill
                    @test cap.buf.W[d][i] == bcfill
                    @test cap.A[d][i, i] == bcfill
                    @test cap.W[d][i, i] == bcfill
                end
            else
                @test isequal(cap.buf.V[i], moms.V[i])
                @test isequal(cap.buf.Γ[i], moms.interface_measure[i])
                for d in 1:N
                    @test isequal(cap.buf.A[d][i], moms.A[d][i])
                    @test isequal(cap.buf.W[d][i], moms.W[d][i])
                end
            end
        end
    end
end

@testset "Full-domain operator structure" begin
    cases = (
        (range(0.0, 1.0; length=11),),
        (range(0.0, 1.0; length=7), range(0.0, 1.0; length=6)),
    )

    for grid in cases
        cap = assembled_capacity(full_moments(grid); bc=0.0)
        N = length(grid)
        ops = DiffusionOps(cap; periodic=ntuple(_ -> false, N))

        @test size(ops.G) == (N * cap.ntotal, cap.ntotal)
        @test size(ops.Winv) == (N * cap.ntotal, N * cap.ntotal)

        L = transpose(ops.G) * ops.Winv * ops.G
        denom = max(norm(L), eps())
        @test norm(L - transpose(L)) / denom < 1e-12

        @test maximum(abs.(diag(cap.Γ))) == 0.0
        hmax = isempty(ops.H.nzval) ? 0.0 : maximum(abs.(ops.H.nzval))
        @test hmax < 1e-12
    end
end

@testset "Default Neumann(0) equivalence" begin
    grid = (range(0.0, 1.0; length=9), range(0.0, 1.0; length=8))
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    src(x, y, t) = x - y + 1.0

    bc_default = BorderConditions()
    model_default = DiffusionModelMono(
        cap,
        DiffusionOps(cap; periodic=periodic_flags(bc_default, 2)),
        1.0;
        source=src,
        bc_border=bc_default,
    )
    sys_default = assemble_mono_system(model_default; t=0.4)

    bc_explicit = BorderConditions(
        ; left=Neumann(0.0), right=Neumann(0.0),
        bottom=Neumann(0.0), top=Neumann(0.0),
    )
    model_explicit = DiffusionModelMono(
        cap,
        DiffusionOps(cap; periodic=periodic_flags(bc_explicit, 2)),
        1.0;
        source=src,
        bc_border=bc_explicit,
    )
    sys_explicit = assemble_mono_system(model_explicit; t=0.4)

    @test sys_default.A == sys_explicit.A
    @test sys_default.b == sys_explicit.b
end

@testset "Periodic handling sanity" begin
    grid = (range(0.0, 1.0; length=17),)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    pbc = BorderConditions(; left=Periodic(), right=Periodic())
    pflags = periodic_flags(pbc, 1)
    ops = DiffusionOps(cap; periodic=pflags)

    _, _, _, _, Dm, _, Sm, _ = build_GHW(cap; periodic=pflags)
    onesv = ones(Float64, cap.ntotal)
    phys = physical_indices(cap.nnodes)

    @test maximum(abs.((Dm[1] * onesv)[phys])) < 1e-12
    @test maximum(abs.((Sm[1] * onesv .- onesv)[phys])) < 1e-12

    layout = layout_mono(cap.ntotal)
    A = spdiagm(0 => ones(Float64, 2 * cap.ntotal))
    b = zeros(Float64, 2 * cap.ntotal)
    A0 = copy(A)
    b0 = copy(b)
    apply_box_bc_mono!(A, b, cap, ops, 1.0, pbc; t=0.0, layout=layout)
    @test A == A0
    @test b == b0
end

@testset "Space/time variable coefficient manufactured 1D" begin
    u_exact(x) = sin(pi * x)
    D(x, t) = 1.0 + t + 0.25 * cos(2pi * x)
    f(x, t) = D(x, t) * pi^2 * sin(pi * x)
    bc = BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0))

    errs = Float64[]
    hs = Float64[]
    for n in (33, 65)
        grid = (range(0.0, 1.0; length=n),)
        cap = assembled_capacity(full_moments(grid); bc=0.0)
        ops = DiffusionOps(cap; periodic=periodic_flags(bc, 1))
        model = DiffusionModelMono(cap, ops, D; source=f, bc_border=bc)
        sys = solve_steady!(model; t=0.3)
        push!(errs, l2_error(cap, sys.x[model.layout.offsets.ω], x -> u_exact(x)))
        push!(hs, step(grid[1]))

        if n == 33
            sys_t0 = assemble_mono_system(model; t=0.0)
            sys_t1 = assemble_mono_system(model; t=0.7)
            @test norm(sys_t0.A - sys_t1.A) > 1e-6
        end
    end

    order = log(errs[1] / errs[2]) / log(hs[1] / hs[2])
    @test order > 1.5
end

@testset "Non-homogeneous Dirichlet profile 1D" begin
    grid = (range(0.0, 1.0; length=41),)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    bc = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(2.0))
    ops = DiffusionOps(cap; periodic=periodic_flags(bc, 1))
    model = DiffusionModelMono(cap, ops, 1.0; source=(x, t) -> 0.0, bc_border=bc)
    sys = solve_steady!(model)
    u = sys.x[model.layout.offsets.ω]
    phys = physical_indices(cap.nnodes)
    maxerr = maximum(abs.(u[phys] .- map(i -> 1.0 + cap.C_ω[i][1], phys)))
    @test maxerr < 2e-2
end

@testset "Embedded interface: no-interface reduction to mono" begin
    grid = (range(0.0, 1.0; length=21),)
    cap = assembled_capacity(full_moments(grid); bc=0.0)
    bc = BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0))
    ops = DiffusionOps(cap; periodic=periodic_flags(bc, 1))
    source(x, t) = pi^2 * sin(pi * x)

    model_plain = DiffusionModelMono(cap, ops, 1.0; source=source, bc_border=bc)
    ic = PenguinBCs.Robin(0.0, 0.0, 0.0)
    model_embed = DiffusionModelMono(cap, ops, 1.0; source=source, bc_border=bc, bc_interface=ic)

    sys_plain = assemble_mono_system(model_plain; t=0.0)
    sys_embed = assemble_mono_system(model_embed; t=0.0)
    ω = model_plain.layout.offsets.ω
    γ = model_plain.layout.offsets.γ

    @test norm(sys_plain.A[ω, ω] - sys_embed.A[ω, ω]) < 1e-12
    @test norm(sys_plain.b[ω] - sys_embed.b[ω]) < 1e-12
    @test norm(sys_embed.A[ω, γ]) < 1e-12
    @test norm(sys_embed.A[γ, ω]) < 1e-12
    @test norm(sys_embed.b[γ]) < 1e-12

    solve!(sys_plain; method=:direct)
    solve!(sys_embed; method=:direct)
    @test maximum(abs.(sys_plain.x[ω] .- sys_embed.x[ω])) < 1e-12

    @test_throws TypeError DiffusionModelMono(
        cap,
        ops,
        1.0;
        source=source,
        bc_border=bc,
        bc_interface=InterfaceConditions(; scalar=ScalarJump(1.0, 1.0, 0.0)),
    )
end

@testset "Embedded interface cut-cell consistency 1D (mono Robin)" begin
    xγ = 0.53
    grid = (range(0.0, 1.0; length=41),)
    moms = geometric_moments((x) -> x - xγ, grid, Float64, nan; method=:vofijul)
    cap = assembled_capacity(moms; bc=0.0)
    bc = BorderConditions(; left=Dirichlet(1.0), right=Dirichlet(1.0))
    ops = DiffusionOps(cap; periodic=periodic_flags(bc, 1))
    ic = PenguinBCs.Robin(1.0, 0.0, 1.0)
    model = DiffusionModelMono(cap, ops, 1.0; source=(x, t) -> 0.0, bc_border=bc, bc_interface=ic)
    sys = solve_steady!(model)

    ω = sys.x[model.layout.offsets.ω]
    γ = sys.x[model.layout.offsets.γ]
    idxω = active_physical_indices(cap)
    idxγ = findall(i -> isfinite(cap.buf.Γ[i]) && cap.buf.Γ[i] > 0.0, 1:cap.ntotal)

    @test !isempty(idxγ)
    @test maximum(abs.(ω[idxω] .- 1.0)) < 2e-2
    @test maximum(abs.(γ[idxγ] .- 1.0)) < 1e-12
end

@testset "Unsteady theta schemes: BE/CN and partial dt regression" begin
    u_exact(x, t) = exp(-pi^2 * t) * sin(pi * x)
    bc = BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0))

    function run_theta(dt, θ; n=65, Tend=0.1, init=u_exact)
        grid = (range(0.0, 1.0; length=n),)
        cap = assembled_capacity(full_moments(grid); bc=0.0)
        ops = DiffusionOps(cap; periodic=periodic_flags(bc, 1))
        model = DiffusionModelMono(cap, ops, 1.0; source=(x, t) -> 0.0, bc_border=bc)

        lay = model.layout.offsets
        nsys = last(lay.γ)
        sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
        u = zeros(Float64, nsys)
        idx = physical_indices(cap.nnodes)
        for i in idx
            u[lay.ω[i]] = init(cap.C_ω[i][1], 0.0)
        end

        dts = Float64[]
        energies = Float64[weighted_energy(cap, u[lay.ω], idx)]
        t = 0.0
        while t < Tend - 1e-12
            dt_step = min(dt, Tend - t)
            push!(dts, dt_step)
            assemble_unsteady_mono!(sys, model, u, t, dt_step, θ)
            solve!(sys; method=:direct, reuse_factorization=false)
            u .= sys.x
            t += dt_step
            push!(energies, weighted_energy(cap, u[lay.ω], idx))
        end

        return cap, model, u[lay.ω], dts, energies
    end

    cap1, _, u1, _, _ = run_theta(0.02, 1.0)
    cap2, _, u2, _, _ = run_theta(0.01, 1.0)
    idx1 = physical_indices(cap1.nnodes)
    idx2 = physical_indices(cap2.nnodes)
    err1 = sqrt(sum((u1[i] - u_exact(cap1.C_ω[i][1], 0.1))^2 for i in idx1) / length(idx1))
    err2 = sqrt(sum((u2[i] - u_exact(cap2.C_ω[i][1], 0.1))^2 for i in idx2) / length(idx2))
    order = log(err1 / err2) / log(2)
    @test order > 0.8

    cap3, _, u3, _, _ = run_theta(0.02, 0.5; n=129)
    cap4, _, u4, _, _ = run_theta(0.01, 0.5; n=129)
    idx3 = physical_indices(cap3.nnodes)
    idx4 = physical_indices(cap4.nnodes)
    err3 = sqrt(sum((u3[i] - u_exact(cap3.C_ω[i][1], 0.1))^2 for i in idx3) / length(idx3))
    err4 = sqrt(sum((u4[i] - u_exact(cap4.C_ω[i][1], 0.1))^2 for i in idx4) / length(idx4))
    order_cn = log(err3 / err4) / log(2)
    @test order_cn > 1.5

    _, _, _, dts, energies = run_theta(0.03, 1.0; n=33, init=(x, t) -> sin(pi * x) + 0.2 * sin(3pi * x))
    @test length(dts) == 4
    @test dts[end] < 0.03
    @test isapprox(sum(dts), 0.1; atol=1e-12)
    @test all(diff(energies) .<= 1e-10)
end

@testset "Unsteady diphasic cut-cell solve robustness" begin
    xγ = 0.53
    grid = (range(0.0, 1.0; length=41),)
    moms = geometric_moments((x) -> x - xγ, grid, Float64, nan; method=:vofijul)
    cap = assembled_capacity(moms; bc=0.0)
    bc = BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0))
    ops = DiffusionOps(cap; periodic=periodic_flags(bc, 1))
    ic = InterfaceConditions(; scalar=ScalarJump(1.0, 1.0, 0.0), flux=FluxJump(1.0, 1.0, 0.0))
    model = DiffusionModelDiph(cap, ops, 1.0, 1.0; source=(x, t) -> (0.0, 0.0), bc_border=bc, bc_interface=ic)

    lay = model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    u0 = zeros(Float64, nsys)

    assemble_unsteady_diph!(sys, model, u0, 0.0, 0.01, 1.0)
    solve!(sys; method=:direct)
    @test all(isfinite, sys.x)
end
