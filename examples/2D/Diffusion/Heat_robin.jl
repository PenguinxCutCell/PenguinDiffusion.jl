using CartesianGeometry: geometric_moments, nan
using CartesianOperators
using PenguinBCs
using PenguinDiffusion
using Roots
using SpecialFunctions

function active_physical_indices(cap)
    LI = LinearIndices(cap.nnodes)
    idx = Int[]
    N = length(cap.nnodes)
    for I in CartesianIndices(cap.nnodes)
        i = LI[I]
        if all(d -> I[d] < cap.nnodes[d], 1:N)
            v = cap.buf.V[i]
            if isfinite(v) && v > 0.0
                push!(idx, i)
            end
        end
    end
    return idx
end

function volume_l2_error(cap, uomega, uexact, idx)
    num = 0.0
    den = 0.0
    for i in idx
        w = cap.buf.V[i]
        ue = uexact(cap.C_ω[i]...)
        d = uomega[i] - ue
        num += w * d^2
        den += w
    end
    return sqrt(num / den)
end

function robin_bessel_roots(N, kR)
    eq(a) = a * besselj1(a) - kR * besselj0(a)
    roots = zeros(Float64, N)
    for m in 1:N
        left = max((m - 0.75) * pi, 1e-6)
        right = (m - 0.25) * pi
        roots[m] = find_zero(eq, (left, right), Bisection())
    end
    return roots
end

function radial_heat_solution(x, y, t; center, radius, k, a, T0, Tinf, roots)
    r = sqrt((x - center[1])^2 + (y - center[2])^2)
    if r >= radius
        return NaN
    end
    s = 0.0
    for alpha in roots
        coeff = 2.0 * k * radius / ((k^2 * radius^2 + alpha^2) * besselj0(alpha))
        s += coeff * exp(-a * alpha^2 * t / radius^2) * besselj0(alpha * (r / radius))
    end
    return (1.0 - s) * (Tinf - T0) + T0
end

function main()
    # 2D unsteady heat in a disk:
    # dt u = Delta u in disk
    # alpha*u + beta*dn(u) = alpha*Tinf on embedded interface
    # with initially uniform T0.
    n = 49
    grid = (range(0.0, 4.0; length=n), range(0.0, 4.0; length=n))
    center = (2.0, 2.0)
    radius = 1.0
    body(x, y) = sqrt((x - center[1])^2 + (y - center[2])^2) - radius

    moms = geometric_moments(body, grid, Float64, nan; method=:vofijul)
    cap = assembled_capacity(moms; bc=0.0)

    T0 = 270.0
    Tinf = 400.0
    k = 3.0
    beta = 1.0
    diffusivity = 1.0
    t_end = 0.05
    dt = 0.25 * step(grid[1])^2

    bc_border = BorderConditions(
        ; left=Dirichlet(Tinf), right=Dirichlet(Tinf),
        bottom=Dirichlet(Tinf), top=Dirichlet(Tinf),
    )
    bc_interface = Robin(k, beta, k * Tinf)

    ops = DiffusionOps(cap; periodic=periodic_flags(bc_border, 2))
    model = DiffusionModelMono(cap, ops, diffusivity; source=0.0, bc_border=bc_border, bc_interface=bc_interface)

    lay = model.layout.offsets
    nsys = last(lay.γ)
    u0 = zeros(Float64, nsys)
    idx = active_physical_indices(cap)
    for i in idx
        u0[lay.ω[i]] = T0
    end

    sol = solve_unsteady!(
        model,
        u0,
        (0.0, t_end);
        dt=dt,
        scheme=:BE,
        method=:direct,
        save_history=false,
    )
    u = sol.system.x

    roots = robin_bessel_roots(80, k * radius)
    u_exact(x, y) = radial_heat_solution(
        x, y, t_end;
        center=center, radius=radius, k=k, a=diffusivity, T0=T0, Tinf=Tinf, roots=roots,
    )

    uomega = u[lay.ω]
    err_l2 = volume_l2_error(cap, uomega, u_exact, idx)
    i_center = idx[argmin(map(i -> (cap.C_ω[i][1] - center[1])^2 + (cap.C_ω[i][2] - center[2])^2, idx))]
    center_value = uomega[i_center]

    println("Heat Robin in disk (unsteady)")
    println("  dt: ", dt, ", final time: ", t_end)
    println("  reused constant operator: ", sol.reused_constant_operator)
    println("  active cells: ", length(idx))
    println("  center temperature (numerical): ", center_value)
    println("  volume-weighted L2 error at final time: ", err_l2)

    @assert err_l2 < 6.0
end

main()
