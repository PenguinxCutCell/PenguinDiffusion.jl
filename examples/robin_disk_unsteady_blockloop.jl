using LinearAlgebra
using Printf
using SpecialFunctions
using Roots

using CartesianGeometry
using CartesianOperators
using PenguinDiffusion

"""
Unsteady monophasic diffusion in a disk with Robin interface, solved by
`unsteady_block_solve` (assembled block loop).

This example reports a proper volume-weighted relative L2 error:
- full vector length is used,
- outside cells have `V=0`, so they contribute zero,
- analytical values are sampled at `moments.barycenter`.

Convergence setup uses `dt = 0.2*h^2` to avoid temporal-error domination.
"""

function j0_zeros_robin(N::Int, h::Float64, R::Float64; guess_shift::Float64=0.25)
    eq(alpha) = alpha * besselj1(alpha) - h * R * besselj0(alpha)
    zs = zeros(Float64, N)
    @inbounds for m in 1:N
        x_left = max((m - guess_shift - 0.5) * pi, 1e-9)
        x_right = (m - guess_shift + 0.5) * pi
        zs[m] = find_zero(eq, (x_left, x_right))
    end
    return zs
end

function radial_heat_value(
    x::Float64,
    y::Float64,
    t::Float64;
    center::Tuple{Float64,Float64},
    R::Float64,
    h::Float64,
    α::Float64,
    Ti::Float64,
    T∞::Float64,
    alphas::Vector{Float64},
)
    r = sqrt((x - center[1])^2 + (y - center[2])^2)
    r >= R && return 0.0

    s = 0.0
    @inbounds for αm in alphas
        An = 2.0 * h * R / ((h^2 * R^2 + αm^2) * besselj0(αm))
        s += An * exp(-α * αm^2 * t / R^2) * besselj0(αm * (r / R))
    end
    return (1.0 - s) * (T∞ - Ti) + Ti
end

function analytic_field!(out::Vector{Float64}, moments, t::Float64; center, R, h, α, Ti, T∞, alphas)
    @inbounds for i in eachindex(out)
        if moments.V[i] > 0
            b = moments.barycenter[i]
            bx = b[1]
            by = b[2]
            out[i] = radial_heat_value(
                bx,
                by,
                t;
                center=center,
                R=R,
                h=h,
                α=α,
                Ti=Ti,
                T∞=T∞,
                alphas=alphas,
            )
        else
            out[i] = 0.0
        end
    end
    return out
end

function volume_rel_l2(moments, unum_full::Vector{Float64}, uref_full::Vector{Float64})
    V = Float64.(moments.V)
    num = sum(@. V * (unum_full - uref_full)^2)
    den = max(sum(@. V * (uref_full)^2), eps(Float64))
    return sqrt(num / den)
end

function run_case(nx::Int, ny::Int; scheme::Symbol=:CN, dt_factor::Float64=0.2, tf::Float64=0.1, nmodes::Int=120)
    lx = 1.0
    ly = 1.0
    radius = ly / 4
    center = (lx / 2, ly / 2)
    Ti = 270.0
    T∞ = 400.0
    α = 1.0
    robin_a = 3.0
    robin_b = 1.0

    x = collect(range(0.0, lx; length=nx + 1))
    y = collect(range(0.0, ly; length=ny + 1))
    hmesh = max(x[2] - x[1], y[2] - y[1])
    dt = dt_factor * hmesh^2

    circle = (x, y, _t=0.0) -> sqrt((x - center[1])^2 + (y - center[2])^2) - radius
    moments = geometric_moments(circle, (x, y), Float64, zero; method=:vofi)

    Nd = length(moments.V)
    bc = BoxBC(Val(2), Float64)
    interface = RobinConstraint(robin_a, robin_b, robin_a * T∞, Nd)
    prob = DiffusionProblem(α, bc, interface, 0.0)
    sys = build_system(moments, prob)

    u0_full = fill(Ti, Nd)
    u0 = u0_full[sys.dof_omega.indices]

    sol = unsteady_block_solve(
        sys,
        u0,
        (0.0, tf);
        dt=dt,
        scheme=scheme,
        save_everystep=false,
    )
    unum_full, _ = full_state(sys, sol.omega[end])

    alphas = j0_zeros_robin(nmodes, robin_a / robin_b, radius)
    uref_full = zeros(Float64, Nd)
    analytic_field!(
        uref_full,
        moments,
        tf;
        center=center,
        R=radius,
        h=robin_a / robin_b,
        α=α,
        Ti=Ti,
        T∞=T∞,
        alphas=alphas,
    )

    rel_l2 = volume_rel_l2(moments, unum_full, uref_full)

    active = sys.dof_omega.indices
    cidx = active[argmin([(moments.barycenter[i][1] - center[1])^2 + (moments.barycenter[i][2] - center[2])^2 for i in active])]
    u_center_num = unum_full[cidx]
    u_center_ref = uref_full[cidx]

    return hmesh, dt, rel_l2, u_center_num, u_center_ref
end

function run_convergence(scheme::Symbol)
    ns = (16, 24, 36, 48, 64, 96)
    hs = Float64[]
    errs = Float64[]

    @printf("\nUnsteady Robin disk with scheme=%s (dt = 0.2*h^2)\n", String(scheme))
    @printf("%6s  %12s  %12s  %14s  %12s\n", "n", "h", "dt", "relL2(full V)", "|center err|")

    for n in ns
        h, dt, rel, uc_num, uc_ref = run_case(n, n; scheme=scheme, dt_factor=0.2, tf=0.1, nmodes=120)
        push!(hs, h)
        push!(errs, rel)
        @printf("%6d  %12.5e  %12.5e  %14.6e  %12.5e\n", n, h, dt, rel, abs(uc_num - uc_ref))
    end

    @printf("Observed rates:\n")
    for i in 2:length(ns)
        rate = log(errs[i - 1] / errs[i]) / log(hs[i - 1] / hs[i])
        @printf("  n=%d -> n=%d : %.3f\n", ns[i - 1], ns[i], rate)
    end
end

function main()
    run_convergence(:BE)
    run_convergence(:CN)
end

main()
