using LinearAlgebra
using Printf
using SpecialFunctions

using CartesianGeometry
using CartesianOperators
using PenguinDiffusion

"""
Diphasic unsteady heat benchmark following:
`BenchPhaseFlow.jl/problems/scalar/diphasic/Heat_2ph_2D.jl`

Uses PenguinDiffusion assembled block time loop:
`diphasic_unsteady_block_solve`.

Error metric convention:
- full-vector length is used (no active-index-only slicing),
- volume-weighted L2 norm, with V=0 naturally removing outside cells.
"""

struct Heat2Ph2DParams
    lx::Float64
    ly::Float64
    x0::Float64
    y0::Float64
    cx::Float64
    cy::Float64
    radius::Float64
    tend::Float64
    dg::Float64
    dl::Float64
    he::Float64
    cg0::Float64
    cl0::Float64
end

Heat2Ph2DParams(;
    lx=8.0,
    ly=8.0,
    x0=0.0,
    y0=0.0,
    cx=4.0,
    cy=4.0,
    radius=1.0,
    tend=0.1,
    dg=1.0,
    dl=1.0,
    he=1.0,
    cg0=1.0,
    cl0=0.0,
) = Heat2Ph2DParams(lx, ly, x0, y0, cx, cy, radius, tend, dg, dl, he, cg0, cl0)

function phi_val(u::Float64, p::Heat2Ph2DParams)
    d = sqrt(p.dg / p.dl)
    term1 = p.dg * sqrt(p.dl) * besselj1(u * p.radius) * bessely0(d * u * p.radius)
    term2 = p.he * p.dl * sqrt(p.dg) * besselj0(u * p.radius) * bessely1(d * u * p.radius)
    return term1 - term2
end

function psi_val(u::Float64, p::Heat2Ph2DParams)
    d = sqrt(p.dg / p.dl)
    term1 = p.dg * sqrt(p.dl) * besselj1(u * p.radius) * besselj0(d * u * p.radius)
    term2 = p.he * p.dl * sqrt(p.dg) * besselj0(u * p.radius) * besselj1(d * u * p.radius)
    return term1 - term2
end

function phase1_integrand(u::Float64, x::Float64, y::Float64, p::Heat2Ph2DParams)
    r = hypot(x - p.cx, y - p.cy)
    ph = phi_val(u, p)
    ps = psi_val(u, p)
    den = u^2 * (ph^2 + ps^2)
    num = exp(-p.dg * u^2 * p.tend) * besselj0(u * r) * besselj1(u * p.radius)
    if !isfinite(num) || !isfinite(den) || iszero(den)
        return 0.0
    end
    return num / den
end

function phase2_integrand(u::Float64, x::Float64, y::Float64, p::Heat2Ph2DParams)
    r = hypot(x - p.cx, y - p.cy)
    ph = phi_val(u, p)
    ps = psi_val(u, p)
    d = sqrt(p.dg / p.dl)
    den = u * (ph^2 + ps^2)
    contrib = besselj0(d * u * r) * ph - bessely0(d * u * r) * ps
    num = exp(-p.dg * u^2 * p.tend) * besselj1(u * p.radius) * contrib
    if !isfinite(num) || !isfinite(den) || iszero(den)
        return 0.0
    end
    return num / den
end

function phase1_exact_function(p::Heat2Ph2DParams; umax_factor::Float64=5.0)
    pref = (4.0 * p.cg0 * p.dg * p.dl^2 * p.he) / (pi^2 * p.radius)
    umin = 1e-8
    umax = umax_factor / sqrt(p.dg * p.tend)
    return (x, y) -> begin
        r = hypot(x - p.cx, y - p.cy)
        r >= p.radius && return 0.0
        val = simpson_integral(u -> phase1_integrand(u, x, y, p), umin, umax; n=6000)
        return pref * val
    end
end

function phase2_exact_function(p::Heat2Ph2DParams; umax_factor::Float64=5.0)
    pref = (2.0 * p.cg0 * p.dg * sqrt(p.dl) * p.he) / pi
    umin = 1e-8
    umax = umax_factor / sqrt(p.dg * p.tend)
    return (x, y) -> begin
        r = hypot(x - p.cx, y - p.cy)
        r < p.radius && return 0.0
        val = simpson_integral(u -> phase2_integrand(u, x, y, p), umin, umax; n=6000)
        return pref * val
    end
end

function simpson_integral(f, a::Float64, b::Float64; n::Int=4000)
    n >= 2 || throw(ArgumentError("simpson_integral requires n >= 2"))
    neven = iseven(n) ? n : (n + 1)
    h = (b - a) / neven
    s = f(a) + f(b)
    @inbounds for k in 1:(neven - 1)
        x = a + k * h
        s += (isodd(k) ? 4.0 : 2.0) * f(x)
    end
    return s * h / 3.0
end

function rel_l2_full_volume(u_num::Vector{Float64}, u_ref::Vector{Float64}, v::AbstractVector)
    vf = Float64.(v)
    num = sum(@. vf * (u_num - u_ref)^2)
    den = max(sum(@. vf * (u_ref)^2), eps(Float64))
    return sqrt(num / den)
end

function run_case(nx::Int, p::Heat2Ph2DParams; umax_factor::Float64=5.0)
    ny = nx
    x = collect(range(p.x0, p.x0 + p.lx; length=nx + 1))
    y = collect(range(p.y0, p.y0 + p.ly; length=ny + 1))

    circle = (x, y, _t=0.0) -> sqrt((x - p.cx)^2 + (y - p.cy)^2) - p.radius
    moments1 = geometric_moments(circle, (x, y), Float64, zero; method=:implicitintegration)
    moments2 = geometric_moments((x, y, t=0.0) -> -circle(x, y, t), (x, y), Float64, zero; method=:implicitintegration)

    nd = length(moments1.V)
    bc1 = BoxBC(Val(2), Float64)
    bc2 = BoxBC(Val(2), Float64)

    scalarjump = ScalarJumpConstraint(ones(Float64, nd), fill(1.0 / p.he, nd), zeros(Float64, nd))
    # Convention note:
    # PenguinDiffusion flux-jump rows use b1*q1 + b2*q2 = g with opposite normals,
    # so physical continuity Dg*∂nTg = Dl*∂nTl maps to (b1, b2) = (Dg, -Dl).
    fluxjump = FluxJumpConstraint(fill(p.dg, nd), fill(-p.dl, nd), zeros(Float64, nd))
    prob = TwoPhaseDiffusionProblem(p.dg, p.dl, bc1, bc2, fluxjump, scalarjump, 0.0, 0.0)
    sys = build_system(moments1, moments2, prob)

    n1 = length(sys.dof_omega1.indices)
    n2 = length(sys.dof_omega2.indices)
    u0 = vcat(fill(p.cg0, n1), fill(p.cl0, n2))

    dt = 0.5 * (p.lx / nx)^2
    sol = diphasic_unsteady_block_solve(
        sys,
        u0,
        (0.0, p.tend);
        dt=dt,
        scheme=:BE,
        save_everystep=false,
    )

    u_red = vcat(sol.omega1[end], sol.omega2[end])
    u1_num, _, u2_num, _ = full_state(sys, u_red)

    u1_ref_fun = phase1_exact_function(p; umax_factor=umax_factor)
    u2_ref_fun = phase2_exact_function(p; umax_factor=umax_factor)

    u1_ref = zeros(Float64, nd)
    u2_ref = zeros(Float64, nd)
    @inbounds for i in 1:nd
        b = moments1.barycenter[i]
        xb = b[1]
        yb = b[2]
        b2 = moments2.barycenter[i]
        xb2 = b2[1]
        yb2 = b2[2]
        u1_ref[i] = u1_ref_fun(xb, yb)
        u2_ref[i] = u2_ref_fun(xb2, yb2)
    end

    rel1 = rel_l2_full_volume(u1_num, u1_ref, moments1.V)
    rel2 = rel_l2_full_volume(u2_num, u2_ref, moments2.V)

    num_comb = sum(@. Float64(moments1.V) * (u1_num - u1_ref)^2) +
               sum(@. Float64(moments2.V) * (u2_num - u2_ref)^2)
    den_comb = max(
        sum(@. Float64(moments1.V) * (u1_ref)^2) +
        sum(@. Float64(moments2.V) * (u2_ref)^2),
        eps(Float64),
    )
    rel_comb = sqrt(num_comb / den_comb)

    h = p.lx / nx
    return h, dt, rel1, rel2, rel_comb, n1, n2, length(sys.dof_gamma.indices)
end

function main(; nx_list::Vector{Int}=[8, 12, 16, 24, 32, 48, 64], p::Heat2Ph2DParams=Heat2Ph2DParams(), umax_factor::Float64=5.0)
    hs = Float64[]
    errs = Float64[]
    errs1 = Float64[]
    errs2 = Float64[]

    @printf("Diphasic Heat_2ph_2D benchmark (PenguinDiffusion block BE)\n")
    @printf("%6s  %12s  %12s  %12s  %12s  %12s  %8s  %8s  %6s\n", "nx", "h", "dt", "rel1(fullV)", "rel2(fullV)", "rel(all)", "n1", "n2", "nG")
    for nx in nx_list
        h, dt, rel1, rel2, rel, n1, n2, ng = run_case(nx, p; umax_factor=umax_factor)
        push!(hs, h)
        push!(errs, rel)
        push!(errs1, rel1)
        push!(errs2, rel2)
        @printf("%6d  %12.5e  %12.5e  %12.5e  %12.5e  %12.5e  %8d  %8d  %6d\n", nx, h, dt, rel1, rel2, rel, n1, n2, ng)
    end

    @printf("Observed rates (combined full-volume rel L2):\n")
    for i in 2:length(nx_list)
        rate = log(errs[i - 1] / errs[i]) / log(hs[i - 1] / hs[i])
        @printf("  nx=%d -> nx=%d : %.3f\n", nx_list[i - 1], nx_list[i], rate)
    end
    @printf("Observed rates phase1/phase2:\n")
    for i in 2:length(nx_list)
        rate1 = log(errs1[i - 1] / errs1[i]) / log(hs[i - 1] / hs[i])
        rate2 = log(errs2[i - 1] / errs2[i]) / log(hs[i - 1] / hs[i])
        @printf("  nx=%d -> nx=%d : phase1=%.3f, phase2=%.3f\n", nx_list[i - 1], nx_list[i], rate1, rate2)
    end
end

main()