using LinearAlgebra
using Printf

using CartesianGeometry
using CartesianOperators
using OrdinaryDiffEq
using PenguinDiffusion
using PenguinSolverCore
using SciMLBase

"""
Unsteady monophasic diffusion inside a disk with Robin interface condition.

Problem:
    ∂t u = α Δu                  in Ω (disk)
    a*u + b*∂n u = g             on Γ
    u(x,0) = u0

Example parameters (matching the user test case style):
    α = 1.0
    a = 3.0
    b = 1.0
    g = 3.0 * 400.0
    u0 = 270.0
"""

function maybe_load_analytic_packages()
    if Base.find_package("SpecialFunctions") === nothing || Base.find_package("Roots") === nothing
        return nothing
    end
    @eval using SpecialFunctions
    @eval using Roots
    return true
end

function j0_zeros_robin(N::Int, h::Float64, R::Float64; guess_shift::Float64=0.25)
    eq(alpha) = alpha * SpecialFunctions.besselj1(alpha) - h * R * SpecialFunctions.besselj0(alpha)
    zs = zeros(Float64, N)
    @inbounds for m in 1:N
        x_left = max((m - guess_shift - 0.5) * pi, 1e-9)
        x_right = (m - guess_shift + 0.5) * pi
        zs[m] = Roots.find_zero(eq, (x_left, x_right))
    end
    return zs
end

function radial_heat_center(t::Float64; R::Float64, h::Float64, α::Float64, T∞::Float64, Ti::Float64, nmodes::Int=200)
    alphas = j0_zeros_robin(nmodes, h, R)
    s = 0.0
    @inbounds for αm in alphas
        An = 2.0 * h * R / ((h^2 * R^2 + αm^2) * SpecialFunctions.besselj0(αm))
        s += An * exp(-α * αm^2 * t / R^2) # center -> J0(0)=1
    end
    return (1.0 - s) * (T∞ - Ti) + Ti
end

function main()
    # Domain and disk geometry.
    lx = 1.0
    ly = 1.0
    nx = 8
    ny = nx
    radius = ly / 4
    center = (lx / 2, ly / 2)

    circle = (x, y, _t=0.0) -> sqrt((x - center[1])^2 + (y - center[2])^2) - radius

    x = collect(range(0.0, lx; length=nx + 1))
    y = collect(range(0.0, ly; length=ny + 1))
    moments = geometric_moments(circle, (x, y), Float64, zero; method=:implicitintegration)

    # PDE coefficients.
    α = 1.0
    robin_a = 3.0
    robin_b = 1.0
    ambient = 400.0
    robin_g = robin_a * ambient

    bc = BoxBC(Val(2), Float64)
    Nd = length(moments.V)
    interface = RobinConstraint(robin_a, robin_b, robin_g, Nd)
    prob = DiffusionProblem(α, bc, interface, 0.0)
    sys = build_system(moments, prob)

    @printf("Built system: n_omega=%d, n_gamma=%d\n", length(sys.dof_omega.indices), length(sys.dof_gamma.indices))

    # Initial condition u0 = 270 inside active region.
    u0_full = fill(270.0, Nd)
    u0 = PenguinSolverCore.restrict(u0_full, sys.dof_omega)

    tspan = (0.0, 0.1)
    odeprob = PenguinSolverCore.sciml_odeproblem(sys, u0, tspan; p=nothing)
    sol = SciMLBase.solve(
        odeprob,
        OrdinaryDiffEq.Rosenbrock23(autodiff=false);
        reltol=1e-8,
        abstol=1e-8,
        saveat=range(tspan[1], tspan[2]; length=2),
    )
    SciMLBase.successful_retcode(sol) || error("time integration failed with retcode=$(sol.retcode)")

    u_end_full, _ = full_state(sys, sol.u[end])
    active = sys.dof_omega.indices
    u_active = u_end_full[active]

    # Report center temperature.
    cidx = active[argmin([(moments.barycenter[i][1] - center[1])^2 + (moments.barycenter[i][2] - center[2])^2 for i in active])]
    u_center_num = u_end_full[cidx]

    @printf("t_final = %.3f\n", tspan[2])
    @printf("u_min(active) = %.6f, u_max(active) = %.6f\n", minimum(u_active), maximum(u_active))
    @printf("u_center_numeric = %.6f\n", u_center_num)

    has_analytic = maybe_load_analytic_packages()
    if has_analytic !== nothing
        u_center_exact = radial_heat_center(tspan[2]; R=radius, h=robin_a / robin_b, α=α, T∞=ambient, Ti=270.0)
        @printf("u_center_analytic = %.6f\n", u_center_exact)
        @printf("|center error| = %.6e\n", abs(u_center_num - u_center_exact))
    else
        println("Analytical center check skipped (install SpecialFunctions + Roots to enable).")
    end
end

main()
