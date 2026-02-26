using LinearAlgebra
using Printf

using CartesianGeometry
using CartesianOperators
using PenguinDiffusion

"""
Steady manufactured Robin problem on a disk (inside-only monophasic solve).

PDE:
    kappa * Δu = f      in Omega (disk)
    a*u + b*kappa*dn(u) = g on Gamma (disk interface)

Manufactured solution:
    u(r) = 1 + r^2

In 2D:
    Δu = 4
    dn(u)|_{r=R} = 2R
Thus:
    f = -4*kappa
    g = a*(1 + R^2) + b*kappa*(2R)
"""

function main()
    # Geometry and PDE parameters.
    x0 = 0.5
    y0 = 0.5
    R = 0.22

    kappa = 2.3
    a = 1.7
    b = 3.4

    # Manufactured source and Robin value (constants).
    f_source = -4.0 * kappa
    g_robin = a * (1.0 + R^2) + b * kappa * (2.0 * R)

    # Grid / moments.
    n = 32
    x = collect(range(0.0, 1.0; length=n))
    y = collect(range(0.0, 1.0; length=n))

    phi(xp, yp, _t=0.0) = sqrt((xp - x0)^2 + (yp - y0)^2) - R
    moments = geometric_moments(phi, (x, y), Float64, zero; method=:implicitintegration)

    # Box BC is only used by the underlying operators; active unknowns are masked to inside cells.
    bc = BoxBC(Val(2), Float64)
    Nd = length(moments.V)
    # CartesianOperators assembles Robin as: a*u + b*dn(u) = g.
    # For the physical form a*u + b*kappa*dn(u) = g, pass b_eff = b*kappa.
    b_eff = b * kappa
    interface = RobinConstraint(a, b_eff, g_robin, Nd)
    prob = DiffusionProblem(kappa, bc, interface, f_source)
    sys = build_system(moments, prob)

    n_omega = length(sys.dof_omega.indices)
    n_gamma = length(sys.dof_gamma.indices)
    @printf("Built system: n_omega=%d, n_gamma=%d\n", n_omega, n_gamma)

    # Build reduced affine steady operator:
    #   L(u) = A*u + c, where
    #   A = L_oo - L_og * C_gamma^{-1} * C_omega
    #   c = L_og * C_gamma^{-1} * r_gamma
    #
    # Steady equation from M*du/dt = kappa*L(u) + b:
    #   0 = kappa*(A*u + c) + b
    #   A*u = -(c + b/kappa)
    active = sys.dof_omega.indices
    V_active = Float64.(sys.moments.V[active])
    b_vec = V_active .* f_source

    A = Matrix(sys.L_oo)
    c = zeros(Float64, n_omega)
    if n_gamma > 0
        S = sys.C_gamma_fact \ Matrix(sys.C_omega)
        q = sys.C_gamma_fact \ sys.r_gamma
        A .-= Matrix(sys.L_og) * S
        c .= Matrix(sys.L_og) * q
    end

    rhs = -(c .+ b_vec ./ kappa)
    u_reduced = A \ rhs

    u_full, _gamma_full = full_state(sys, u_reduced)

    # Compare against manufactured exact solution on active omega cells.
    u_exact_full = zeros(Float64, Nd)
    @inbounds for i in 1:Nd
        xb = sys.moments.barycenter[i][1]
        yb = sys.moments.barycenter[i][2]
        r2 = (xb - x0)^2 + (yb - y0)^2
        u_exact_full[i] = 1.0 + r2
    end

    err = u_full[active] .- u_exact_full[active]
    ref = u_exact_full[active]
    rel_l2 = sqrt(sum(V_active .* (err .^ 2)) / sum(V_active .* (ref .^ 2)))
    max_abs = maximum(abs, err)

    @printf("Relative L2 error (volume-weighted, active cells): %.6e\n", rel_l2)
    @printf("Max abs error (active cells): %.6e\n", max_abs)
end

main()
