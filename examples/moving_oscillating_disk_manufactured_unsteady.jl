using LinearAlgebra
using Printf

using CartesianOperators
using PenguinDiffusion

"""
Moving monophasic diffusion in an oscillating disk with space-time geometry rebuild.

Manufactured-style setup inspired by:
- oscillating radius R(t) = Rm + Ra*sin(2π t / T)
- reference field phi_ana(x,y,t) = R(t)*cos(πx)*cos(πy) inside the disk
- source term as provided in the benchmark note

The example uses:
- `build_moving_system(...)`
- `moving_unsteady_block_solve(...)`
- `robin_gfun` callback for time-varying interface Dirichlet data (a=1, b=0)
"""

function main()
    # Box/grid
    lx = 4.0
    ly = 4.0
    nx = 36
    ny = 36
    x = collect(range(0.0, lx; length=nx + 1))
    y = collect(range(0.0, ly; length=ny + 1))
    dims = (length(x), length(y))
    nd = prod(dims)
    li = LinearIndices(dims)

    # Oscillation parameters
    radius_mean = 1.0
    radius_amp = 0.5
    period = 1.0
    x0 = lx / 2
    y0 = ly / 2
    dcoeff = 1.0

    # Time integration
    t0 = 0.05
    tf = 0.45
    dt = 0.01

    # Moving body level set: wet/active region is phi <= 0 (inside disk).
    oscillating_body = function (xp, yp, t)
        rt = radius_mean + radius_amp * sin(2pi * t / period)
        return sqrt((xp - x0)^2 + (yp - y0)^2) - rt
    end

    # User-provided analytical profile (inside moving disk).
    phi_ana = function (xp, yp, t)
        r = sqrt((xp - x0)^2 + (yp - y0)^2)
        rt = radius_mean + radius_amp * sin(2pi * t / period)
        if isapprox(t, 0.0; atol=1e-14)
            return 0.0
        end
        if r > rt
            return 0.0
        end
        return rt * cos(pi * xp) * cos(pi * yp)
    end

    # User-provided source term.
    source_term = function (xp, yp, _zp, t)
        r = sqrt((xp - x0)^2 + (yp - y0)^2)
        rt = radius_mean + radius_amp * sin(2pi * t / period)
        if r > rt
            return 0.0
        end
        term1 = (pi / period) * cos(pi * xp) * cos(pi * yp) * cos(2pi * t / period)
        term2 = 2pi^2 * dcoeff * (1 + 0.5 * sin(2pi * t / period)) * cos(pi * xp) * cos(pi * yp)
        return -(term1 + term2)
    end

    # Build full-node fields (node-shaped, padded layout).
    function field_from_scalar(f, t)
        out = zeros(Float64, nd)
        @inbounds for I in CartesianIndices(dims)
            idx = li[I]
            out[idx] = f(x[I[1]], y[I[2]], t)
        end
        return out
    end

    function source_field(t)
        out = zeros(Float64, nd)
        @inbounds for I in CartesianIndices(dims)
            idx = li[I]
            out[idx] = source_term(x[I[1]], y[I[2]], 0.0, t)
        end
        return out
    end

    bc = BoxBC(Val(2), Float64)
    interface = RobinConstraint(1.0, 0.0, 0.0, nd) # g is updated by robin_gfun each step
    prob = DiffusionProblem(dcoeff, bc, interface, (_sys, _u, _p, t) -> source_field(t))
    sys = build_moving_system(oscillating_body, (x, y), prob; t0=t0, t1=t0 + dt)

    u0_full = field_from_scalar(phi_ana, t0)
    robin_gfun = (_sys, _u, _p, t) -> field_from_scalar(phi_ana, t)

    sol = moving_unsteady_block_solve(
        sys,
        u0_full,
        (t0, tf);
        dt=dt,
        scheme=:CN,
        robin_gfun=robin_gfun,
        save_everystep=true,
    )

    # Final-time comparison on full node field, volume-weighted by current cap volume.
    u_num_full = zeros(Float64, nd)
    u_num_full[sys.omega_idx] .= sol.omega[end]
    u_ref_full = field_from_scalar(phi_ana, tf)

    v_full = zeros(Float64, nd)
    v_full[sys.omega_idx] .= sys.Vn1

    num = sum(@. v_full * (u_num_full - u_ref_full)^2)
    den = max(sum(@. v_full * (u_ref_full)^2), eps(Float64))
    rel_l2 = sqrt(num / den)

    @printf("Moving oscillating-disk manufactured run\n")
    @printf("nω=%d, nγ=%d, nsteps=%d\n", length(sys.omega_idx), length(sys.gamma_idx), length(sol.t) - 1)
    @printf("t0=%.3f, tf=%.3f, dt=%.3f\n", t0, tf, dt)
    @printf("volume-weighted rel L2 at tf: %.6e\n", rel_l2)
end

main()
