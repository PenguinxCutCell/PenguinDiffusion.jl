using CartesianGeometry: geometric_moments, nan
using CartesianGrids
using CartesianOperators
using PenguinBCs
using PenguinDiffusion

# 2D moving embedded-boundary diffusion with manufactured smooth exact solution.
# PDE in moving domain Omega(t):
#     d_t u - D Delta u = f
# with exact:
#     u(x,y,t) = exp(lambda*t) * (c0 + c1*x + c2*y)
# Embedded interface uses a true Robin(alpha, beta, g) with alpha != 0 and beta != 0.

grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (129, 129))

const D = 1.0
const lambda = 0.45
const c0 = 0.75
const c1 = 0.60
const c2 = -0.40

const R = 0.20
const cx0 = 0.50
const cy0 = 0.50
const ax = 0.005
const ay = 0.003
const omega_m = 2pi

center_x(t) = cx0 + ax * sin(omega_m * t)
center_y(t) = cy0 + ay * cos(omega_m * t)
body(x, y, t) = sqrt((x - center_x(t))^2 + (y - center_y(t))^2) - R

u_exact(x, y, t) = exp(lambda * t) * (c0 + c1 * x + c2 * y)
source(x, y, t) = lambda * u_exact(x, y, t)

function dn_exact(x, y, t)
    rx = x - center_x(t)
    ry = y - center_y(t)
    r = hypot(rx, ry)
    nx = rx / r
    ny = ry / r
    return exp(lambda * t) * (c1 * nx + c2 * ny)
end

const alpha = 1.30
const beta = 0.85
g_interface(x, y, t) = alpha * u_exact(x, y, t) + beta * dn_exact(x, y, t)

bc_box = BorderConditions(
    ; left=Dirichlet(u_exact), right=Dirichlet(u_exact),
    bottom=Dirichlet(u_exact), top=Dirichlet(u_exact),
)
bc_embedded = Robin(alpha, beta, g_interface)

model = MovingDiffusionModelMono(
    grid,
    body,
    D;
    source=source,
    bc_border=bc_box,
    bc_interface=bc_embedded,
    coeff_mode=:harmonic,
    geom_method=:vofijul,
)

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

function volume_l2_error(cap, uomega, uexact_t, idx, weights)
    num = 0.0
    den = 0.0
    for i in idx
        w = weights[i]
        ue = uexact_t(cap.C_ω[i]...)
        d = uomega[i] - ue
        num += w * d^2
        den += w
    end
    return sqrt(num / den)
end

m0 = geometric_moments((x, y) -> body(x, y, 0.0), grid1d(grid), Float64, nan; method=:vofijul)
cap0 = assembled_capacity(m0; bc=0.0)
u0 = [u_exact(cap0.C_ω[i][1], cap0.C_ω[i][2], 0.0) for i in 1:cap0.ntotal]

dx, dy = meshsize(grid)
dt = 0.5 * min(dx, dy)^2 / D
tspan = (0.0, 0.04)
t_end = tspan[2]

sol = solve_unsteady_moving!(model, u0, tspan; dt=dt, scheme=:BE, method=:direct, save_history=false)

cap = model.cap_slab
lay = model.layout.offsets
uomega = sol.system.x[lay.ω]
idx = active_physical_indices(cap)
l2_vol = volume_l2_error(cap, uomega, (x, y) -> u_exact(x, y, t_end), idx, model.Vn1)

println("Moving heat Robin manufactured (2D, real Robin)")
println("  interface robin: alpha=", alpha, ", beta=", beta)
println("  grid: ", grid.n, ", dt: ", dt, ", t_end: ", t_end)
println("  active cells(final): ", length(idx))
println("  volume-weighted L2 error: ", l2_vol)

@assert l2_vol < 2.0e-3
