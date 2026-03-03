using CartesianGeometry: geometric_moments, nan
using CartesianGrids
using CartesianOperators
using PenguinBCs
using PenguinDiffusion

# 2D moving embedded-boundary diffusion with manufactured smooth exact solution.
# PDE in moving domain Ω(t):
#     ∂t u - D Δu = f
# with exact:
#     u(x,y,t) = exp(λ t) sin(πx) sin(πy)
# Embedded interface uses Robin(α, β, g) with α=1, β=0, g=u_exact (Dirichlet-equivalent).

grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (128, 128))

const D = 1.0
const λ = 0.35
const R = 0.20
const cx0 = 0.50
const cy0 = 0.50
const ax = 0.10
const ay = 0.06
const ωm = 2pi

center_x(t) = cx0 + ax * sin(ωm * t)
center_y(t) = cy0 + ay * cos(ωm * t)
body(x, y, t) = sqrt((x - center_x(t))^2 + (y - center_y(t))^2) - R

u_exact(x, y, t) = exp(λ * t) * sin(pi * x) * sin(pi * y)
source(x, y, t) = (λ + 2.0 * pi^2 * D) * u_exact(x, y, t)

bc_value(x, y, t) = u_exact(x, y, t)
bc_box = BorderConditions(; left=Dirichlet(bc_value), right=Dirichlet(bc_value), bottom=Dirichlet(bc_value), top=Dirichlet(bc_value))
bc_embedded = PenguinBCs.Robin(1.0, 0.0, bc_value)

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

function volume_l2_error(cap, uω, uexact_t, idx, weights)
    num = 0.0
    den = 0.0
    for i in idx
        w = weights[i]
        ue = uexact_t(cap.C_ω[i]...)
        d = uω[i] - ue
        num += w * d^2
        den += w
    end
    return sqrt(num / den)
end

m0 = geometric_moments((x, y) -> body(x, y, 0.0), grid1d(grid), Float64, nan; method=:vofijul)
cap0 = assembled_capacity(m0; bc=0.0)
u0 = [u_exact(cap0.C_ω[i][1], cap0.C_ω[i][2], 0.0) for i in 1:cap0.ntotal]

dt = 0.0025
tspan = (0.0, 0.05)
t_end = tspan[2]

sol = solve_unsteady_moving!(model, u0, tspan; dt=dt, scheme=:BE, method=:direct, save_history=false)

cap = model.cap_slab
lay = model.layout.offsets
uω = sol.system.x[lay.ω]
idx = active_physical_indices(cap)
l2_vol = volume_l2_error(cap, uω, (x, y) -> u_exact(x, y, t_end), idx, model.Vn1)

println("Moving heat Robin manufactured (2D)")
println("  grid: ", grid.n, ", dt: ", dt, ", t_end: ", t_end)
println("  active cells(final): ", length(idx))
println("  volume-weighted L2 error: ", l2_vol)

@assert l2_vol < 6.0e-3
