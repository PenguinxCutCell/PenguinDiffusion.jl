using CartesianGeometry: geometric_moments, nan
using CartesianGrids
using CartesianOperators
using PenguinBCs
using PenguinDiffusion

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

# 2D moving diphasic diffusion manufactured solution.
# Both phases share the same exact field and D1=D2=1, so jump conditions are homogeneous:
#   u1 - u2 = 0,   ∇u1·n + ∇u2·n = 0
# which is enforced via scalar/flux jump with zero right-hand side.

grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (49, 49))

const R = 0.20
const cx0 = 0.50
const cy0 = 0.50
const ax = 0.08
const ay = 0.05
const ωm = 2pi

center_x(t) = cx0 + ax * sin(ωm * t)
center_y(t) = cy0 + ay * cos(ωm * t)
body(x, y, t) = sqrt((x - center_x(t))^2 + (y - center_y(t))^2) - R

u_exact(x, y, t) = exp(-2pi^2 * t) * sin(pi * x) * sin(pi * y)

bc_value(x, y, t) = u_exact(x, y, t)
bc_box = BorderConditions(
    ; left=Dirichlet(bc_value), right=Dirichlet(bc_value),
    bottom=Dirichlet(bc_value), top=Dirichlet(bc_value),
)
ic = InterfaceConditions(; scalar=ScalarJump(1.0, 1.0, 0.0), flux=FluxJump(1.0, 1.0, 0.0))

model = MovingDiffusionModelDiph(
    grid,
    body,
    1.0,
    1.0;
    source=(0.0, 0.0),
    bc_border=bc_box,
    ic=ic,
    coeff_mode=:harmonic,
    geom_method=:vofijul,
)

m0_1 = geometric_moments((x, y) -> body(x, y, 0.0), grid1d(grid), Float64, nan; method=:vofijul)
m0_2 = geometric_moments((x, y) -> -body(x, y, 0.0), grid1d(grid), Float64, nan; method=:vofijul)
cap0_1 = assembled_capacity(m0_1; bc=0.0)
cap0_2 = assembled_capacity(m0_2; bc=0.0)

u0ω1 = [u_exact(cap0_1.C_ω[i][1], cap0_1.C_ω[i][2], 0.0) for i in 1:cap0_1.ntotal]
u0ω2 = [u_exact(cap0_2.C_ω[i][1], cap0_2.C_ω[i][2], 0.0) for i in 1:cap0_2.ntotal]
u0 = vcat(u0ω1, u0ω2)

dt = 0.0025
tspan = (0.0, 0.05)
t_end = tspan[2]

sol = solve_unsteady_moving!(model, u0, tspan; dt=dt, scheme=:BE, method=:direct, save_history=false)

cap1 = model.cap1_slab
cap2 = model.cap2_slab
lay = model.layout.offsets
u1 = sol.system.x[lay.ω1]
u2 = sol.system.x[lay.ω2]
idx1 = active_physical_indices(cap1)
idx2 = active_physical_indices(cap2)

err1 = volume_l2_error(cap1, u1, (x, y) -> u_exact(x, y, t_end), idx1, model.V1n1)
err2 = volume_l2_error(cap2, u2, (x, y) -> u_exact(x, y, t_end), idx2, model.V2n1)

println("Moving diphasic heat manufactured (2D)")
println("  grid: ", grid.n, ", dt: ", dt, ", t_end: ", t_end)
println("  active cells(final): phase1=", length(idx1), ", phase2=", length(idx2))
println("  phase-1 volume-weighted L2 error: ", err1)
println("  phase-2 volume-weighted L2 error: ", err2)

@assert err1 < 1.5e-2
@assert err2 < 1.5e-2
