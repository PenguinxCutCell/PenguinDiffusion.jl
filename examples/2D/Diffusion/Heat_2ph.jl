using CartesianGeometry: geometric_moments, nan
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

# 2D diphasic unsteady diffusion with an interface-consistent manufactured pair.
#
# Exact fields are identical in both phases so scalar jump and flux jump are
# consistent with:
#   ScalarJump(1, 1, 0): u1 - u2 = 0
#   FluxJump(1, 1, 0):   q1 + q2 = 0  (equal diffusivities, opposite normals)
# We use a classic homogeneous Dirichlet heat mode.

n = 65
grid = (range(0.0, 1.0; length=n), range(0.0, 1.0; length=n))
body(x, y) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.25
body_c(x, y) = -body(x, y) # complementary body for phase 2

moms = geometric_moments(body, grid, Float64, nan; method=:vofijul)
cap = assembled_capacity(moms; bc=0.0)

moms2 = geometric_moments(body_c, grid, Float64, nan; method=:vofijul)
cap2 = assembled_capacity(moms2; bc=0.0)

u1_exact(x, y, t) = exp(-2pi^2 * t) * sin(pi * x) * sin(pi * y)
u2_exact(x, y, t) = u1_exact(x, y, t)

bc_border = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)
ic = InterfaceConditions(; scalar=ScalarJump(1.0, 1.0, 0.0), flux=FluxJump(1.0, 1.0, 0.0))
ops = DiffusionOps(cap; periodic=periodic_flags(bc_border, 2))
ops2 = DiffusionOps(cap2; periodic=periodic_flags(bc_border, 2))

src1 = (x, y, t) -> 0.0
src2 = (x, y, t) -> 0.0

model = DiffusionModelDiph(
    cap,
    ops,
    1.0,
    src1,
    cap2,
    ops2,
    1.0,
    src2;
    bc_border=bc_border,
    bc_interface=ic,
)

t_end = 0.05
dt = 0.25 * step(grid[1])^2
idx1 = active_physical_indices(cap)
idx2 = active_physical_indices(cap2)
nt = cap.ntotal
u0ω1 = zeros(Float64, nt)
u0ω2 = zeros(Float64, nt)
for i in idx1
    x, y = cap.C_ω[i]
    u0ω1[i] = u1_exact(x, y, 0.0)
end
for i in idx2
    x, y = cap2.C_ω[i]
    u0ω2[i] = u2_exact(x, y, 0.0)
end
u0 = vcat(u0ω1, u0ω2)

sol = solve_unsteady!(
    model,
    u0,
    (0.0, t_end);
    dt=dt,
    scheme=:CN,
    method=:direct,
    save_history=false,
)

lay = model.layout.offsets
u1 = sol.system.x[lay.ω1]
u2 = sol.system.x[lay.ω2]
err1 = volume_l2_error(cap, u1, (x, y) -> u1_exact(x, y, t_end), idx1)
err2 = volume_l2_error(cap2, u2, (x, y) -> u2_exact(x, y, t_end), idx2)

println("2D diphasic heat (unsteady)")
println("  dt: ", dt, ", final time: ", t_end)
println("  reused constant operator: ", sol.reused_constant_operator)
println("  phase-1 volume-weighted L2 error: ", err1)
println("  phase-2 volume-weighted L2 error: ", err2)

@assert err1 < 5.0e-2
@assert err2 < 2.0e-2
