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

# 1D diphasic steady Poisson with interface coupling:
# Here we use the Fedkiw-style continuity conditions
# [u] = 0 and [∂n u] = 0 on an embedded interface.
#
# Manufactured reference:
# u1 = u2 = 0, f1 = f2 = 0, Dirichlet(0) on the outer box.

n = 129
xI = 0.5
grid = (range(0.0, 1.0; length=n),)
body(x) = x - xI

body_c(x) = -body(x)

moms = geometric_moments(body, grid, Float64, nan; method=:vofijul)
cap = assembled_capacity(moms; bc=0.0)

moms2 = geometric_moments(body_c, grid, Float64, nan; method=:vofijul)
cap2 = assembled_capacity(moms2; bc=0.0)

bc_border = BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0))
ops = DiffusionOps(cap; periodic=periodic_flags(bc_border, 1))
ops2 = DiffusionOps(cap2; periodic=periodic_flags(bc_border, 1))
ic = InterfaceConditions(; scalar=ScalarJump(1.0, 1.0, 0.0), flux=FluxJump(1.0, 1.0, 0.0))

# Steady manufactured solution: both phases identically zero with zero source
src1 = (x, t) -> 0.0
src2 = (x, t) -> 0.0
model = DiffusionModelDiph(cap, ops, 1.0, src1, cap2, ops2, 1.0, src2; bc_border=bc_border, bc_interface=ic)

sys = solve_steady!(model)
lay = model.layout.offsets
idx = active_physical_indices(cap)

u1 = sys.x[lay.ω1]
u2 = sys.x[lay.ω2]
err1 = volume_l2_error(cap, u1, x -> 0.0, idx)
err2 = volume_l2_error(cap, u2, x -> 0.0, idx)

println("1D diphasic Poisson")
println("  active cells: ", length(idx))
println("  phase-1 volume-weighted L2 error: ", err1)
println("  phase-2 volume-weighted L2 error: ", err2)

@assert err1 < 1.0e-12
@assert err2 < 1.0e-12
