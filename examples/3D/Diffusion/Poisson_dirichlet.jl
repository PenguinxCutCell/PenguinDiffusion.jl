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

# 3D manufactured steady Poisson outside a sphere:
# -Delta u = f in outer region (box minus sphere),
# embedded interface Dirichlet via Robin(alpha=1,beta=0,g=u_exact),
# outer box Dirichlet from the same analytical solution.

n = 17
grid = (
    range(0.0, 1.0; length=n),
    range(0.0, 1.0; length=n),
    range(0.0, 1.0; length=n),
)
center = (0.5, 0.5, 0.5)
radius = 0.2
body(x, y, z) = radius - sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2)

moms = geometric_moments(body, grid, Float64, nan; method=:vofijul)
cap = assembled_capacity(moms; bc=0.0)

u_exact(x, y, z) = x^2 + y^2 + z^2 + x * y - z
source(x, y, z, t) = -6.0

bc_border = BorderConditions(
    ; left=Dirichlet(u_exact), right=Dirichlet(u_exact),
    bottom=Dirichlet(u_exact), top=Dirichlet(u_exact),
    backward=Dirichlet(u_exact), forward=Dirichlet(u_exact),
)
bc_interface = Robin(1.0, 0.0, u_exact)

ops = DiffusionOps(cap; periodic=periodic_flags(bc_border, 3))
model = DiffusionModelMono(cap, ops, 1.0; source=source, bc_border=bc_border, bc_interface=bc_interface)
sys = solve_steady!(model)

idx = active_physical_indices(cap)
uomega = sys.x[model.layout.offsets.ω]
err_l2 = volume_l2_error(cap, uomega, u_exact, idx)
vol_num = sum(cap.buf.V[i] for i in idx)
vol_ref = 1.0 - 4.0 * pi * radius^3 / 3.0

println("Poisson outside sphere with embedded Dirichlet")
println("  active cells: ", length(idx))
println("  volume(active): ", vol_num, " (reference: ", vol_ref, ")")
println("  volume-weighted L2 error: ", err_l2)

@assert err_l2 < 8.0e-2
