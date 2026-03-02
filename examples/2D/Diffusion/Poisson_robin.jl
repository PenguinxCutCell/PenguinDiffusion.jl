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

# 2D steady Poisson inside a disk:
# -Delta u = 1 in the disk
# alpha*u + beta*dn(u) = g on the embedded interface (disk boundary)
# analytical: u(r) = C - r^2/4

n = 97
grid = (range(0.0, 4.0; length=n), range(0.0, 4.0; length=n))
center = (2.0, 2.0)
radius = 1.0
body(x, y) = sqrt((x - center[1])^2 + (y - center[2])^2) - radius

moms = geometric_moments(body, grid, Float64, nan; method=:vofijul)
cap = assembled_capacity(moms; bc=0.0)

alpha = 1.0
beta = 1.0
g = 1.0
C = g + radius^2 / 4 + radius / 2
u_exact(x, y) = C - ((x - center[1])^2 + (y - center[2])^2) / 4
source(x, y, t) = 1.0

bc_border = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)
bc_interface = Robin(alpha, beta, g)

ops = DiffusionOps(cap; periodic=periodic_flags(bc_border, 2))
model = DiffusionModelMono(cap, ops, 1.0; source=source, bc_border=bc_border, bc_interface=bc_interface)
sys = solve_steady!(model)

idx = active_physical_indices(cap)
uomega = sys.x[model.layout.offsets.ω]
err_l2 = volume_l2_error(cap, uomega, u_exact, idx)
vol_num = sum(cap.buf.V[i] for i in idx)
vol_ref = pi * radius^2

println("Poisson Robin in disk")
println("  active cells: ", length(idx))
println("  volume(active): ", vol_num, " (reference: ", vol_ref, ")")
println("  volume-weighted L2 error: ", err_l2)

@assert err_l2 < 2.0e-2
