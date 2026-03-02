using CartesianGeometry: geometric_moments, nan
using CartesianOperators
using PenguinBCs
using PenguinDiffusion
using PenguinSolverCore
using SparseArrays

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

# 1D manufactured steady Poisson with no embedded body:
# -u'' = f, x in [0,1], with homogeneous Neumann on both boundaries.
# Exact solution: u(x) = cos(pi*x), f = pi^2*cos(pi*x).
# Pure Neumann is singular up to a constant; we fix one dof as gauge.

n = 129
grid = (range(0.0, 1.0; length=n),)
body(x) = -1.0

moms = geometric_moments(body, grid, Float64, nan; method=:vofijul)
cap = assembled_capacity(moms; bc=0.0)

u_exact(x) = cos(pi * x)
source(x, t) = pi^2 * cos(pi * x)

bc_border = BorderConditions(; left=Neumann(0.0), right=Neumann(0.0))
ops = DiffusionOps(cap; periodic=periodic_flags(bc_border, 1))
model = DiffusionModelMono(cap, ops, 1.0; source=source, bc_border=bc_border)

lay = model.layout.offsets
nsys = last(lay.γ)
sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
assemble_steady_mono!(sys, model, 0.0)

# Gauge fix: pin one physical dof to the analytical value.
ig = lay.ω[1]
for j in 1:size(sys.A, 2)
    sys.A[ig, j] = 0.0
end
sys.A[ig, ig] = 1.0
sys.b[ig] = u_exact(cap.C_ω[1][1])

solve!(sys; method=:direct)

idx = active_physical_indices(cap)
uomega = sys.x[lay.ω]
err_l2 = volume_l2_error(cap, uomega, x -> u_exact(x), idx)

println("Poisson nobody with Neumann (MMS)")
println("  active cells: ", length(idx))
println("  volume-weighted L2 error: ", err_l2)

@assert err_l2 < 2.0e-3
