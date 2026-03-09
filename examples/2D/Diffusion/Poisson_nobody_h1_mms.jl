using CartesianGeometry: geometric_moments, integrate, nan
using CartesianOperators
using PenguinAnalysis
using PenguinBCs
using PenguinDiffusion

function physical_primal(v::AbstractVector, nnodes::NTuple{2,Int})
    M = reshape(v, nnodes)
    return vec(@view M[1:(nnodes[1] - 1), 1:(nnodes[2] - 1)])
end

function h1_weights_from_moments(moms, wbary, nnodes::NTuple{2,Int})
    W1 = vec(@view reshape(moms.W[1], nnodes)[2:(nnodes[1] - 1), 1:(nnodes[2] - 1)])
    W2 = vec(@view reshape(moms.W[2], nnodes)[1:(nnodes[1] - 1), 2:(nnodes[2] - 1)])
    B1 = vec(@view reshape(wbary[1], nnodes)[2:(nnodes[1] - 1), 1:(nnodes[2] - 1)])
    B2 = vec(@view reshape(wbary[2], nnodes)[1:(nnodes[1] - 1), 2:(nnodes[2] - 1)])
    return (W1, W2), (B1, B2)
end

# 2D manufactured Poisson in full box (no embedded interface):
#   -Δu = f on [0,1]^2 with Dirichlet boundary from exact solution.
# We evaluate the bulk H1 seminorm error via PenguinAnalysis using:
#   - staggered wet weights W from CartesianGeometry moments,
#   - staggered barycenters Wbary from integrate(Tuple{2}, ...),
#   - analytical directional gradients sampled at Wbary.

body(x, y) = -1.0
u_exact(x, y) = sin(pi * x) * sin(pi * y)
dx_exact(x, y) = pi * cos(pi * x) * sin(pi * y)
dy_exact(x, y) = pi * sin(pi * x) * cos(pi * y)
source(x, y, t) = 2.0 * pi^2 * sin(pi * x) * sin(pi * y)

errs_h1 = Float64[]
hs = Float64[]

println("2D nobody Poisson MMS (H1 seminorm with PenguinAnalysis)")
for n in (33, 65, 129)
    grid = (range(0.0, 1.0; length=n), range(0.0, 1.0; length=n))
    moms = geometric_moments(body, grid, Float64, nan; method=:vofijul)
    cap = assembled_capacity(moms; bc=0.0)

    bc = BorderConditions(
        ; left=Dirichlet(u_exact), right=Dirichlet(u_exact),
        bottom=Dirichlet(u_exact), top=Dirichlet(u_exact),
    )
    ops = DiffusionOps(cap; periodic=periodic_flags(bc, 2))
    model = DiffusionModelMono(cap, ops, 1.0; source=source, bc_border=bc, bc_interface=Robin(0.0, 1.0, 0.0))
    sys = solve_steady!(model)
    uω = sys.x[model.layout.offsets.ω]

    nn = cap.nnodes
    dims = (nn[1] - 1, nn[2] - 1)
    spacing = (step(grid[1]), step(grid[2]))

    wbary = integrate(Tuple{2}, body, grid, Float64, nan, moms.barycenter; method=:vofijul)
    W, Wbary = h1_weights_from_moments(moms, wbary, nn)

    u_phys = physical_primal(uω, nn)
    V_phys = physical_primal(cap.buf.V, nn)
    ct_phys = physical_primal(cap.cell_type, nn)

    cellgeom = CellMeasure(V_phys; celltype=ct_phys)
    h1geom = H1Measure(W; Wbary=Wbary, celltype=ct_phys)

    err_h1 = h1_seminorm_error(u_phys, (dx_exact, dy_exact), cellgeom, h1geom, dims, spacing; region=:all)
    push!(errs_h1, err_h1)
    push!(hs, spacing[1])

    println("  n=", n, ", h=", spacing[1], ", H1 error=", err_h1)
end

ord = pairwise_orders(errs_h1, hs)
println("  pairwise H1 orders: ", ord)
println("  overall H1 order: ", overall_order(errs_h1, hs))

@assert !isempty(ord)
@assert minimum(ord) > 1.0
