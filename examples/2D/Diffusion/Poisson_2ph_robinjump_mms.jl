using CartesianGeometry: geometric_moments, nan
using CartesianOperators
using PenguinBCs
using PenguinAnalysis
using PenguinDiffusion

interface_indices(cap) = findall(i -> isfinite(cap.buf.Γ[i]) && cap.buf.Γ[i] > 0.0, 1:cap.ntotal)

# 2D diphasic manufactured steady case with RobinJump + FluxJump.
# Interface is the vertical line x = ξ.
# We refine in x while keeping ny fixed so Γ (interface segment length per cut cell)
# stays constant across the refinement sweep.

Lx = 1.0
Ly = 1.0
ξ = 0.37

k1 = 1.5
k2 = 0.7
α = 2.0
β = 0.35

ny = 17
hy = Ly / (ny - 1)

# Linear manufactured fields in each phase.
A1 = 1.0
m1 = -0.9
c = 0.2
m2 = (k1 / k2) * m1
A2 = 0.3

u1_exact(x, y) = A1 + m1 * x + c * y
u2_exact(x, y) = A2 + m2 * x + c * y
u_exact(x, y, t=0.0) = x <= ξ ? u1_exact(x, y) : u2_exact(x, y)

# In the current diphasic assembly, qk are interface-integrated normal fluxes
# (they include segment length Γ). For this planar interface with fixed ny,
# Γ = hy is constant along the interface.
q1_density = k1 * m1
q2_density = -k2 * m2
gγ = α * (u2_exact(ξ, 0.0) - u1_exact(ξ, 0.0)) + β * (q1_density - q2_density) * hy

errs1 = Float64[]  # weighted L2_h on phase 1 support
errs2 = Float64[]  # weighted L2_h on phase 2 support
errs12 = Float64[] # weighted combined L2_h over both phases
flux_res = Float64[]
robin_res = Float64[]
hs = Float64[]

println("2D diphasic RobinJump manufactured (x-refinement, ny fixed)")
println("  alpha=", α, ", beta=", β, ", gγ=", gγ, ", k1=", k1, ", k2=", k2)
println("  interface x = ", ξ, ", ny = ", ny, ", hy = ", hy)

for nx in (33, 65, 129)
    grid = (range(0.0, Lx; length=nx), range(0.0, Ly; length=ny))
    moms1 = geometric_moments((x, y) -> x - ξ, grid, Float64, nan; method=:vofijul)
    moms2 = geometric_moments((x, y) -> -(x - ξ), grid, Float64, nan; method=:vofijul)
    cap1 = assembled_capacity(moms1; bc=0.0)
    cap2 = assembled_capacity(moms2; bc=0.0)

    bc = BorderConditions(
        ; left=Dirichlet(u_exact), right=Dirichlet(u_exact),
        bottom=Dirichlet(u_exact), top=Dirichlet(u_exact),
    )
    ops1 = DiffusionOps(cap1; periodic=periodic_flags(bc, 2))
    ops2 = DiffusionOps(cap2; periodic=periodic_flags(bc, 2))
    ic = InterfaceConditions(; scalar=RobinJump(α, β, gγ), flux=FluxJump(1.0, 1.0, 0.0))

    model = DiffusionModelDiph(cap1, ops1, k1, 0.0, cap2, ops2, k2, 0.0; bc_border=bc, ic=ic)
    sys = solve_steady!(model)
    lay = model.layout.offsets

    uω1 = sys.x[lay.ω1]
    uγ1 = sys.x[lay.γ1]
    uω2 = sys.x[lay.ω2]
    uγ2 = sys.x[lay.γ2]

    u1_ref = [u1_exact(cap1.C_ω[i]...) for i in 1:cap1.ntotal]
    u2_ref = [u2_exact(cap2.C_ω[i]...) for i in 1:cap2.ntotal]

    # Use PenguinAnalysis weighted norms with explicit capacity adapters.
    g1 = CellMeasure(cap1.buf.V; celltype=cap1.cell_type)
    g2 = CellMeasure(cap2.buf.V; celltype=cap2.cell_type)

    err1 = lp_error(uω1, u1_ref, g1; p=2, region=:all)
    err2 = lp_error(uω2, u2_ref, g2; p=2, region=:all)
    err12 = lp_error((uω1, uω2), (u1_ref, u2_ref), (g1, g2); p=2, region=:all)
    push!(errs1, err1)
    push!(errs2, err2)
    push!(errs12, err12)
    push!(hs, step(grid[1]))

    idxγ = interface_indices(cap1)
    q1 = k1 .* (ops1.H' * (ops1.Winv * (ops1.G * uω1 + ops1.H * uγ1)))
    q2 = k2 .* (ops2.H' * (ops2.Winv * (ops2.G * uω2 + ops2.H * uγ2)))
    fres = maximum(abs.(q1[idxγ] .+ q2[idxγ]))
    rres = maximum(abs.(α .* (uγ2[idxγ] .- uγ1[idxγ]) .+ β .* (q1[idxγ] .- q2[idxγ]) .- gγ))
    push!(flux_res, fres)
    push!(robin_res, rres)

    println("  nx=", nx, ", hx=", step(grid[1]),
            " | err1=", err1, ", err2=", err2, ", err12=", err12,
            " | flux_res=", fres, ", robin_res=", rres)
end

println("  max phase-1 error over sweep: ", maximum(errs1))
println("  max phase-2 error over sweep: ", maximum(errs2))
println("  max combined error over sweep: ", maximum(errs12))
if maximum(errs12) > 1e-12
    println("  pairwise combined orders (PenguinAnalysis): ", pairwise_orders(errs12, hs))
else
    println("  pairwise combined orders (PenguinAnalysis): round-off dominated, not meaningful")
end
println("  max flux residual over sweep: ", maximum(flux_res))
println("  max Robin residual over sweep: ", maximum(robin_res))

# The linear manufactured profile should be matched to round-off for this setup.
@assert maximum(errs1) < 1e-10
@assert maximum(errs2) < 1e-10
@assert maximum(errs12) < 1e-10
@assert maximum(flux_res) < 1e-10
@assert maximum(robin_res) < 1e-10
