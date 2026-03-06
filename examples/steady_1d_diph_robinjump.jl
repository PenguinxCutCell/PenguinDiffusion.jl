using CartesianGeometry: geometric_moments, nan
using CartesianOperators
using PenguinBCs
using PenguinDiffusion

const HAS_MAKIE = try
    @eval using CairoMakie
    true
catch
    false
end

function active_phase_indices(cap)
    idx = Int[]
    LI = LinearIndices(cap.nnodes)
    for I in CartesianIndices(cap.nnodes)
        halo = any(d -> I[d] == cap.nnodes[d], 1:length(cap.nnodes))
        halo && continue
        i = LI[I]
        v = cap.buf.V[i]
        if isfinite(v) && v > 0.0
            push!(idx, i)
        end
    end
    return idx
end

interface_indices(cap) = findall(i -> isfinite(cap.buf.Γ[i]) && cap.buf.Γ[i] > 0.0, 1:cap.ntotal)

L = 1.0
ξ = 0.37
U0 = 1.0
UL = 0.0
α = 2.0
β = 0.35
gγ = 0.4
k1 = 1.5
k2 = 0.7

# Convention used by current diphasic assembly:
#   α*(u2-u1) + β*(q1-q2) = gγ
#   q1 + q2 = 0
# with q1 = k1*∂x u1 and q2 = -k2*∂x u2 for this 1D left/right partition.
den = 2 * β - α * (ξ / k1 + (L - ξ) / k2)
q = (gγ - α * (UL - U0)) / den

T1 = x -> U0 + (q / k1) * x
T2 = x -> UL - (q / k2) * (L - x)

grid = (range(0.0, L; length=161),)
moms1 = geometric_moments((x) -> x - ξ, grid, Float64, nan; method=:vofijul)
moms2 = geometric_moments((x) -> -(x - ξ), grid, Float64, nan; method=:vofijul)

cap1 = assembled_capacity(moms1; bc=0.0)
cap2 = assembled_capacity(moms2; bc=0.0)

bc = BorderConditions(; left=Dirichlet(U0), right=Dirichlet(UL))
ops1 = DiffusionOps(cap1; periodic=periodic_flags(bc, 1))
ops2 = DiffusionOps(cap2; periodic=periodic_flags(bc, 1))

ic = InterfaceConditions(
    ; scalar=RobinJump(α, β, gγ),
    flux=FluxJump(1.0, 1.0, 0.0),
)

model = DiffusionModelDiph(cap1, ops1, k1, 0.0, cap2, ops2, k2, 0.0; bc_border=bc, ic=ic)
sys = solve_steady!(model)
lay = model.layout.offsets

uω1 = sys.x[lay.ω1]
uγ1 = sys.x[lay.γ1]
uω2 = sys.x[lay.ω2]
uγ2 = sys.x[lay.γ2]

idx1 = active_phase_indices(cap1)
idx2 = active_phase_indices(cap2)
idxγ = interface_indices(cap1)

err1 = sqrt(sum((uω1[i] - T1(cap1.C_ω[i][1]))^2 for i in idx1) / length(idx1))
err2 = sqrt(sum((uω2[i] - T2(cap2.C_ω[i][1]))^2 for i in idx2) / length(idx2))

q1 = k1 .* (model.ops1.H' * (model.ops1.Winv * (model.ops1.G * uω1 + model.ops1.H * uγ1)))
q2 = k2 .* (model.ops2.H' * (model.ops2.Winv * (model.ops2.G * uω2 + model.ops2.H * uγ2)))

flux_res = maximum(abs.(q1[idxγ] .+ q2[idxγ]))
robin_res = maximum(abs.(α .* (uγ2[idxγ] .- uγ1[idxγ]) .+ β .* (q1[idxγ] .- q2[idxγ]) .- gγ))

println("Steady 1D diphasic Robin-jump example")
println("  parameters: alpha=", α, ", beta=", β, ", gγ=", gγ, ", k1=", k1, ", k2=", k2)
println("  grid nodes: ", length(grid[1]), ", interface position: ", ξ)
println("  L2 error phase 1: ", err1)
println("  L2 error phase 2: ", err2)
println("  max flux residual (q1 + q2): ", flux_res)
println("  max Robin residual: ", robin_res)

if HAS_MAKIE
    x1 = [cap1.C_ω[i][1] for i in idx1]
    y1 = [uω1[i] for i in idx1]
    p1 = sortperm(x1)
    x1s = x1[p1]
    y1s = y1[p1]

    x2 = [cap2.C_ω[i][1] for i in idx2]
    y2 = [uω2[i] for i in idx2]
    p2 = sortperm(x2)
    x2s = x2[p2]
    y2s = y2[p2]

    xγ = [cap1.C_γ[i][1] for i in idxγ]
    yγ1 = [uγ1[i] for i in idxγ]
    yγ2 = [uγ2[i] for i in idxγ]
    pγ = sortperm(xγ)
    xγs = xγ[pγ]
    yγ1s = yγ1[pγ]
    yγ2s = yγ2[pγ]

    fig = Figure(size=(950, 500))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="u", title="1D Diphasic RobinJump + FluxJump")
    lines!(ax, x1s, y1s; label="u1 numeric", linewidth=3)
    lines!(ax, x2s, y2s; label="u2 numeric", linewidth=3)
    lines!(ax, x1s, T1.(x1s); linestyle=:dash, label="u1 exact")
    lines!(ax, x2s, T2.(x2s); linestyle=:dash, label="u2 exact")
    scatter!(ax, xγs, yγ1s; markersize=7, label="uγ1")
    scatter!(ax, xγs, yγ2s; markersize=7, label="uγ2")
    axislegend(ax; position=:rb)

    outpath = joinpath(@__DIR__, "steady_1d_diph_robinjump.png")
    save(outpath, fig)
    println("  plot saved: ", outpath)
else
    println("  CairoMakie not available: skipping plot generation.")
end

@assert err1 < 1e-9
@assert err2 < 1e-9
@assert flux_res < 1e-9
@assert robin_res < 1e-9
