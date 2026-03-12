using CartesianGeometry: geometric_moments, integrate, nan
using CartesianOperators
using CartesianGrids: CartesianGrid, grid1d
using PenguinBCs
using PenguinDiffusion
using STLInputs

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

function coarse_midplane(A::Array{<:Real,3}; stride::Int=4)
    k = cld(size(A, 3), 2)
    return A[1:stride:end, 1:stride:end, k], k
end

# Exterior steady diffusion around an embedded STL cube:
# -Delta T = 0 in the fluid (outside cube)
# outer box: T = 0 (Dirichlet)
# embedded cube boundary: T = 1 (Dirichlet via Robin(alpha=1,beta=0,g=1))

stl_path = normpath(joinpath(@__DIR__, "..", "..", "assets", "cube_ascii.stl"))
mesh = load_stl(stl_path)

n = 33
grid = CartesianGrid((-1.95, -1.95, -1.95), (2.05, 2.05, 2.05), (n, n, n))
xyz = (collect(grid1d(grid, 1)), collect(grid1d(grid, 2)), collect(grid1d(grid, 3)))

phi = sdf_on_grid(mesh, grid; n_rays=5)
body_solid = body_from_sdf(grid, phi)          # phi < 0 inside cube
body_fluid(x, y, z) = -body_solid(x, y, z)     # negative in exterior fluid region

moms = geometric_moments(body_fluid, xyz, Float64, nan; method=:vofijul)
cap = assembled_capacity(moms; bc=0.0)

bc_border = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    backward=Dirichlet(0.0), forward=Dirichlet(0.0),
)
bc_interface = Robin(1.0, 0.0, 1.0)

ops = DiffusionOps(cap; periodic=periodic_flags(bc_border, 3))
model = DiffusionModelMono(cap, ops, 1.0; source=0.0, bc_border=bc_border, bc_interface=bc_interface)
sys = solve_steady!(model)

idx = active_physical_indices(cap)
uω = sys.x[model.layout.offsets.ω]

Vsolid, _, Γsolid, _, _ = integrate(Tuple{0}, body_solid, xyz, Float64, nan; method=:vofijul)
vol_solid = sum(v for v in Vsolid if isfinite(v) && v > 0)
area_solid = sum(a for a in Γsolid if isfinite(a) && a > 0)

ncut = count(t -> isfinite(t) && t < 0.0, cap.cell_type)

u_arr = reshape(uω, cap.nnodes)
uphi, kphi = coarse_midplane(phi)
uu, ku = coarse_midplane(u_arr)

println("Embedded STL cube diffusion (exterior domain)")
println("  mesh area/volume      : ", mesh_area(mesh), " / ", mesh_volume(mesh))
println("  cut area/volume       : ", area_solid, " / ", vol_solid)
println("  cut cells             : ", ncut)
println("  active fluid cells    : ", length(idx))
println("  phi mid-plane index   : ", kphi, " (z=", xyz[3][kphi], ")")
println("  T mid-plane index     : ", ku, " (z=", xyz[3][ku], ")")
println("  phi mid-plane (coarse):")
display(uphi)
println("  T mid-plane (coarse):")
display(uu)
