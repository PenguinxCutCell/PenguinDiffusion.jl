using CartesianGeometry: geometric_moments, nan
using CartesianOperators
using PenguinBCs
using PenguinDiffusion
using SpecialFunctions

# Reproduction of the diphasic unsteady disk case with generic interface-transfer
# post-processing through `compute_interface_exchange_metrics`.
# The reported transfer index is a generic adimensional number:
#     transfer_index = exchange_coefficient * Lchar / D
# which can map to Sherwood/Nusselt depending on the chosen scalar.

nx, ny = 32, 32
lx, ly = 8.0, 8.0
x0, y0 = 0.0, 0.0
grid = (range(x0, x0 + lx; length=nx + 1), range(y0, y0 + ly; length=ny + 1))

radius = ly / 4
center = (lx / 2, ly / 2)
body(x, y) = sqrt((x - center[1])^2 + (y - center[2])^2) - radius
body_c(x, y) = -body(x, y)

moms1 = geometric_moments(body, grid, Float64, nan; method=:vofijul)
moms2 = geometric_moments(body_c, grid, Float64, nan; method=:vofijul)
cap1 = assembled_capacity(moms1; bc=0.0)
cap2 = assembled_capacity(moms2; bc=0.0)

bc_border = BorderConditions() # default is homogeneous Neumann on the outer box
ops1 = DiffusionOps(cap1; periodic=periodic_flags(bc_border, 2))
ops2 = DiffusionOps(cap2; periodic=periodic_flags(bc_border, 2))

He = 1.0
Dg, Dl = 1.0, 1.0
ic = InterfaceConditions(; scalar=ScalarJump(He, 1.0, 0.0), flux=FluxJump(Dg, Dl, 0.0))

model = DiffusionModelDiph(cap1, ops1, Dg, 0.0, cap2, ops2, Dl, 0.0; bc_border=bc_border, ic=ic)

nt = cap1.ntotal
cg0, cl0 = 1.0, 0.0
u0 = vcat(fill(cg0, nt), fill(cl0, nt))

dt = 0.5 * min(step(grid[1]), step(grid[2]))^2
t_end = 1.0
sol = solve_unsteady!(model, u0, (0.0, t_end); dt=dt, scheme=:BE, method=:direct, save_history=false)

# Generic interface-transfer outputs from the new utility.
Lchar = 2 * radius
metrics = compute_interface_exchange_metrics(
    model,
    sol.system;
    characteristic_scale=Lchar,
    reference_value=(cl0, cl0),
)

# Manual assembly path (legacy post-processing) for verification.
lay = model.layout.offsets
Tω1 = @view sol.system.x[lay.ω1]
Tγ1 = @view sol.system.x[lay.γ1]
Tω2 = @view sol.system.x[lay.ω2]
Tγ2 = @view sol.system.x[lay.γ2]

Q1 = ops1.H' * (ops1.Winv * (ops1.G * Tω1 + ops1.H * Tγ1))
Q2 = ops2.H' * (ops2.Winv * (ops2.G * Tω2 + ops2.H * Tγ2))
Γ1 = cap1.buf.Γ
Γ2 = cap2.buf.Γ
mask1 = map(i -> isfinite(Γ1[i]) && Γ1[i] > 0.0 && isfinite(Q1[i]), eachindex(Γ1))
mask2 = map(i -> isfinite(Γ2[i]) && Γ2[i] > 0.0 && isfinite(Q2[i]), eachindex(Γ2))
q1_manual = -Dg * sum(Q1[mask1]) / sum(Γ1[mask1])
q2_manual = -Dl * sum(Q2[mask2]) / sum(Γ2[mask2])

# Semi-analytical reference (integral form from the legacy benchmark).
R0 = radius
D = sqrt(Dg / Dl)
Phi(u) = Dg * sqrt(Dl) * besselj1(u * R0) * bessely0(D * u * R0) - He * Dl * sqrt(Dg) * besselj0(u * R0) * bessely1(D * u * R0)
Psi(u) = Dg * sqrt(Dl) * besselj1(u * R0) * besselj0(D * u * R0) - He * Dl * sqrt(Dg) * besselj0(u * R0) * besselj1(D * u * R0)
cg_prefactor() = (4 * cg0 * Dg * Dl^2 * He) / (pi^2 * R0)
cl_prefactor() = (2 * cg0 * Dg * sqrt(Dl) * He) / pi

function simpson_integral(f, a, b; n=20_000)
    n = iseven(n) ? n : n + 1
    h = (b - a) / n
    s1 = 0.0
    s2 = 0.0
    @inbounds for k in 1:2:(n - 1)
        x = a + k * h
        y = f(x)
        s1 += isfinite(y) ? y : 0.0
    end
    @inbounds for k in 2:2:(n - 2)
        x = a + k * h
        y = f(x)
        s2 += isfinite(y) ? y : 0.0
    end
    fa = f(a)
    fb = f(b)
    fa = isfinite(fa) ? fa : 0.0
    fb = isfinite(fb) ? fb : 0.0
    return (h / 3) * (fa + fb + 4 * s1 + 2 * s2)
end

function interfacial_concentrations(t; Ufac=5.0)
    Umax = Ufac / sqrt(Dg * t)
    umin = 1e-8

    integrand_cg(u) = begin
        Φu = Phi(u)
        Ψu = Psi(u)
        denom = u^2 * (Φu^2 + Ψu^2)
        if !isfinite(denom) || denom == 0.0
            return 0.0
        end
        val = exp(-Dg * u^2 * t) * besselj0(u * R0) * besselj1(u * R0) / denom
        return isfinite(val) ? val : 0.0
    end

    integrand_cl(u) = begin
        Φu = Phi(u)
        Ψu = Psi(u)
        denom = u * (Φu^2 + Ψu^2)
        if !isfinite(denom) || denom == 0.0
            return 0.0
        end
        contrib = besselj0(D * u * R0) * Φu - bessely0(D * u * R0) * Ψu
        val = exp(-Dg * u^2 * t) * besselj1(u * R0) * contrib / denom
        return isfinite(val) ? val : 0.0
    end

    Ig = simpson_integral(integrand_cg, umin, Umax)
    Il = simpson_integral(integrand_cl, umin, Umax)

    cgR = cg_prefactor() * Ig
    clR = cl_prefactor() * Il
    return (cgR=cgR, clR=clR, mismatch=cgR - He * clR)
end

function interfacial_flux_gas(t)
    Ag = (4 * cg0 * Dg * Dl^2 * He) / (pi^2 * R0)
    Umax = 5.0 / sqrt(Dg * t)
    umin = 1e-8

    integrand(u) = begin
        Φu = Phi(u)
        Ψu = Psi(u)
        denom = u * (Φu^2 + Ψu^2)
        if !isfinite(denom) || denom == 0.0
            return 0.0
        end
        val = exp(-Dg * u^2 * t) * besselj1(u * R0)^2 / denom
        return isfinite(val) ? val : 0.0
    end

    I = simpson_integral(integrand, umin, Umax)
    return Dg * Ag * I
end

q1_ana = interfacial_flux_gas(t_end)
q2_ana = -q1_ana
cana = interfacial_concentrations(t_end)
h1_ana = q1_ana / (cana.cgR - cl0)
h2_ana = q2_ana / (cana.clR - cl0)
sh1_ana = h1_ana * Lchar / Dg
sh2_ana = h2_ana * Lchar / Dl

println("Diphasic disk transient transfer (PenguinDiffusion)")
println("  grid: ", (nx, ny), ", dt: ", dt, ", t_end: ", t_end)
println("  utility mean flux (phase1): ", metrics.phase1.mean_normal_flux)
println("  utility mean flux (phase2): ", metrics.phase2.mean_normal_flux)
println("  manual mean flux (phase1):  ", q1_manual)
println("  manual mean flux (phase2):  ", q2_manual)
println("  analytical mean flux (phase1): ", q1_ana)
println("  analytical mean flux (phase2): ", q2_ana)
println("  utility mean interface value (phase1): ", metrics.phase1.mean_interface_value)
println("  utility mean interface value (phase2): ", metrics.phase2.mean_interface_value)
println("  analytical interface value (phase1): ", cana.cgR)
println("  analytical interface value (phase2): ", cana.clR)
println("  utility exchange coefficient (phase1): ", metrics.phase1.exchange_coefficient)
println("  utility exchange coefficient (phase2): ", metrics.phase2.exchange_coefficient)
println("  analytical exchange coefficient (phase1): ", h1_ana)
println("  analytical exchange coefficient (phase2): ", h2_ana)
println("  utility transfer index (phase1): ", metrics.phase1.transfer_index)
println("  utility transfer index (phase2): ", metrics.phase2.transfer_index)
println("  analytical transfer index (phase1): ", sh1_ana)
println("  analytical transfer index (phase2): ", sh2_ana)
println("  flux balance (utility): ", metrics.flux_balance)

@assert isapprox(metrics.phase1.mean_normal_flux, q1_manual; atol=1e-12, rtol=1e-12)
@assert isapprox(metrics.phase2.mean_normal_flux, q2_manual; atol=1e-12, rtol=1e-12)
@assert abs(metrics.flux_balance) < 1e-10

rel_flux_err = abs(metrics.phase1.mean_normal_flux - q1_ana) / abs(q1_ana)
rel_c_err = abs(metrics.phase1.mean_interface_value - cana.cgR) / abs(cana.cgR)
@assert rel_flux_err < 0.03
@assert rel_c_err < 0.02
