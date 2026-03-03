module PenguinDiffusion

using LinearAlgebra
using SparseArrays
using StaticArrays

using CartesianOperators
using PenguinBCs
using PenguinSolverCore

export DiffusionModelMono, DiffusionModelDiph
export assemble_steady_mono!, assemble_unsteady_mono!
export assemble_steady_diph!, assemble_unsteady_diph!
export solve_steady!, solve_unsteady!

struct DiffusionModelMono{N,T,DT,ST,IT}
    ops::DiffusionOps{N,T}
    cap::AssembledCapacity{N,T}
    D::DT
    source::ST
    bc_border::BorderConditions
    bc_interface::IT
    layout::UnknownLayout
    coeff_mode::Symbol
end

function DiffusionModelMono(
    cap::AssembledCapacity{N,T},
    ops::DiffusionOps{N,T},
    D;
    source=((args...) -> zero(T)),
    bc_border::BorderConditions=BorderConditions(),
    bc_interface::Union{Nothing,PenguinBCs.Robin}=nothing,
    layout::UnknownLayout=layout_mono(cap.ntotal),
    coeff_mode::Symbol=:harmonic,
) where {N,T}
    coeff_mode_eff = _normalize_coeff_mode(coeff_mode)
    return DiffusionModelMono{N,T,typeof(D),typeof(source),typeof(bc_interface)}(
        ops, cap, D, source, bc_border, bc_interface, layout, coeff_mode_eff
    )
end

struct DiffusionModelDiph{N,T,D1T,D2T,S1T,S2T,IT}
    ops1::DiffusionOps{N,T}
    cap1::AssembledCapacity{N,T}
    D1::D1T
    source1::S1T
    ops2::DiffusionOps{N,T}
    cap2::AssembledCapacity{N,T}
    D2::D2T
    source2::S2T
    bc_border::BorderConditions
    ic::IT
    layout::UnknownLayout
    coeff_mode::Symbol
end

function DiffusionModelDiph(
    cap1::AssembledCapacity{N,T},
    ops1::DiffusionOps{N,T},
    D1,
    source1,
    cap2::AssembledCapacity{N,T},
    ops2::DiffusionOps{N,T},
    D2,
    source2;
    bc_border::BorderConditions=BorderConditions(),
    ic::Union{Nothing,InterfaceConditions}=nothing,
    bc_interface::Union{Nothing,InterfaceConditions}=nothing,
    layout::UnknownLayout=layout_diph(cap1.ntotal),
    coeff_mode::Symbol=:harmonic,
) where {N,T}
    cap1.ntotal == cap2.ntotal || throw(ArgumentError("cap1 and cap2 must have identical ntotal"))
    cap1.nnodes == cap2.nnodes || throw(ArgumentError("cap1 and cap2 must have identical nnodes"))
    if !(ic === nothing) && !(bc_interface === nothing) && (ic !== bc_interface)
        throw(ArgumentError("provide at most one interface condition via `ic` or `bc_interface`"))
    end
    ic_eff = ic === nothing ? bc_interface : ic
    coeff_mode_eff = _normalize_coeff_mode(coeff_mode)
    return DiffusionModelDiph{N,T,typeof(D1),typeof(D2),typeof(source1),typeof(source2),typeof(ic_eff)}(
        ops1, cap1, D1, source1, ops2, cap2, D2, source2, bc_border, ic_eff, layout, coeff_mode_eff
    )
end

function DiffusionModelDiph(
    cap::AssembledCapacity{N,T},
    ops::DiffusionOps{N,T},
    D1,
    D2;
    source=((args...) -> (zero(T), zero(T))),
    bc_border::BorderConditions=BorderConditions(),
    ic::Union{Nothing,InterfaceConditions}=nothing,
    bc_interface::Union{Nothing,InterfaceConditions}=nothing,
    layout::UnknownLayout=layout_diph(cap.ntotal),
    coeff_mode::Symbol=:harmonic,
) where {N,T}
    source1, source2 = if source isa Tuple && length(source) == 2
        (source[1], source[2])
    elseif source isa Function
        (;
            s1=(args...) -> begin
                s = applicable(source, args...) ? source(args...) : source(args[1:(end - 1)]...)
                s[1]
            end,
            s2=(args...) -> begin
                s = applicable(source, args...) ? source(args...) : source(args[1:(end - 1)]...)
                s[2]
            end,
        ) |> x -> (x.s1, x.s2)
    else
        throw(ArgumentError("diph source must be a function returning a tuple or a tuple of two callbacks/constants"))
    end
    return DiffusionModelDiph(
        cap,
        ops,
        D1,
        source1,
        cap,
        ops,
        D2,
        source2;
        bc_border=bc_border,
        ic=ic,
        bc_interface=bc_interface,
        layout=layout,
        coeff_mode=coeff_mode,
    )
end

function _eval_fun_or_const(v, x::SVector{N,T}, t::T) where {N,T}
    if v isa Number
        return convert(T, v)
    elseif v isa Function
        if applicable(v, x..., t)
            return convert(T, v(x..., t))
        elseif applicable(v, x...)
            return convert(T, v(x...))
        end
    end
    throw(ArgumentError("callback/value must be numeric, (x...), or (x..., t)"))
end

function _source_values_mono(cap::AssembledCapacity{N,T}, source, t::T) where {N,T}
    out = Vector{T}(undef, cap.ntotal)
    @inbounds for i in eachindex(out)
        out[i] = _eval_fun_or_const(source, cap.C_ω[i], t)
    end
    return out
end

function _source_values_diph(
    cap1::AssembledCapacity{N,T},
    source1,
    cap2::AssembledCapacity{N,T},
    source2,
    t::T,
) where {N,T}
    return _source_values_mono(cap1, source1, t), _source_values_mono(cap2, source2, t)
end

function _sample_coeff(cap::AssembledCapacity{N,T}, D, t::T) where {N,T}
    out = Vector{T}(undef, cap.ntotal)
    @inbounds for i in eachindex(out)
        out[i] = convert(T, eval_coeff(D, cap.C_ω[i], t, i))
    end
    return out
end

function _normalize_coeff_mode(mode::Symbol)::Symbol
    if mode === :harmonic || mode === :arithmetic || mode === :face || mode === :cell
        return mode
    end
    throw(ArgumentError("unknown coeff_mode `$mode`; expected :harmonic, :arithmetic, :face, or :cell"))
end

function _harmonic_mean(a::T, b::T) where {T}
    den = a + b
    return iszero(den) ? zero(T) : (T(2) * a * b) / den
end

function _face_coeff_values(
    cap::AssembledCapacity{N,T},
    D,
    t::T,
    mode::Symbol,
) where {N,T}
    mode_eff = _normalize_coeff_mode(mode)
    nt = cap.ntotal
    vals = Vector{T}(undef, N * nt)
    LI = LinearIndices(cap.nnodes)
    cω = cap.C_ω

    if mode_eff === :face
        @inbounds for d in 1:N
            offset = (d - 1) * nt
            xlow = convert(T, cap.xyz[d][1])
            for I in CartesianIndices(cap.nnodes)
                lin = LI[I]
                if any(k -> I[k] == cap.nnodes[k], 1:N)
                    vals[offset + lin] = zero(T)
                    continue
                end
                xface = if I[d] == 1
                    SVector{N,T}(ntuple(k -> (k == d ? xlow : cω[lin][k]), N))
                else
                    Iminus = CartesianIndex(ntuple(k -> (k == d ? I[k] - 1 : I[k]), N))
                    linm = LI[Iminus]
                    SVector{N,T}(ntuple(k -> (cω[lin][k] + cω[linm][k]) / T(2), N))
                end
                vals[offset + lin] = convert(T, eval_coeff(D, xface, t, lin))
            end
        end
        return vals
    end

    kcell = _sample_coeff(cap, D, t)
    @inbounds for d in 1:N
        offset = (d - 1) * nt
        for I in CartesianIndices(cap.nnodes)
            lin = LI[I]
            if any(k -> I[k] == cap.nnodes[k], 1:N)
                vals[offset + lin] = zero(T)
                continue
            end
            ki = kcell[lin]
            if mode_eff === :cell
                vals[offset + lin] = ki
            elseif I[d] == 1
                vals[offset + lin] = ki
            else
                Iminus = CartesianIndex(ntuple(k -> (k == d ? I[k] - 1 : I[k]), N))
                kim = kcell[LI[Iminus]]
                vals[offset + lin] = mode_eff === :harmonic ? _harmonic_mean(ki, kim) : (ki + kim) / T(2)
            end
        end
    end
    return vals
end

function _interface_mask(cap::AssembledCapacity{N,T}) where {N,T}
    m = Vector{Bool}(undef, cap.ntotal)
    @inbounds for i in eachindex(m)
        γ = cap.buf.Γ[i]
        m[i] = isfinite(γ) && γ > zero(T)
    end
    return m
end

function _interface_diagonals_mono(cap::AssembledCapacity{N,T}, ic::Union{Nothing,PenguinBCs.Robin}, t::T) where {N,T}
    α = zeros(T, cap.ntotal)
    β = zeros(T, cap.ntotal)
    g = zeros(T, cap.ntotal)
    ic === nothing && return α, β, g

    mask = _interface_mask(cap)
    @inbounds for i in eachindex(mask)
        mask[i] || continue
        x = cap.C_γ[i]
        α[i] = convert(T, eval_bc(ic.α, x, t))
        β[i] = convert(T, eval_bc(ic.β, x, t))
        g[i] = convert(T, eval_bc(ic.value, x, t))
    end
    return α, β, g
end

function _interface_coupling_diph(
    cap1::AssembledCapacity{N,T},
    cap2::AssembledCapacity{N,T},
    ic::Union{Nothing,InterfaceConditions},
    t::T,
) where {N,T}
    nt = cap1.ntotal
    αs1 = zeros(T, nt)
    αs2 = zeros(T, nt)
    βs1 = zeros(T, nt)
    βs2 = zeros(T, nt)
    gs = zeros(T, nt)

    αf1 = zeros(T, nt)
    αf2 = zeros(T, nt)
    βf1 = zeros(T, nt)
    βf2 = zeros(T, nt)
    gf = zeros(T, nt)

    ic === nothing && return αs1, αs2, βs1, βs2, gs, αf1, αf2, βf1, βf2, gf

    tol = sqrt(eps(T))
    @inbounds for i in 1:nt
        γ1 = cap1.buf.Γ[i]
        γ2 = cap2.buf.Γ[i]
        active1 = isfinite(γ1) && γ1 > zero(T)
        active2 = isfinite(γ2) && γ2 > zero(T)
        if active1 != active2
            throw(ArgumentError("cap1/cap2 interface masks differ at index $i"))
        end
        active1 || continue
        if abs(γ1 - γ2) > tol * max(one(T), abs(γ1), abs(γ2))
            throw(ArgumentError("cap1/cap2 interface measures differ at index $i"))
        end

        x = cap1.C_γ[i]

        if ic.scalar isa ScalarJump
            αs1[i] = convert(T, eval_bc(ic.scalar.α₁, x, t))
            αs2[i] = convert(T, eval_bc(ic.scalar.α₂, x, t))
            gs[i] = convert(T, eval_bc(ic.scalar.value, x, t))
        elseif ic.scalar isa RobinJump
            αs1[i] = convert(T, eval_bc(ic.scalar.α, x, t))
            αs2[i] = convert(T, eval_bc(ic.scalar.α, x, t))
            βs1[i] = convert(T, eval_bc(ic.scalar.β, x, t))
            βs2[i] = convert(T, eval_bc(ic.scalar.β, x, t))
            gs[i] = convert(T, eval_bc(ic.scalar.value, x, t))
        elseif !(ic.scalar === nothing)
            throw(ArgumentError("unsupported scalar interface condition type $(typeof(ic.scalar))"))
        end

        if ic.flux isa FluxJump
            βf1[i] = convert(T, eval_bc(ic.flux.β₁, x, t))
            βf2[i] = convert(T, eval_bc(ic.flux.β₂, x, t))
            gf[i] = convert(T, eval_bc(ic.flux.value, x, t))
        elseif ic.flux isa RobinJump
            αf1[i] = convert(T, eval_bc(ic.flux.α, x, t))
            αf2[i] = convert(T, eval_bc(ic.flux.α, x, t))
            βf1[i] = convert(T, eval_bc(ic.flux.β, x, t))
            βf2[i] = convert(T, eval_bc(ic.flux.β, x, t))
            gf[i] = convert(T, eval_bc(ic.flux.value, x, t))
        elseif !(ic.flux === nothing)
            throw(ArgumentError("unsupported flux interface condition type $(typeof(ic.flux))"))
        end
    end
    return αs1, αs2, βs1, βs2, gs, αf1, αf2, βf1, βf2, gf
end

function _insert_block!(A::SparseMatrixCSC{T,Int}, rows::UnitRange{Int}, cols::UnitRange{Int}, B::SparseMatrixCSC{T,Int}) where {T}
    size(B, 1) == length(rows) || throw(DimensionMismatch("block rows do not match target range"))
    size(B, 2) == length(cols) || throw(DimensionMismatch("block cols do not match target range"))
    @inbounds for j in 1:size(B, 2)
        for p in nzrange(B, j)
            i = B.rowval[p]
            A[rows[i], cols[j]] = A[rows[i], cols[j]] + B.nzval[p]
        end
    end
    return A
end

function _insert_vec!(b::Vector{T}, rows::UnitRange{Int}, v::Vector{T}) where {T}
    length(v) == length(rows) || throw(DimensionMismatch("vector block length mismatch"))
    @inbounds for i in eachindex(v)
        b[rows[i]] += v[i]
    end
    return b
end

function _set_row_identity!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, row::Int) where {T}
    @inbounds for j in 1:size(A, 2)
        A[row, j] = zero(T)
    end
    A[row, row] = one(T)
    b[row] = zero(T)
    return A, b
end

function _cell_activity_masks(cap::AssembledCapacity{N,T}) where {N,T}
    nt = cap.ntotal
    activeω = BitVector(undef, nt)
    activeγ = BitVector(undef, nt)
    LI = LinearIndices(cap.nnodes)
    for I in CartesianIndices(cap.nnodes)
        lin = LI[I]
        halo = any(d -> I[d] == cap.nnodes[d], 1:N)
        if halo
            activeω[lin] = false
            activeγ[lin] = false
            continue
        end
        v = cap.buf.V[lin]
        γ = cap.buf.Γ[lin]
        activeω[lin] = isfinite(v) && v > zero(T)
        activeγ[lin] = isfinite(γ) && γ > zero(T)
    end
    return activeω, activeγ
end

function _mono_row_activity(cap::AssembledCapacity{N,T}, lay) where {N,T}
    activeω, activeγ = _cell_activity_masks(cap)
    nsys = maximum((last(lay.ω), last(lay.γ)))
    active = falses(nsys)
    @inbounds for i in 1:cap.ntotal
        active[lay.ω[i]] = activeω[i]
        active[lay.γ[i]] = activeγ[i]
    end
    return active
end

function _diph_row_activity(cap1::AssembledCapacity{N,T}, cap2::AssembledCapacity{N,T}, lay) where {N,T}
    activeω1, activeγ1 = _cell_activity_masks(cap1)
    activeω2, activeγ2 = _cell_activity_masks(cap2)
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
    active = falses(nsys)
    @inbounds for i in 1:cap1.ntotal
        active[lay.ω1[i]] = activeω1[i]
        active[lay.γ1[i]] = activeγ1[i]
        active[lay.ω2[i]] = activeω2[i]
        active[lay.γ2[i]] = activeγ2[i]
    end
    return active
end

function _apply_row_identity_constraints!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    active_rows::BitVector,
) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("row-identity constraints require square matrix"))
    length(b) == n || throw(ArgumentError("rhs length mismatch"))
    length(active_rows) == n || throw(ArgumentError("active row mask length mismatch"))

    p = Vector{T}(undef, n)
    @inbounds for i in 1:n
        ai = active_rows[i]
        p[i] = ai ? zero(T) : one(T)
        ai || (b[i] = zero(T))
    end

    @inbounds for j in 1:size(A, 2)
        aj = active_rows[j]
        for k in nzrange(A, j)
            if !(aj && active_rows[A.rowval[k]])
                # Explicit overwrite avoids NaN propagation from values like 0*NaN.
                A.nzval[k] = zero(T)
            end
        end
    end
    dropzeros!(A)

    Aout = A + spdiagm(0 => p)
    return Aout, b
end

function _scale_rows!(A::SparseMatrixCSC{T,Int}, rows::UnitRange{Int}, α::T) where {T}
    α == one(T) && return A
    r1 = first(rows)
    r2 = last(rows)
    @inbounds for j in 1:size(A, 2)
        for p in nzrange(A, j)
            i = A.rowval[p]
            if r1 <= i <= r2
                A.nzval[p] *= α
            end
        end
    end
    return A
end

function _weighted_core_ops(
    cap::AssembledCapacity{N,T},
    ops::DiffusionOps{N,T},
    D,
    t::T,
    mode::Symbol,
) where {N,T}
    G = ops.G
    H = ops.H
    κface = _face_coeff_values(cap, D, t, mode)
    Wκ = spdiagm(0 => (ops.Winv.nzval .* κface))
    K = G' * Wκ * G
    C = G' * Wκ * H
    J = H' * Wκ * G
    L = H' * Wκ * H
    return K, C, J, L
end

function _is_canonical_mono_layout(lay, nt::Int)
    return lay.ω == (1:nt) && lay.γ == ((nt + 1):(2 * nt))
end

function _is_canonical_diph_layout(lay, nt::Int)
    return lay.ω1 == (1:nt) &&
           lay.γ1 == ((nt + 1):(2 * nt)) &&
           lay.ω2 == ((2 * nt + 1):(3 * nt)) &&
           lay.γ2 == ((3 * nt + 1):(4 * nt))
end

function assemble_steady_mono!(sys::LinearSystem{T}, model::DiffusionModelMono{N,T}, t::T) where {N,T}
    nt = model.cap.ntotal
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω), last(lay.γ)))

    K, C, J, L = _weighted_core_ops(model.cap, model.ops, model.D, t, model.coeff_mode)
    fω = _source_values_mono(model.cap, model.source, t)
    α, β, gγ = _interface_diagonals_mono(model.cap, model.bc_interface, t)

    Iβ = spdiagm(0 => β)
    Iα = spdiagm(0 => α)
    Iγ = model.cap.Γ

    A11 = K
    A12 = C
    A21 = Iβ * J
    A22 = Iβ * L + Iα * Iγ
    if model.bc_interface === nothing
        A12 = spzeros(T, nt, nt)
        A21 = spzeros(T, nt, nt)
        A22 = spdiagm(0 => ones(T, nt))
    end
    b1 = model.cap.V * fω
    b2 = Iγ * gγ

    A, b = if _is_canonical_mono_layout(lay, nt)
        ([A11 A12; A21 A22], vcat(b1, b2))
    else
        Awork = spzeros(T, nsys, nsys)
        bwork = zeros(T, nsys)
        _insert_block!(Awork, lay.ω, lay.ω, A11)
        _insert_block!(Awork, lay.ω, lay.γ, A12)
        _insert_block!(Awork, lay.γ, lay.ω, A21)
        _insert_block!(Awork, lay.γ, lay.γ, A22)
        _insert_vec!(bwork, lay.ω, b1)
        _insert_vec!(bwork, lay.γ, b2)
        (Awork, bwork)
    end

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing

    apply_box_bc_mono!(sys.A, sys.b, model.cap, model.ops, model.D, model.bc_border; t=t, layout=model.layout)
    active_rows = _mono_row_activity(model.cap, lay)
    sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
    return sys
end

function assemble_unsteady_mono!(sys::LinearSystem{T}, model::DiffusionModelMono{N,T}, uⁿ, t::T, dt::T, scheme) where {N,T}
    θ = scheme isa Real ? convert(T, scheme) : one(T)
    assemble_steady_mono!(sys, model, t + θ * dt)

    lay = model.layout.offsets
    nt = model.cap.ntotal
    nsys = maximum((last(lay.ω), last(lay.γ)))

    ufull = if length(uⁿ) == nsys
        Vector{T}(uⁿ)
    elseif length(uⁿ) == nt
        v = zeros(T, nsys)
        v[lay.ω] .= Vector{T}(uⁿ)
        v
    else
        v = zeros(T, nsys)
        v[lay.ω] .= Vector{T}(uⁿ[lay.ω])
        v
    end

    if θ != one(T)
        Aω_prev = sys.A[lay.ω, :]
        corr = Aω_prev * ufull
        _scale_rows!(sys.A, lay.ω, θ)
        _insert_vec!(sys.b, lay.ω, (-(one(T) - θ)) .* corr)
    end

    M = model.cap.buf.V ./ dt
    sys.A = sys.A + sparse(lay.ω, lay.ω, M, nsys, nsys)

    uω = Vector{T}(ufull[lay.ω])
    _insert_vec!(sys.b, lay.ω, M .* uω)
    active_rows = _mono_row_activity(model.cap, lay)
    sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
    sys.cache = nothing
    return sys
end

function assemble_steady_diph!(sys::LinearSystem{T}, model::DiffusionModelDiph{N,T}, t::T) where {N,T}
    nt = model.cap1.ntotal
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

    K1, C1, J1, L1 = _weighted_core_ops(model.cap1, model.ops1, model.D1, t, model.coeff_mode)
    K2, C2, J2, L2 = _weighted_core_ops(model.cap2, model.ops2, model.D2, t, model.coeff_mode)
    f1, f2 = _source_values_diph(model.cap1, model.source1, model.cap2, model.source2, t)
    αs1, αs2, βs1, βs2, gs, αf1, αf2, βf1, βf2, gf = _interface_coupling_diph(model.cap1, model.cap2, model.ic, t)

    Iαs1 = spdiagm(0 => αs1)
    Iαs2 = spdiagm(0 => αs2)
    Iβs1 = spdiagm(0 => βs1)
    Iβs2 = spdiagm(0 => βs2)
    Iαf1 = spdiagm(0 => αf1)
    Iαf2 = spdiagm(0 => αf2)
    Iβf1 = spdiagm(0 => βf1)
    Iβf2 = spdiagm(0 => βf2)
    IΓ1 = model.cap1.Γ
    IΓ2 = model.cap2.Γ

    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)

    Aωω1 = K1
    Aωγ1 = C1
    Aωω2 = K2
    Aωγ2 = C2
    _insert_block!(A, lay.ω1, lay.ω1, Aωω1)
    _insert_block!(A, lay.ω1, lay.γ1, Aωγ1)
    _insert_block!(A, lay.ω2, lay.ω2, Aωω2)
    _insert_block!(A, lay.ω2, lay.γ2, Aωγ2)

    if model.ic === nothing
        _insert_block!(A, lay.γ1, lay.γ1, spdiagm(0 => ones(T, nt)))
        _insert_block!(A, lay.γ2, lay.γ2, spdiagm(0 => ones(T, nt)))
    else
        has_scalar = !(model.ic.scalar === nothing)
        has_flux = !(model.ic.flux === nothing)

        if has_scalar
            Aγ1ω1 = Iβs1 * J1
            Aγ1γ1 = Iβs1 * L1 - Iαs1
            Aγ1ω2 = -(Iβs2 * J2)
            Aγ1γ2 = -(Iβs2 * L2) + Iαs2

            _insert_block!(A, lay.γ1, lay.ω1, Aγ1ω1)
            _insert_block!(A, lay.γ1, lay.γ1, Aγ1γ1)
            _insert_block!(A, lay.γ1, lay.ω2, Aγ1ω2)
            _insert_block!(A, lay.γ1, lay.γ2, Aγ1γ2)
            _insert_vec!(b, lay.γ1, gs)
        else
            _insert_block!(A, lay.γ1, lay.γ1, spdiagm(0 => ones(T, nt)))
        end

        if has_flux
            Aγ2ω1 = Iβf1 * J1
            Aγ2γ1 = Iβf1 * L1 - Iαf1
            Aγ2ω2 = Iβf2 * J2
            Aγ2γ2 = Iβf2 * L2 + Iαf2

            _insert_block!(A, lay.γ2, lay.ω1, Aγ2ω1)
            _insert_block!(A, lay.γ2, lay.γ1, Aγ2γ1)
            _insert_block!(A, lay.γ2, lay.ω2, Aγ2ω2)
            _insert_block!(A, lay.γ2, lay.γ2, Aγ2γ2)
            _insert_vec!(b, lay.γ2, gf)
        else
            _insert_block!(A, lay.γ2, lay.γ2, spdiagm(0 => ones(T, nt)))
        end
    end

    _insert_vec!(b, lay.ω1, model.cap1.V * f1)
    _insert_vec!(b, lay.ω2, model.cap2.V * f2)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing

    layω1 = UnknownLayout(model.cap1.ntotal, (ω=lay.ω1,))
    layω2 = UnknownLayout(model.cap2.ntotal, (ω=lay.ω2,))
    apply_box_bc_mono!(sys.A, sys.b, model.cap1, model.ops1, model.D1, model.bc_border; t=t, layout=layω1)
    apply_box_bc_mono!(sys.A, sys.b, model.cap2, model.ops2, model.D2, model.bc_border; t=t, layout=layω2)
    active_rows = _diph_row_activity(model.cap1, model.cap2, lay)
    sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
    return sys
end

function assemble_unsteady_diph!(sys::LinearSystem{T}, model::DiffusionModelDiph{N,T}, uⁿ, t::T, dt::T, scheme) where {N,T}
    θ = scheme isa Real ? convert(T, scheme) : one(T)
    assemble_steady_diph!(sys, model, t + θ * dt)

    lay = model.layout.offsets
    nt = model.cap1.ntotal
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

    ufull = if length(uⁿ) == nsys
        Vector{T}(uⁿ)
    elseif length(uⁿ) == 2 * nt
        v = zeros(T, nsys)
        v[lay.ω1] .= Vector{T}(uⁿ[1:nt])
        v[lay.ω2] .= Vector{T}(uⁿ[(nt + 1):(2 * nt)])
        v
    else
        v = zeros(T, nsys)
        v[lay.ω1] .= Vector{T}(uⁿ[lay.ω1])
        v[lay.ω2] .= Vector{T}(uⁿ[lay.ω2])
        v
    end

    if θ != one(T)
        Aω1_prev = sys.A[lay.ω1, :]
        Aω2_prev = sys.A[lay.ω2, :]
        corr1 = Aω1_prev * ufull
        corr2 = Aω2_prev * ufull
        _scale_rows!(sys.A, lay.ω1, θ)
        _scale_rows!(sys.A, lay.ω2, θ)
        _insert_vec!(sys.b, lay.ω1, (-(one(T) - θ)) .* corr1)
        _insert_vec!(sys.b, lay.ω2, (-(one(T) - θ)) .* corr2)
    end

    M1 = model.cap1.buf.V ./ dt
    M2 = model.cap2.buf.V ./ dt
    rows = vcat(collect(lay.ω1), collect(lay.ω2))
    vals = vcat(M1, M2)
    sys.A = sys.A + sparse(rows, rows, vals, nsys, nsys)

    u1 = Vector{T}(ufull[lay.ω1])
    u2 = Vector{T}(ufull[lay.ω2])
    _insert_vec!(sys.b, lay.ω1, M1 .* u1)
    _insert_vec!(sys.b, lay.ω2, M2 .* u2)
    active_rows = _diph_row_activity(model.cap1, model.cap2, lay)
    sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
    sys.cache = nothing
    return sys
end

function PenguinSolverCore.assemble!(sys::LinearSystem{T}, model::DiffusionModelMono{N,T}, t, dt) where {N,T}
    assemble_steady_mono!(sys, model, convert(T, t))
end

function PenguinSolverCore.assemble!(sys::LinearSystem{T}, model::DiffusionModelDiph{N,T}, t, dt) where {N,T}
    assemble_steady_diph!(sys, model, convert(T, t))
end

function solve_steady!(model::DiffusionModelMono{N,T}; t::T=zero(T), method::Symbol=:direct, kwargs...) where {N,T}
    n = maximum((last(model.layout.offsets.ω), last(model.layout.offsets.γ)))
    sys = LinearSystem(spzeros(T, n, n), zeros(T, n))
    assemble_steady_mono!(sys, model, t)
    solve!(sys; method=method, kwargs...)
    return sys
end

function solve_steady!(model::DiffusionModelDiph{N,T}; t::T=zero(T), method::Symbol=:direct, kwargs...) where {N,T}
    lay = model.layout.offsets
    n = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
    sys = LinearSystem(spzeros(T, n, n), zeros(T, n))
    assemble_steady_diph!(sys, model, t)
    solve!(sys; method=method, kwargs...)
    return sys
end

function _theta_from_scheme(::Type{T}, scheme) where {T}
    if scheme isa Symbol
        if scheme === :BE
            return one(T)
        elseif scheme === :CN
            return convert(T, 0.5)
        end
        throw(ArgumentError("unknown scheme `$scheme`; expected :BE or :CN"))
    elseif scheme isa Real
        return convert(T, scheme)
    end
    throw(ArgumentError("scheme must be a Symbol (:BE/:CN) or a numeric theta"))
end

function _value_time_dependent(v, x::SVector{N,T}) where {N,T}
    return v isa Function && applicable(v, x..., zero(T))
end

function _coeff_time_dependent(D, x::SVector{N,T}) where {N,T}
    if D isa Function
        return applicable(D, x..., zero(T)) || applicable(D, 1, zero(T))
    end
    return false
end

function _source_mono_time_dependent(source, x::SVector{N,T}) where {N,T}
    return source isa Function && applicable(source, x..., zero(T))
end

function _source_diph_time_dependent(source, x::SVector{N,T}) where {N,T}
    if source isa Function
        return applicable(source, x..., zero(T))
    elseif source isa Tuple && length(source) == 2
        return _value_time_dependent(source[1], x) || _value_time_dependent(source[2], x)
    end
    return false
end

function _border_values_time_dependent(bc::BorderConditions, x::SVector{N,T}) where {N,T}
    for side_bc in values(bc.borders)
        if side_bc isa Dirichlet || side_bc isa Neumann
            _value_time_dependent(side_bc.value, x) && return true
        elseif side_bc isa Robin
            (_value_time_dependent(side_bc.α, x) ||
             _value_time_dependent(side_bc.β, x) ||
             _value_time_dependent(side_bc.value, x)) && return true
        end
    end
    return false
end

function _interface_mono_matrix_time_dependent(ic::Union{Nothing,PenguinBCs.Robin}, x::SVector{N,T}) where {N,T}
    ic === nothing && return false
    return _value_time_dependent(ic.α, x) || _value_time_dependent(ic.β, x)
end

function _interface_mono_rhs_time_dependent(ic::Union{Nothing,PenguinBCs.Robin}, x::SVector{N,T}) where {N,T}
    ic === nothing && return false
    return _value_time_dependent(ic.value, x)
end

function _interface_diph_matrix_time_dependent(ic::Union{Nothing,InterfaceConditions}, x::SVector{N,T}) where {N,T}
    ic === nothing && return false
    for comp in (ic.scalar, ic.flux)
        comp === nothing && continue
        if comp isa ScalarJump
            (_value_time_dependent(comp.α₁, x) || _value_time_dependent(comp.α₂, x)) && return true
        elseif comp isa FluxJump
            (_value_time_dependent(comp.β₁, x) || _value_time_dependent(comp.β₂, x)) && return true
        elseif comp isa RobinJump
            (_value_time_dependent(comp.α, x) || _value_time_dependent(comp.β, x)) && return true
        end
    end
    return false
end

function _interface_diph_rhs_time_dependent(ic::Union{Nothing,InterfaceConditions}, x::SVector{N,T}) where {N,T}
    ic === nothing && return false
    for comp in (ic.scalar, ic.flux)
        comp === nothing && continue
        _value_time_dependent(comp.value, x) && return true
    end
    return false
end

function _mono_matrix_time_dependent(model::DiffusionModelMono{N,T}) where {N,T}
    xω = model.cap.C_ω[1]
    xγ = model.cap.C_γ[1]
    return _coeff_time_dependent(model.D, xω) ||
           _interface_mono_matrix_time_dependent(model.bc_interface, xγ)
end

function _mono_rhs_time_dependent(model::DiffusionModelMono{N,T}) where {N,T}
    xω = model.cap.C_ω[1]
    xγ = model.cap.C_γ[1]
    return _source_mono_time_dependent(model.source, xω) ||
           _border_values_time_dependent(model.bc_border, xω) ||
           _interface_mono_rhs_time_dependent(model.bc_interface, xγ)
end

function _diph_matrix_time_dependent(model::DiffusionModelDiph{N,T}) where {N,T}
    xω1 = model.cap1.C_ω[1]
    xω2 = model.cap2.C_ω[1]
    xγ = model.cap1.C_γ[1]
    return _coeff_time_dependent(model.D1, xω1) ||
        _coeff_time_dependent(model.D2, xω2) ||
        _interface_diph_matrix_time_dependent(model.ic, xγ)
end

function _diph_rhs_time_dependent(model::DiffusionModelDiph{N,T}) where {N,T}
    xω1 = model.cap1.C_ω[1]
    xω2 = model.cap2.C_ω[1]
    xγ = model.cap1.C_γ[1]
    return _source_mono_time_dependent(model.source1, xω1) ||
        _source_mono_time_dependent(model.source2, xω2) ||
        _border_values_time_dependent(model.bc_border, xω1) ||
        _interface_diph_rhs_time_dependent(model.ic, xγ)
end

function _init_unsteady_state_mono(model::DiffusionModelMono{N,T}, u0) where {N,T}
    lay = model.layout.offsets
    nt = model.cap.ntotal
    nsys = maximum((last(lay.ω), last(lay.γ)))
    u = zeros(T, nsys)
    if length(u0) == nsys
        u .= Vector{T}(u0)
    elseif length(u0) == nt
        u[lay.ω] .= Vector{T}(u0)
    else
        throw(DimensionMismatch("u0 length must be $nt (ω block) or $nsys (full system)"))
    end
    return u
end

function _init_unsteady_state_diph(model::DiffusionModelDiph{N,T}, u0) where {N,T}
    lay = model.layout.offsets
    nt = model.cap1.ntotal
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
    u = zeros(T, nsys)
    if length(u0) == nsys
        u .= Vector{T}(u0)
    elseif length(u0) == 2 * nt
        u0v = Vector{T}(u0)
        u[lay.ω1] .= u0v[1:nt]
        u[lay.ω2] .= u0v[(nt + 1):(2 * nt)]
    else
        throw(DimensionMismatch("u0 length must be $(2 * nt) (ω1+ω2) or $nsys (full system)"))
    end
    return u
end

function _prepare_constant_unsteady_mono(model::DiffusionModelMono{N,T}, t0::T, dt::T, θ::T) where {N,T}
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω), last(lay.γ)))
    sys0 = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_steady_mono!(sys0, model, t0 + θ * dt)
    Asteady = sys0.A
    bsteady = copy(sys0.b)
    Aconst = copy(Asteady)
    θ != one(T) && _scale_rows!(Aconst, lay.ω, θ)
    M = model.cap.buf.V ./ dt
    Aconst = Aconst + sparse(lay.ω, lay.ω, M, nsys, nsys)
    Aω_prev = Asteady[lay.ω, :]
    return Aconst, bsteady, Aω_prev, M
end

function _prepare_constant_unsteady_diph(model::DiffusionModelDiph{N,T}, t0::T, dt::T, θ::T) where {N,T}
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
    sys0 = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_steady_diph!(sys0, model, t0 + θ * dt)
    Asteady = sys0.A
    bsteady = copy(sys0.b)
    Aconst = copy(Asteady)
    θ != one(T) && _scale_rows!(Aconst, lay.ω1, θ)
    θ != one(T) && _scale_rows!(Aconst, lay.ω2, θ)
    M1 = model.cap1.buf.V ./ dt
    M2 = model.cap2.buf.V ./ dt
    rows = vcat(collect(lay.ω1), collect(lay.ω2))
    vals = vcat(M1, M2)
    Aconst = Aconst + sparse(rows, rows, vals, nsys, nsys)
    Aω1_prev = Asteady[lay.ω1, :]
    Aω2_prev = Asteady[lay.ω2, :]
    return Aconst, bsteady, Aω1_prev, Aω2_prev, M1, M2
end

function _set_constant_rhs_mono!(
    b::Vector{T},
    bsteady::Vector{T},
    Aω_prev::SparseMatrixCSC{T,Int},
    M::Vector{T},
    lay,
    u::Vector{T},
    θ::T,
) where {T}
    copyto!(b, bsteady)
    if θ != one(T)
        corr = Aω_prev * u
        _insert_vec!(b, lay.ω, (-(one(T) - θ)) .* corr)
    end
    _insert_vec!(b, lay.ω, M .* Vector{T}(u[lay.ω]))
    return b
end

function _set_constant_rhs_diph!(
    b::Vector{T},
    bsteady::Vector{T},
    Aω1_prev::SparseMatrixCSC{T,Int},
    Aω2_prev::SparseMatrixCSC{T,Int},
    M1::Vector{T},
    M2::Vector{T},
    lay,
    u::Vector{T},
    θ::T,
) where {T}
    copyto!(b, bsteady)
    if θ != one(T)
        corr1 = Aω1_prev * u
        corr2 = Aω2_prev * u
        _insert_vec!(b, lay.ω1, (-(one(T) - θ)) .* corr1)
        _insert_vec!(b, lay.ω2, (-(one(T) - θ)) .* corr2)
    end
    _insert_vec!(b, lay.ω1, M1 .* Vector{T}(u[lay.ω1]))
    _insert_vec!(b, lay.ω2, M2 .* Vector{T}(u[lay.ω2]))
    return b
end

function solve_unsteady!(
    model::DiffusionModelMono{N,T},
    u0,
    tspan::Tuple{T,T};
    dt::T,
    scheme=:BE,
    method::Symbol=:direct,
    save_history::Bool=true,
    kwargs...,
) where {N,T}
    t0, tend = tspan
    tend >= t0 || throw(ArgumentError("tspan must satisfy tend >= t0"))
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    θ = _theta_from_scheme(T, scheme)

    u = _init_unsteady_state_mono(model, u0)
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω), last(lay.γ)))

    matrix_dep = _mono_matrix_time_dependent(model)
    rhs_dep = _mono_rhs_time_dependent(model)
    constant_operator = !matrix_dep && !rhs_dep

    times = T[t0]
    states = Vector{Vector{T}}()
    save_history && push!(states, copy(u))

    tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
    t = t0

    if constant_operator
        Aconst, bsteady, Aω_prev, M = _prepare_constant_unsteady_mono(model, t0, dt, θ)
        sys = LinearSystem(Aconst, copy(bsteady); x=copy(u))
        while t < tend - tol
            dt_step = min(dt, tend - t)
            if abs(dt_step - dt) <= tol
                _set_constant_rhs_mono!(sys.b, bsteady, Aω_prev, M, lay, u, θ)
                solve!(sys; method=method, reuse_factorization=true, kwargs...)
            else
                assemble_unsteady_mono!(sys, model, u, t, dt_step, θ)
                solve!(sys; method=method, reuse_factorization=false, kwargs...)
            end
            u .= sys.x
            t += dt_step
            push!(times, t)
            save_history && push!(states, copy(u))
        end
        if !save_history
            states = [copy(u)]
            times = T[t]
        end
        return (times=times, states=states, system=sys, reused_constant_operator=true)
    end

    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys); x=copy(u))
    while t < tend - tol
        dt_step = min(dt, tend - t)
        assemble_unsteady_mono!(sys, model, u, t, dt_step, θ)
        solve!(sys; method=method, reuse_factorization=false, kwargs...)
        u .= sys.x
        t += dt_step
        push!(times, t)
        save_history && push!(states, copy(u))
    end
    if !save_history
        states = [copy(u)]
        times = T[t]
    end
    return (times=times, states=states, system=sys, reused_constant_operator=false)
end

function solve_unsteady!(
    model::DiffusionModelDiph{N,T},
    u0,
    tspan::Tuple{T,T};
    dt::T,
    scheme=:BE,
    method::Symbol=:direct,
    save_history::Bool=true,
    kwargs...,
) where {N,T}
    t0, tend = tspan
    tend >= t0 || throw(ArgumentError("tspan must satisfy tend >= t0"))
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    θ = _theta_from_scheme(T, scheme)

    u = _init_unsteady_state_diph(model, u0)
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

    matrix_dep = _diph_matrix_time_dependent(model)
    rhs_dep = _diph_rhs_time_dependent(model)
    constant_operator = !matrix_dep && !rhs_dep

    times = T[t0]
    states = Vector{Vector{T}}()
    save_history && push!(states, copy(u))

    tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
    t = t0

    if constant_operator
        Aconst, bsteady, Aω1_prev, Aω2_prev, M1, M2 = _prepare_constant_unsteady_diph(model, t0, dt, θ)
        sys = LinearSystem(Aconst, copy(bsteady); x=copy(u))
        while t < tend - tol
            dt_step = min(dt, tend - t)
            if abs(dt_step - dt) <= tol
                _set_constant_rhs_diph!(sys.b, bsteady, Aω1_prev, Aω2_prev, M1, M2, lay, u, θ)
                solve!(sys; method=method, reuse_factorization=true, kwargs...)
            else
                assemble_unsteady_diph!(sys, model, u, t, dt_step, θ)
                solve!(sys; method=method, reuse_factorization=false, kwargs...)
            end
            u .= sys.x
            t += dt_step
            push!(times, t)
            save_history && push!(states, copy(u))
        end
        if !save_history
            states = [copy(u)]
            times = T[t]
        end
        return (times=times, states=states, system=sys, reused_constant_operator=true)
    end

    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys); x=copy(u))
    while t < tend - tol
        dt_step = min(dt, tend - t)
        assemble_unsteady_diph!(sys, model, u, t, dt_step, θ)
        solve!(sys; method=method, reuse_factorization=false, kwargs...)
        u .= sys.x
        t += dt_step
        push!(times, t)
        save_history && push!(states, copy(u))
    end
    if !save_history
        states = [copy(u)]
        times = T[t]
    end
    return (times=times, states=states, system=sys, reused_constant_operator=false)
end

end
