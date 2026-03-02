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
export solve_steady!

struct DiffusionModelMono{N,T,DT,ST,IT}
    ops::DiffusionOps{N,T}
    cap::AssembledCapacity{N,T}
    D::DT
    source::ST
    bc_border::BorderConditions
    bc_interface::IT
    layout::UnknownLayout
end

function DiffusionModelMono(
    cap::AssembledCapacity{N,T},
    ops::DiffusionOps{N,T},
    D;
    source=((args...) -> zero(T)),
    bc_border::BorderConditions=BorderConditions(),
    bc_interface::Union{Nothing,PenguinBCs.Robin}=nothing,
    layout::UnknownLayout=layout_mono(cap.ntotal),
) where {N,T}
    return DiffusionModelMono{N,T,typeof(D),typeof(source),typeof(bc_interface)}(
        ops, cap, D, source, bc_border, bc_interface, layout
    )
end

struct DiffusionModelDiph{N,T,D1T,D2T,ST,IT}
    ops::DiffusionOps{N,T}
    cap::AssembledCapacity{N,T}
    D1::D1T
    D2::D2T
    source::ST
    bc_border::BorderConditions
    bc_interface::IT
    layout::UnknownLayout
end

function DiffusionModelDiph(
    cap::AssembledCapacity{N,T},
    ops::DiffusionOps{N,T},
    D1,
    D2;
    source=((args...) -> (zero(T), zero(T))),
    bc_border::BorderConditions=BorderConditions(),
    bc_interface::Union{Nothing,InterfaceConditions}=nothing,
    layout::UnknownLayout=layout_diph(cap.ntotal),
) where {N,T}
    return DiffusionModelDiph{N,T,typeof(D1),typeof(D2),typeof(source),typeof(bc_interface)}(
        ops, cap, D1, D2, source, bc_border, bc_interface, layout
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

function _source_values_diph(cap::AssembledCapacity{N,T}, source, t::T) where {N,T}
    f1 = Vector{T}(undef, cap.ntotal)
    f2 = Vector{T}(undef, cap.ntotal)
    @inbounds for i in eachindex(f1)
        x = cap.C_ω[i]
        if source isa Function
            s = applicable(source, x..., t) ? source(x..., t) : source(x...)
            f1[i] = convert(T, s[1])
            f2[i] = convert(T, s[2])
        elseif source isa Tuple && length(source) == 2
            f1[i] = _eval_fun_or_const(source[1], x, t)
            f2[i] = _eval_fun_or_const(source[2], x, t)
        else
            throw(ArgumentError("diph source must be a function returning a tuple or a tuple of two callbacks/constants"))
        end
    end
    return f1, f2
end

function _sample_coeff(cap::AssembledCapacity{N,T}, D, t::T) where {N,T}
    out = Vector{T}(undef, cap.ntotal)
    @inbounds for i in eachindex(out)
        out[i] = convert(T, eval_coeff(D, cap.C_ω[i], t, i))
    end
    return out
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

function _interface_diagonals_diph(cap::AssembledCapacity{N,T}, ic::Union{Nothing,InterfaceConditions}, t::T) where {N,T}
    α1 = zeros(T, cap.ntotal)
    α2 = zeros(T, cap.ntotal)
    β1 = zeros(T, cap.ntotal)
    β2 = zeros(T, cap.ntotal)
    g = zeros(T, cap.ntotal)
    ic === nothing && return α1, α2, β1, β2, g

    mask = _interface_mask(cap)
    @inbounds for i in eachindex(mask)
        mask[i] || continue
        x = cap.C_γ[i]
        if ic.scalar isa ScalarJump
            α1[i] += convert(T, eval_bc(ic.scalar.α₁, x, t))
            α2[i] += convert(T, eval_bc(ic.scalar.α₂, x, t))
            g[i] = convert(T, eval_bc(ic.scalar.value, x, t))
        elseif ic.scalar isa RobinJump
            α1[i] += convert(T, eval_bc(ic.scalar.α, x, t))
            α2[i] += convert(T, eval_bc(ic.scalar.α, x, t))
            β1[i] += convert(T, eval_bc(ic.scalar.β, x, t))
            β2[i] += convert(T, eval_bc(ic.scalar.β, x, t))
            g[i] = convert(T, eval_bc(ic.scalar.value, x, t))
        elseif !(ic.scalar === nothing)
            throw(ArgumentError("unsupported scalar interface condition type $(typeof(ic.scalar))"))
        end

        if ic.flux isa FluxJump
            β1[i] += convert(T, eval_bc(ic.flux.β₁, x, t))
            β2[i] += convert(T, eval_bc(ic.flux.β₂, x, t))
            if ic.scalar === nothing
                g[i] = convert(T, eval_bc(ic.flux.value, x, t))
            end
        elseif ic.flux isa RobinJump
            α1[i] += convert(T, eval_bc(ic.flux.α, x, t))
            α2[i] += convert(T, eval_bc(ic.flux.α, x, t))
            β1[i] += convert(T, eval_bc(ic.flux.β, x, t))
            β2[i] += convert(T, eval_bc(ic.flux.β, x, t))
            if ic.scalar === nothing
                g[i] = convert(T, eval_bc(ic.flux.value, x, t))
            end
        elseif !(ic.flux === nothing)
            throw(ArgumentError("unsupported flux interface condition type $(typeof(ic.flux))"))
        end
    end
    return α1, α2, β1, β2, g
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

function _diph_row_activity(cap::AssembledCapacity{N,T}, lay) where {N,T}
    activeω, activeγ = _cell_activity_masks(cap)
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))
    active = falses(nsys)
    @inbounds for i in 1:cap.ntotal
        aω = activeω[i]
        aγ = activeγ[i]
        active[lay.ω1[i]] = aω
        active[lay.γ1[i]] = aγ
        active[lay.ω2[i]] = aω
        active[lay.γ2[i]] = aγ
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

function _core_ops(model)
    G = model.ops.G
    H = model.ops.H
    Winv = model.ops.Winv
    K = G' * Winv * G
    C = G' * Winv * H
    J = H' * Winv * G
    L = H' * Winv * H
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

    K, C, J, L = _core_ops(model)
    Dω = _sample_coeff(model.cap, model.D, t)
    fω = _source_values_mono(model.cap, model.source, t)
    α, β, gγ = _interface_diagonals_mono(model.cap, model.bc_interface, t)

    ID = spdiagm(0 => Dω)
    Iβ = spdiagm(0 => β)
    Iα = spdiagm(0 => α)
    Iγ = model.cap.Γ

    A11 = ID * K
    A12 = ID * C
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
    nt = model.cap.ntotal
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

    K, C, J, L = _core_ops(model)
    D1 = _sample_coeff(model.cap, model.D1, t)
    D2 = _sample_coeff(model.cap, model.D2, t)
    f1, f2 = _source_values_diph(model.cap, model.source, t)
    α1, α2, β1, β2, gγ = _interface_diagonals_diph(model.cap, model.bc_interface, t)

    ID1 = spdiagm(0 => D1)
    ID2 = spdiagm(0 => D2)
    Iβ1 = spdiagm(0 => β1)
    Iβ2 = spdiagm(0 => β2)
    Iα1 = spdiagm(0 => α1)
    Iα2 = spdiagm(0 => α2)
    Iγ = model.cap.Γ

    Aωω1 = ID1 * K
    Aωγ1 = ID1 * C
    Aγω1 = Iβ1 * J
    Aγγ1 = Iβ1 * L + Iα1 * Iγ

    Aωω2 = ID2 * K
    Aωγ2 = ID2 * C
    Aγω2 = Iβ2 * J
    Aγγ2 = Iβ2 * L + Iα2 * Iγ
    if model.bc_interface === nothing
        Aωγ1 = spzeros(T, nt, nt)
        Aγω1 = spzeros(T, nt, nt)
        Aγγ1 = spdiagm(0 => ones(T, nt))
        Aωγ2 = spzeros(T, nt, nt)
        Aγω2 = spzeros(T, nt, nt)
        Aγγ2 = spdiagm(0 => ones(T, nt))
    end

    bω1 = model.cap.V * f1
    bγ1 = Iγ * gγ
    bω2 = model.cap.V * f2
    bγ2 = Iγ * gγ

    A, b = if _is_canonical_diph_layout(lay, nt)
        Aphase1 = [Aωω1 Aωγ1; Aγω1 Aγγ1]
        Aphase2 = [Aωω2 Aωγ2; Aγω2 Aγγ2]
        (blockdiag(Aphase1, Aphase2), vcat(bω1, bγ1, bω2, bγ2))
    else
        Awork = spzeros(T, nsys, nsys)
        bwork = zeros(T, nsys)
        _insert_block!(Awork, lay.ω1, lay.ω1, Aωω1)
        _insert_block!(Awork, lay.ω1, lay.γ1, Aωγ1)
        _insert_block!(Awork, lay.γ1, lay.ω1, Aγω1)
        _insert_block!(Awork, lay.γ1, lay.γ1, Aγγ1)

        _insert_block!(Awork, lay.ω2, lay.ω2, Aωω2)
        _insert_block!(Awork, lay.ω2, lay.γ2, Aωγ2)
        _insert_block!(Awork, lay.γ2, lay.ω2, Aγω2)
        _insert_block!(Awork, lay.γ2, lay.γ2, Aγγ2)

        _insert_vec!(bwork, lay.ω1, bω1)
        _insert_vec!(bwork, lay.γ1, bγ1)
        _insert_vec!(bwork, lay.ω2, bω2)
        _insert_vec!(bwork, lay.γ2, bγ2)
        (Awork, bwork)
    end

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing

    layω1 = UnknownLayout(model.cap.ntotal, (ω=lay.ω1,))
    layω2 = UnknownLayout(model.cap.ntotal, (ω=lay.ω2,))
    apply_box_bc_mono!(sys.A, sys.b, model.cap, model.ops, model.D1, model.bc_border; t=t, layout=layω1)
    apply_box_bc_mono!(sys.A, sys.b, model.cap, model.ops, model.D2, model.bc_border; t=t, layout=layω2)
    active_rows = _diph_row_activity(model.cap, lay)
    sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
    return sys
end

function assemble_unsteady_diph!(sys::LinearSystem{T}, model::DiffusionModelDiph{N,T}, uⁿ, t::T, dt::T, scheme) where {N,T}
    θ = scheme isa Real ? convert(T, scheme) : one(T)
    assemble_steady_diph!(sys, model, t + θ * dt)

    lay = model.layout.offsets
    nt = model.cap.ntotal
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

    M = model.cap.buf.V ./ dt
    rows = vcat(collect(lay.ω1), collect(lay.ω2))
    vals = vcat(M, M)
    sys.A = sys.A + sparse(rows, rows, vals, nsys, nsys)

    u1 = Vector{T}(ufull[lay.ω1])
    u2 = Vector{T}(ufull[lay.ω2])
    _insert_vec!(sys.b, lay.ω1, M .* u1)
    _insert_vec!(sys.b, lay.ω2, M .* u2)
    active_rows = _diph_row_activity(model.cap, lay)
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

end
