module PenguinDiffusion

using LinearAlgebra
using SparseArrays
using StaticArrays

using CartesianGeometry: GeometricMoments, geometric_moments, nan
using CartesianGrids: CartesianGrid, SpaceTimeCartesianGrid, grid1d
using CartesianOperators
using PenguinBCs
using PenguinSolverCore

export DiffusionModelMono, DiffusionModelDiph
export MovingDiffusionModelMono, MovingDiffusionModelDiph
export assemble_steady_mono!, assemble_unsteady_mono!
export assemble_steady_diph!, assemble_unsteady_diph!
export assemble_unsteady_mono_moving!, assemble_unsteady_diph_moving!
export solve_steady!, solve_unsteady!, solve_unsteady_moving!
export compute_interface_exchange_metrics

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

mutable struct MovingDiffusionModelMono{N,T,DT,ST,IT,BT}
    grid::CartesianGrid{N,T}
    body::BT
    D::DT
    source::ST
    bc_border::BorderConditions
    bc_interface::IT
    layout::UnknownLayout
    coeff_mode::Symbol
    geom_method::Symbol
    cap_slab::Union{Nothing,AssembledCapacity{N,T}}
    ops_slab::Union{Nothing,DiffusionOps{N,T}}
    Vn::Vector{T}
    Vn1::Vector{T}
end

mutable struct MovingDiffusionModelDiph{N,T,D1T,D2T,S1T,S2T,IT,B1T,B2T}
    grid::CartesianGrid{N,T}
    body1::B1T
    body2::B2T
    D1::D1T
    source1::S1T
    D2::D2T
    source2::S2T
    bc_border::BorderConditions
    ic::IT
    layout::UnknownLayout
    coeff_mode::Symbol
    geom_method::Symbol
    cap1_slab::Union{Nothing,AssembledCapacity{N,T}}
    ops1_slab::Union{Nothing,DiffusionOps{N,T}}
    cap2_slab::Union{Nothing,AssembledCapacity{N,T}}
    ops2_slab::Union{Nothing,DiffusionOps{N,T}}
    V1n::Vector{T}
    V1n1::Vector{T}
    V2n::Vector{T}
    V2n1::Vector{T}
end

function MovingDiffusionModelMono(
    grid::CartesianGrid{N,T},
    body,
    D;
    source=((args...) -> zero(T)),
    bc_border::BorderConditions=BorderConditions(),
    bc_interface::Union{Nothing,PenguinBCs.Robin}=nothing,
    layout::UnknownLayout=layout_mono(prod(grid.n)),
    coeff_mode::Symbol=:harmonic,
    geom_method::Symbol=:vofijul,
) where {N,T}
    coeff_mode_eff = _normalize_coeff_mode(coeff_mode)
    nt = prod(grid.n)
    return MovingDiffusionModelMono{
        N,T,typeof(D),typeof(source),typeof(bc_interface),typeof(body)
    }(
        grid,
        body,
        D,
        source,
        bc_border,
        bc_interface,
        layout,
        coeff_mode_eff,
        geom_method,
        nothing,
        nothing,
        zeros(T, nt),
        zeros(T, nt),
    )
end

function MovingDiffusionModelDiph(
    grid::CartesianGrid{N,T},
    body1,
    D1,
    D2;
    source=((args...) -> (zero(T), zero(T))),
    body2=nothing,
    bc_border::BorderConditions=BorderConditions(),
    ic::Union{Nothing,InterfaceConditions}=nothing,
    bc_interface::Union{Nothing,InterfaceConditions}=nothing,
    layout::UnknownLayout=layout_diph(prod(grid.n)),
    coeff_mode::Symbol=:harmonic,
    geom_method::Symbol=:vofijul,
) where {N,T}
    if !(ic === nothing) && !(bc_interface === nothing) && (ic !== bc_interface)
        throw(ArgumentError("provide at most one interface condition via `ic` or `bc_interface`"))
    end
    ic_eff = ic === nothing ? bc_interface : ic
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

    coeff_mode_eff = _normalize_coeff_mode(coeff_mode)
    nt = prod(grid.n)
    return MovingDiffusionModelDiph{
        N,T,typeof(D1),typeof(D2),typeof(source1),typeof(source2),typeof(ic_eff),typeof(body1),typeof(body2)
    }(
        grid,
        body1,
        body2,
        D1,
        source1,
        D2,
        source2,
        bc_border,
        ic_eff,
        layout,
        coeff_mode_eff,
        geom_method,
        nothing,
        nothing,
        nothing,
        nothing,
        zeros(T, nt),
        zeros(T, nt),
        zeros(T, nt),
        zeros(T, nt),
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

psip_cn(Vn, Vn1) = (iszero(Vn) && iszero(Vn1)) ? 0.0 : 0.5
psim_cn(Vn, Vn1) = (iszero(Vn) && iszero(Vn1)) ? 0.0 : 0.5
psip_be(Vn, Vn1) = (iszero(Vn) && iszero(Vn1)) ? 0.0 : 1.0
psim_be(Vn, Vn1) = 0.0

function _psi_functions(scheme)
    if scheme isa Symbol
        if scheme === :CN
            return psip_cn, psim_cn
        elseif scheme === :BE
            return psip_be, psim_be
        end
    elseif scheme isa Real
        θ = Float64(scheme)
        psip = (Vn, Vn1) -> (iszero(Vn) && iszero(Vn1) ? 0.0 : θ)
        psim = (Vn, Vn1) -> (iszero(Vn) && iszero(Vn1) ? 0.0 : (1.0 - θ))
        return psip, psim
    end
    throw(ArgumentError("moving scheme must be :BE, :CN, or numeric θ"))
end

function _eval_levelset_time(body, x::SVector{N,T}, t::T) where {N,T}
    if applicable(body, x..., t)
        return convert(T, body(x..., t))
    elseif applicable(body, x...)
        return convert(T, body(x...))
    end
    throw(ArgumentError("level-set callback must accept (x...) or (x..., t)"))
end

function _space_moments_at_time(
    model::MovingDiffusionModelMono{N,T},
    xyz_space::NTuple{N,AbstractVector{T}},
    t::T,
) where {N,T}
    body_t = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t)
    return geometric_moments(body_t, xyz_space, T, nan; method=model.geom_method)
end

function _phase_levelset_value(model::MovingDiffusionModelDiph{N,T}, phase::Int, x::SVector{N,T}, t::T) where {N,T}
    if phase == 1
        return _eval_levelset_time(model.body1, x, t)
    elseif phase == 2
        if model.body2 === nothing
            return -_eval_levelset_time(model.body1, x, t)
        end
        return _eval_levelset_time(model.body2, x, t)
    end
    throw(ArgumentError("phase must be 1 or 2"))
end

function _space_moments_at_time(
    model::MovingDiffusionModelDiph{N,T},
    xyz_space::NTuple{N,AbstractVector{T}},
    t::T,
    phase::Int,
) where {N,T}
    body_t = (x...) -> _phase_levelset_value(model, phase, SVector{N,T}(x), t)
    return geometric_moments(body_t, xyz_space, T, nan; method=model.geom_method)
end

function _slice_spacetime_to_space(
    vec_st::AbstractVector,
    nn_space::NTuple{N,Int},
    nt::Int,
    it::Int,
) where {N}
    dims_st = (nn_space..., nt)
    li_st = LinearIndices(dims_st)
    li_sp = LinearIndices(nn_space)
    out = similar(vec_st, prod(nn_space))
    @inbounds for I in CartesianIndices(nn_space)
        out[li_sp[I]] = vec_st[li_st[Tuple(I)..., it]]
    end
    return out
end

function reduce_slab_to_space(
    m_st::GeometricMoments{N1,T},
    nn_space::NTuple{N,Int},
) where {N1,N,T}
    N1 == N + 1 || throw(ArgumentError("expected slab moments dimension $(N + 1), got $N1"))
    nt = length(m_st.xyz[N1])
    nt == 2 || throw(ArgumentError("space-time reduction expects 2 time nodes, got $nt"))

    V = _slice_spacetime_to_space(m_st.V, nn_space, nt, 1)
    Γ = _slice_spacetime_to_space(m_st.interface_measure, nn_space, nt, 1)
    ctype = _slice_spacetime_to_space(m_st.cell_type, nn_space, nt, 1)
    A = ntuple(d -> _slice_spacetime_to_space(m_st.A[d], nn_space, nt, 1), N)
    B = ntuple(d -> _slice_spacetime_to_space(m_st.B[d], nn_space, nt, 1), N)
    W = ntuple(d -> _slice_spacetime_to_space(m_st.W[d], nn_space, nt, 1), N)

    bary_st = _slice_spacetime_to_space(m_st.barycenter, nn_space, nt, 1)
    baryγ_st = _slice_spacetime_to_space(m_st.barycenter_interface, nn_space, nt, 1)
    nγ_st = _slice_spacetime_to_space(m_st.interface_normal, nn_space, nt, 1)

    bary = Vector{SVector{N,T}}(undef, length(V))
    baryγ = Vector{SVector{N,T}}(undef, length(V))
    nγ = Vector{SVector{N,T}}(undef, length(V))
    @inbounds for i in eachindex(V)
        bi = bary_st[i]
        bγi = baryγ_st[i]
        ni = nγ_st[i]
        bary[i] = SVector{N,T}(ntuple(d -> bi[d], N))
        baryγ[i] = SVector{N,T}(ntuple(d -> bγi[d], N))
        nγ[i] = SVector{N,T}(ntuple(d -> ni[d], N))
    end

    xyz = ntuple(d -> collect(T, m_st.xyz[d]), N)
    return GeometricMoments(V, bary, Γ, ctype, baryγ, nγ, A, B, W, xyz)
end

function _build_moving_slab!(
    model::MovingDiffusionModelMono{N,T},
    t::T,
    dt::T,
) where {N,T}
    xyz_space = grid1d(model.grid)
    moms_n = _space_moments_at_time(model, xyz_space, t)
    moms_n1 = _space_moments_at_time(model, xyz_space, t + dt)

    stgrid = SpaceTimeCartesianGrid(model.grid, T[t, t + dt])
    xyz_st = grid1d(stgrid)
    body_st = (x...) -> begin
        xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
        _eval_levelset_time(model.body, xs, convert(T, x[N + 1]))
    end
    moms_st = geometric_moments(body_st, xyz_st, T, nan; method=model.geom_method)
    moms_slab = reduce_slab_to_space(moms_st, model.grid.n)
    cap_slab = assembled_capacity(moms_slab; bc=zero(T))
    pflags = periodic_flags(model.bc_border, N)
    ops_slab = DiffusionOps(cap_slab; periodic=pflags)

    model.cap_slab = cap_slab
    model.ops_slab = ops_slab
    model.Vn .= moms_n.V
    model.Vn1 .= moms_n1.V
    return model
end

function _build_moving_slab!(
    model::MovingDiffusionModelDiph{N,T},
    t::T,
    dt::T,
) where {N,T}
    xyz_space = grid1d(model.grid)
    moms1_n = _space_moments_at_time(model, xyz_space, t, 1)
    moms1_n1 = _space_moments_at_time(model, xyz_space, t + dt, 1)
    moms2_n = _space_moments_at_time(model, xyz_space, t, 2)
    moms2_n1 = _space_moments_at_time(model, xyz_space, t + dt, 2)

    stgrid = SpaceTimeCartesianGrid(model.grid, T[t, t + dt])
    xyz_st = grid1d(stgrid)
    body1_st = (x...) -> begin
        xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
        _phase_levelset_value(model, 1, xs, convert(T, x[N + 1]))
    end
    body2_st = (x...) -> begin
        xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
        _phase_levelset_value(model, 2, xs, convert(T, x[N + 1]))
    end

    moms1_st = geometric_moments(body1_st, xyz_st, T, nan; method=model.geom_method)
    moms2_st = geometric_moments(body2_st, xyz_st, T, nan; method=model.geom_method)
    moms1_slab = reduce_slab_to_space(moms1_st, model.grid.n)
    moms2_slab = reduce_slab_to_space(moms2_st, model.grid.n)

    cap1_slab = assembled_capacity(moms1_slab; bc=zero(T))
    cap2_slab = assembled_capacity(moms2_slab; bc=zero(T))
    pflags = periodic_flags(model.bc_border, N)
    ops1_slab = DiffusionOps(cap1_slab; periodic=pflags)
    ops2_slab = DiffusionOps(cap2_slab; periodic=pflags)

    model.cap1_slab = cap1_slab
    model.ops1_slab = ops1_slab
    model.cap2_slab = cap2_slab
    model.ops2_slab = ops2_slab
    model.V1n .= moms1_n.V
    model.V1n1 .= moms1_n1.V
    model.V2n .= moms2_n.V
    model.V2n1 .= moms2_n1.V
    return model
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

function _resolve_diffusivity_scale(diffusivity_scale, D, ::Type{T}) where {T}
    if diffusivity_scale === nothing
        D isa Number || throw(ArgumentError("diffusivity_scale must be provided when diffusivity is not a scalar number"))
        return convert(T, D)
    end
    diffusivity_scale isa Number || throw(ArgumentError("diffusivity_scale must be a scalar number"))
    return convert(T, diffusivity_scale)
end

function _resolve_diph_diffusivity_scales(diffusivity_scale, D1, D2, ::Type{T}) where {T}
    if diffusivity_scale === nothing
        (D1 isa Number && D2 isa Number) ||
            throw(ArgumentError("diffusivity_scale=(d1,d2) must be provided when phase diffusivities are not scalar numbers"))
        return convert(T, D1), convert(T, D2)
    end
    (diffusivity_scale isa Tuple && length(diffusivity_scale) == 2) ||
        throw(ArgumentError("diffusivity_scale must be a 2-tuple `(d1, d2)` for diphasic models"))
    d1, d2 = diffusivity_scale
    (d1 isa Number && d2 isa Number) ||
        throw(ArgumentError("diffusivity_scale entries must be scalar numbers"))
    return convert(T, d1), convert(T, d2)
end

function _resolve_diph_reference_values(reference_value, ::Type{T}) where {T}
    if reference_value isa Tuple
        length(reference_value) == 2 || throw(ArgumentError("reference_value tuple must have length 2"))
        return convert(T, reference_value[1]), convert(T, reference_value[2])
    end
    return convert(T, reference_value), convert(T, reference_value)
end

function _phase_exchange_metrics(
    cap::AssembledCapacity{N,T},
    ops::DiffusionOps{N,T},
    uω::AbstractVector{T},
    uγ::AbstractVector{T},
    diffusivity_scale::T,
    characteristic_scale::T,
    reference_value::T,
) where {N,T}
    nt = cap.ntotal
    length(uω) == nt || throw(DimensionMismatch("length(uω) must be $nt"))
    length(uγ) == nt || throw(DimensionMismatch("length(uγ) must be $nt"))
    characteristic_scale > zero(T) || throw(ArgumentError("characteristic_scale must be positive"))
    iszero(diffusivity_scale) && throw(ArgumentError("diffusivity_scale must be non-zero"))

    # q_nds is the per-interface-cell integrated normal gradient contribution.
    q_nds = ops.H' * (ops.Winv * (ops.G * uω + ops.H * uγ))
    Γ = cap.buf.Γ

    interface_measure = zero(T)
    integrated_normal_gradient = zero(T)
    weighted_interface_value = zero(T)
    @inbounds for i in eachindex(Γ)
        γ = Γ[i]
        qi = q_nds[i]
        ui = uγ[i]
        if isfinite(γ) && γ > zero(T) && isfinite(qi) && isfinite(ui)
            interface_measure += γ
            integrated_normal_gradient += qi
            weighted_interface_value += ui * γ
        end
    end
    interface_measure > zero(T) || throw(ArgumentError("no active interface cells found in provided capacity"))

    integrated_normal_flux = -diffusivity_scale * integrated_normal_gradient
    mean_normal_gradient = integrated_normal_gradient / interface_measure
    mean_normal_flux = integrated_normal_flux / interface_measure
    mean_interface_value = weighted_interface_value / interface_measure

    Δref = mean_interface_value - reference_value
    exchange_coefficient = iszero(Δref) ? T(NaN) : (mean_normal_flux / Δref)
    transfer_index = exchange_coefficient * characteristic_scale / diffusivity_scale

    return (
        interface_measure=interface_measure,
        integrated_normal_gradient=integrated_normal_gradient,
        integrated_normal_flux=integrated_normal_flux,
        mean_normal_gradient=mean_normal_gradient,
        mean_normal_flux=mean_normal_flux,
        mean_interface_value=mean_interface_value,
        reference_value=reference_value,
        exchange_coefficient=exchange_coefficient,
        transfer_index=transfer_index,
    )
end

function _extract_mono_state(
    state::AbstractVector,
    lay,
    ::Type{T},
) where {T}
    length(state) >= last(lay.γ) || throw(DimensionMismatch("state vector does not contain the mono γ block"))
    return Vector{T}(state[lay.ω]), Vector{T}(state[lay.γ])
end

function _extract_diph_state(
    state::AbstractVector,
    lay,
    ::Type{T},
) where {T}
    length(state) >= last(lay.γ2) || throw(DimensionMismatch("state vector does not contain the diph γ2 block"))
    uω1 = Vector{T}(state[lay.ω1])
    uγ1 = Vector{T}(state[lay.γ1])
    uω2 = Vector{T}(state[lay.ω2])
    uγ2 = Vector{T}(state[lay.γ2])
    return uω1, uγ1, uω2, uγ2
end

"""
    compute_interface_exchange_metrics(model, state; diffusivity_scale=nothing, characteristic_scale=1, reference_value=0)

Compute generic interface-transfer diagnostics from a solved state:
- integrated and mean normal gradient
- integrated and mean normal diffusive flux
- mean interface value
- an exchange coefficient relative to `reference_value`
- a generic dimensionless transfer index (`exchange_coefficient * characteristic_scale / diffusivity_scale`)

For diphasic models, returns `(phase1=..., phase2=..., flux_balance=...)`.
"""
function compute_interface_exchange_metrics(
    model::DiffusionModelMono{N,T},
    state::AbstractVector;
    diffusivity_scale=nothing,
    characteristic_scale::Real=one(T),
    reference_value=zero(T),
) where {N,T}
    dscale = _resolve_diffusivity_scale(diffusivity_scale, model.D, T)
    L = convert(T, characteristic_scale)
    cref = convert(T, reference_value)
    lay = model.layout.offsets
    uω, uγ = _extract_mono_state(state, lay, T)
    return _phase_exchange_metrics(model.cap, model.ops, uω, uγ, dscale, L, cref)
end

function compute_interface_exchange_metrics(
    model::MovingDiffusionModelMono{N,T},
    state::AbstractVector;
    diffusivity_scale=nothing,
    characteristic_scale::Real=one(T),
    reference_value=zero(T),
) where {N,T}
    cap = model.cap_slab
    cap === nothing && throw(ArgumentError("moving mono model has no slab capacity; assemble at least one moving step first"))
    ops = model.ops_slab
    ops === nothing && throw(ArgumentError("moving mono model has no slab operators; assemble at least one moving step first"))
    dscale = _resolve_diffusivity_scale(diffusivity_scale, model.D, T)
    L = convert(T, characteristic_scale)
    cref = convert(T, reference_value)
    lay = model.layout.offsets
    uω, uγ = _extract_mono_state(state, lay, T)
    return _phase_exchange_metrics(cap, ops, uω, uγ, dscale, L, cref)
end

function compute_interface_exchange_metrics(
    model::DiffusionModelDiph{N,T},
    state::AbstractVector;
    diffusivity_scale=nothing,
    characteristic_scale::Real=one(T),
    reference_value=zero(T),
) where {N,T}
    d1, d2 = _resolve_diph_diffusivity_scales(diffusivity_scale, model.D1, model.D2, T)
    c1, c2 = _resolve_diph_reference_values(reference_value, T)
    L = convert(T, characteristic_scale)
    lay = model.layout.offsets
    uω1, uγ1, uω2, uγ2 = _extract_diph_state(state, lay, T)

    m1 = _phase_exchange_metrics(model.cap1, model.ops1, uω1, uγ1, d1, L, c1)
    m2 = _phase_exchange_metrics(model.cap2, model.ops2, uω2, uγ2, d2, L, c2)
    return (phase1=m1, phase2=m2, flux_balance=(m1.integrated_normal_flux + m2.integrated_normal_flux))
end

function compute_interface_exchange_metrics(
    model::MovingDiffusionModelDiph{N,T},
    state::AbstractVector;
    diffusivity_scale=nothing,
    characteristic_scale::Real=one(T),
    reference_value=zero(T),
) where {N,T}
    cap1 = model.cap1_slab
    cap1 === nothing && throw(ArgumentError("moving diph model has no phase-1 slab capacity; assemble at least one moving step first"))
    cap2 = model.cap2_slab
    cap2 === nothing && throw(ArgumentError("moving diph model has no phase-2 slab capacity; assemble at least one moving step first"))
    ops1 = model.ops1_slab
    ops1 === nothing && throw(ArgumentError("moving diph model has no phase-1 slab operators; assemble at least one moving step first"))
    ops2 = model.ops2_slab
    ops2 === nothing && throw(ArgumentError("moving diph model has no phase-2 slab operators; assemble at least one moving step first"))
    d1, d2 = _resolve_diph_diffusivity_scales(diffusivity_scale, model.D1, model.D2, T)
    c1, c2 = _resolve_diph_reference_values(reference_value, T)
    L = convert(T, characteristic_scale)
    lay = model.layout.offsets
    uω1, uγ1, uω2, uγ2 = _extract_diph_state(state, lay, T)

    m1 = _phase_exchange_metrics(cap1, ops1, uω1, uγ1, d1, L, c1)
    m2 = _phase_exchange_metrics(cap2, ops2, uω2, uγ2, d2, L, c2)
    return (phase1=m1, phase2=m2, flux_balance=(m1.integrated_normal_flux + m2.integrated_normal_flux))
end

compute_interface_exchange_metrics(model, sys::LinearSystem; kwargs...) =
    compute_interface_exchange_metrics(model, sys.x; kwargs...)

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

function assemble_unsteady_mono_moving!(
    sys::LinearSystem{T},
    model::MovingDiffusionModelMono{N,T},
    uⁿ,
    t::T,
    dt::T;
    scheme=:CN,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    θ = _theta_from_scheme(T, scheme)
    psip, psim = _psi_functions(scheme)

    _build_moving_slab!(model, t, dt)
    cap = something(model.cap_slab)
    ops = something(model.ops_slab)

    nt = cap.ntotal
    lay = model.layout.offsets
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
        if length(uⁿ) >= last(lay.γ)
            v[lay.γ] .= Vector{T}(uⁿ[lay.γ])
        end
        v
    end

    K, C, J, L = _weighted_core_ops(cap, ops, model.D, t + θ * dt, model.coeff_mode)
    α, β, gγ_n1 = _interface_diagonals_mono(cap, model.bc_interface, t + dt)
    _, _, gγ_n = _interface_diagonals_mono(cap, model.bc_interface, t)
    fω_n = _source_values_mono(cap, model.source, t)
    fω_n1 = _source_values_mono(cap, model.source, t + dt)

    M1 = spdiagm(0 => model.Vn1)
    M0 = spdiagm(0 => model.Vn)
    Ψp = spdiagm(0 => T[psip(model.Vn[i], model.Vn1[i]) for i in 1:nt])
    Ψm = spdiagm(0 => T[psim(model.Vn[i], model.Vn1[i]) for i in 1:nt])

    Iβ = spdiagm(0 => β)
    Iα = spdiagm(0 => α)
    Iγ = cap.Γ

    A11 = M1 + θ * (K * Ψp)
    A12 = -(M1 - M0) + θ * (C * Ψp)
    A21 = Iβ * J * Ψp
    A22 = Iβ * L * Ψp + Iα * Iγ
    if model.bc_interface === nothing
        A12 = spzeros(T, nt, nt)
        A21 = spzeros(T, nt, nt)
        A22 = spdiagm(0 => ones(T, nt))
    end

    uω = Vector{T}(ufull[lay.ω])
    uγ = Vector{T}(ufull[lay.γ])
    bω = (M0 - (one(T) - θ) * (K * Ψm)) * uω
    bω .-= (one(T) - θ) .* ((C * Ψm) * uγ)
    bω .+= θ .* (cap.V * fω_n1) .+ (one(T) - θ) .* (cap.V * fω_n)
    bγ = θ .* Iγ * gγ_n1 .+ (one(T) - θ) .* Iγ * gγ_n

    A, b = if _is_canonical_mono_layout(lay, nt)
        ([A11 A12; A21 A22], vcat(bω, bγ))
    else
        Awork = spzeros(T, nsys, nsys)
        bwork = zeros(T, nsys)
        _insert_block!(Awork, lay.ω, lay.ω, A11)
        _insert_block!(Awork, lay.ω, lay.γ, A12)
        _insert_block!(Awork, lay.γ, lay.ω, A21)
        _insert_block!(Awork, lay.γ, lay.γ, A22)
        _insert_vec!(bwork, lay.ω, bω)
        _insert_vec!(bwork, lay.γ, bγ)
        (Awork, bwork)
    end

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing

    apply_box_bc_mono!(sys.A, sys.b, cap, ops, model.D, model.bc_border; t=t + θ * dt, layout=model.layout)
    active_rows = _mono_row_activity(cap, lay)
    sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
    return sys
end

function assemble_unsteady_diph_moving!(
    sys::LinearSystem{T},
    model::MovingDiffusionModelDiph{N,T},
    uⁿ,
    t::T,
    dt::T;
    scheme=:CN,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    θ = _theta_from_scheme(T, scheme)
    psip, psim = _psi_functions(scheme)

    _build_moving_slab!(model, t, dt)
    cap1 = something(model.cap1_slab)
    ops1 = something(model.ops1_slab)
    cap2 = something(model.cap2_slab)
    ops2 = something(model.ops2_slab)

    nt = cap1.ntotal
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

    ufull = if length(uⁿ) == nsys
        Vector{T}(uⁿ)
    elseif length(uⁿ) == 2 * nt
        v = zeros(T, nsys)
        v[lay.ω1] .= Vector{T}(uⁿ[1:nt])
        v[lay.ω2] .= Vector{T}(uⁿ[(nt + 1):(2 * nt)])
        v
    else
        throw(DimensionMismatch("uⁿ length must be $(2 * nt) (ω1+ω2) or $nsys (full system)"))
    end

    K1, C1, J1, L1 = _weighted_core_ops(cap1, ops1, model.D1, t + θ * dt, model.coeff_mode)
    K2, C2, J2, L2 = _weighted_core_ops(cap2, ops2, model.D2, t + θ * dt, model.coeff_mode)
    f1_n, f2_n = _source_values_diph(cap1, model.source1, cap2, model.source2, t)
    f1_n1, f2_n1 = _source_values_diph(cap1, model.source1, cap2, model.source2, t + dt)
    αs1, αs2, βs1, βs2, gs, αf1, αf2, βf1, βf2, gf = _interface_coupling_diph(cap1, cap2, model.ic, t + dt)

    M1n = spdiagm(0 => model.V1n)
    M1n1 = spdiagm(0 => model.V1n1)
    M2n = spdiagm(0 => model.V2n)
    M2n1 = spdiagm(0 => model.V2n1)
    Ψ1p = spdiagm(0 => T[psip(model.V1n[i], model.V1n1[i]) for i in 1:nt])
    Ψ1m = spdiagm(0 => T[psim(model.V1n[i], model.V1n1[i]) for i in 1:nt])
    Ψ2p = spdiagm(0 => T[psip(model.V2n[i], model.V2n1[i]) for i in 1:nt])
    Ψ2m = spdiagm(0 => T[psim(model.V2n[i], model.V2n1[i]) for i in 1:nt])

    Iαs1 = spdiagm(0 => αs1)
    Iαs2 = spdiagm(0 => αs2)
    Iβs1 = spdiagm(0 => βs1)
    Iβs2 = spdiagm(0 => βs2)
    Iαf1 = spdiagm(0 => αf1)
    Iαf2 = spdiagm(0 => αf2)
    Iβf1 = spdiagm(0 => βf1)
    Iβf2 = spdiagm(0 => βf2)

    A11 = M1n1 + θ * (K1 * Ψ1p)
    A12 = -(M1n1 - M1n) + θ * (C1 * Ψ1p)
    A33 = M2n1 + θ * (K2 * Ψ2p)
    A34 = -(M2n1 - M2n) + θ * (C2 * Ψ2p)

    uω1 = Vector{T}(ufull[lay.ω1])
    uγ1 = Vector{T}(ufull[lay.γ1])
    uω2 = Vector{T}(ufull[lay.ω2])
    uγ2 = Vector{T}(ufull[lay.γ2])
    bω1 = (M1n - (one(T) - θ) * (K1 * Ψ1m)) * uω1
    bω1 .-= (one(T) - θ) .* ((C1 * Ψ1m) * uγ1)
    bω1 .+= θ .* (cap1.V * f1_n1) .+ (one(T) - θ) .* (cap1.V * f1_n)
    bω2 = (M2n - (one(T) - θ) * (K2 * Ψ2m)) * uω2
    bω2 .-= (one(T) - θ) .* ((C2 * Ψ2m) * uγ2)
    bω2 .+= θ .* (cap2.V * f2_n1) .+ (one(T) - θ) .* (cap2.V * f2_n)

    Z = spzeros(T, nt, nt)
    I = spdiagm(0 => ones(T, nt))
    A21 = Z
    A22 = I
    A23 = Z
    A24 = Z
    A41 = Z
    A42 = Z
    A43 = Z
    A44 = I
    bγ1 = zeros(T, nt)
    bγ2 = zeros(T, nt)

    if !(model.ic === nothing)
        has_scalar = !(model.ic.scalar === nothing)
        has_flux = !(model.ic.flux === nothing)

        if has_scalar
            A21 = Iβs1 * J1
            A22 = Iβs1 * L1 - Iαs1
            A23 = -(Iβs2 * J2)
            A24 = -(Iβs2 * L2) + Iαs2
            bγ1 = gs
        end
        if has_flux
            A41 = Iβf1 * J1
            A42 = Iβf1 * L1 - Iαf1
            A43 = Iβf2 * J2
            A44 = Iβf2 * L2 + Iαf2
            bγ2 = gf
        end
    end

    A, b = if _is_canonical_diph_layout(lay, nt)
        (
            [A11 A12 Z Z;
             A21 A22 A23 A24;
             Z Z A33 A34;
             A41 A42 A43 A44],
            vcat(bω1, bγ1, bω2, bγ2),
        )
    else
        Awork = spzeros(T, nsys, nsys)
        bwork = zeros(T, nsys)
        _insert_block!(Awork, lay.ω1, lay.ω1, A11)
        _insert_block!(Awork, lay.ω1, lay.γ1, A12)
        _insert_block!(Awork, lay.γ1, lay.ω1, A21)
        _insert_block!(Awork, lay.γ1, lay.γ1, A22)
        _insert_block!(Awork, lay.γ1, lay.ω2, A23)
        _insert_block!(Awork, lay.γ1, lay.γ2, A24)
        _insert_block!(Awork, lay.ω2, lay.ω2, A33)
        _insert_block!(Awork, lay.ω2, lay.γ2, A34)
        _insert_block!(Awork, lay.γ2, lay.ω1, A41)
        _insert_block!(Awork, lay.γ2, lay.γ1, A42)
        _insert_block!(Awork, lay.γ2, lay.ω2, A43)
        _insert_block!(Awork, lay.γ2, lay.γ2, A44)
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

    layω1 = UnknownLayout(nt, (ω=lay.ω1,))
    layω2 = UnknownLayout(nt, (ω=lay.ω2,))
    apply_box_bc_mono!(sys.A, sys.b, cap1, ops1, model.D1, model.bc_border; t=t + θ * dt, layout=layω1)
    apply_box_bc_mono!(sys.A, sys.b, cap2, ops2, model.D2, model.bc_border; t=t + θ * dt, layout=layω2)
    active_rows = _diph_row_activity(cap1, cap2, lay)
    sys.A, sys.b = _apply_row_identity_constraints!(sys.A, sys.b, active_rows)
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

function _init_unsteady_state_moving(model::MovingDiffusionModelMono{N,T}, u0) where {N,T}
    lay = model.layout.offsets
    nt = prod(model.grid.n)
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

function _init_unsteady_state_moving(model::MovingDiffusionModelDiph{N,T}, u0) where {N,T}
    lay = model.layout.offsets
    nt = prod(model.grid.n)
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

function solve_unsteady_moving!(
    model::MovingDiffusionModelMono{N,T},
    u0,
    tspan::Tuple{T,T};
    dt::T,
    scheme=:CN,
    method::Symbol=:direct,
    save_history::Bool=true,
    kwargs...,
) where {N,T}
    t0, tend = tspan
    tend >= t0 || throw(ArgumentError("tspan must satisfy tend >= t0"))
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    _theta_from_scheme(T, scheme) # validates accepted scheme values

    u = _init_unsteady_state_moving(model, u0)
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω), last(lay.γ)))

    times = T[t0]
    states = Vector{Vector{T}}()
    save_history && push!(states, copy(u))

    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys); x=copy(u))
    tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
    t = t0

    while t < tend - tol
        dt_step = min(dt, tend - t)
        assemble_unsteady_mono_moving!(sys, model, u, t, dt_step; scheme=scheme)
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

function solve_unsteady_moving!(
    model::MovingDiffusionModelDiph{N,T},
    u0,
    tspan::Tuple{T,T};
    dt::T,
    scheme=:CN,
    method::Symbol=:direct,
    save_history::Bool=true,
    kwargs...,
) where {N,T}
    t0, tend = tspan
    tend >= t0 || throw(ArgumentError("tspan must satisfy tend >= t0"))
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    _theta_from_scheme(T, scheme) # validates accepted scheme values

    u = _init_unsteady_state_moving(model, u0)
    lay = model.layout.offsets
    nsys = maximum((last(lay.ω1), last(lay.γ1), last(lay.ω2), last(lay.γ2)))

    times = T[t0]
    states = Vector{Vector{T}}()
    save_history && push!(states, copy(u))

    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys); x=copy(u))
    tol = sqrt(eps(T)) * max(one(T), abs(t0), abs(tend))
    t = t0

    while t < tend - tol
        dt_step = min(dt, tend - t)
        assemble_unsteady_diph_moving!(sys, model, u, t, dt_step; scheme=scheme)
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
    model::MovingDiffusionModelMono{N,T},
    u0,
    tspan::Tuple{T,T};
    dt::T,
    scheme=:CN,
    method::Symbol=:direct,
    save_history::Bool=true,
    kwargs...,
) where {N,T}
    return solve_unsteady_moving!(
        model,
        u0,
        tspan;
        dt=dt,
        scheme=scheme,
        method=method,
        save_history=save_history,
        kwargs...,
    )
end

function solve_unsteady!(
    model::MovingDiffusionModelDiph{N,T},
    u0,
    tspan::Tuple{T,T};
    dt::T,
    scheme=:CN,
    method::Symbol=:direct,
    save_history::Bool=true,
    kwargs...,
) where {N,T}
    return solve_unsteady_moving!(
        model,
        u0,
        tspan;
        dt=dt,
        scheme=scheme,
        method=method,
        save_history=save_history,
        kwargs...,
    )
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
