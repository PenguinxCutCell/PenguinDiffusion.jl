function _normalize_fluxjump_constraint(c::CartesianOperators.FluxJumpConstraint, Nd::Int, ::Type{T}) where {T}
    length(c.b1) == Nd || throw(DimensionMismatch("fluxjump b1 has length $(length(c.b1)); expected $Nd"))
    length(c.b2) == Nd || throw(DimensionMismatch("fluxjump b2 has length $(length(c.b2)); expected $Nd"))
    length(c.g) == Nd || throw(DimensionMismatch("fluxjump g has length $(length(c.g)); expected $Nd"))
    if c isa CartesianOperators.FluxJumpConstraint{T}
        return c
    end
    return CartesianOperators.FluxJumpConstraint(convert(Vector{T}, c.b1), convert(Vector{T}, c.b2), convert(Vector{T}, c.g))
end

function _normalize_fluxjump_constraint(c::NamedTuple, Nd::Int, ::Type{T}) where {T}
    haskey(c, :b1) || throw(ArgumentError("fluxjump NamedTuple must include key :b1"))
    haskey(c, :b2) || throw(ArgumentError("fluxjump NamedTuple must include key :b2"))
    haskey(c, :g) || throw(ArgumentError("fluxjump NamedTuple must include key :g"))
    return _normalize_fluxjump_constraint(
        CartesianOperators.FluxJumpConstraint(c.b1, c.b2, c.g, Nd),
        Nd,
        T,
    )
end

function _normalize_fluxjump_constraint(c::Tuple, Nd::Int, ::Type{T}) where {T}
    length(c) == 3 || throw(DimensionMismatch("fluxjump tuple payload must have length 3 (b1, b2, g)"))
    return _normalize_fluxjump_constraint(
        CartesianOperators.FluxJumpConstraint(c[1], c[2], c[3], Nd),
        Nd,
        T,
    )
end

function _normalize_fluxjump_constraint(c, Nd::Int, ::Type{T}) where {T}
    throw(ArgumentError("unsupported fluxjump payload $(typeof(c)); expected FluxJumpConstraint, NamedTuple, or tuple"))
end

function _normalize_scalarjump_constraint(c::CartesianOperators.ScalarJumpConstraint, Nd::Int, ::Type{T}) where {T}
    length(c.α1) == Nd || throw(DimensionMismatch("scalarjump α1 has length $(length(c.α1)); expected $Nd"))
    length(c.α2) == Nd || throw(DimensionMismatch("scalarjump α2 has length $(length(c.α2)); expected $Nd"))
    length(c.g) == Nd || throw(DimensionMismatch("scalarjump g has length $(length(c.g)); expected $Nd"))
    if c isa CartesianOperators.ScalarJumpConstraint{T}
        return c
    end
    return CartesianOperators.ScalarJumpConstraint(convert(Vector{T}, c.α1), convert(Vector{T}, c.α2), convert(Vector{T}, c.g))
end

function _normalize_scalarjump_constraint(c::NamedTuple, Nd::Int, ::Type{T}) where {T}
    α1 = haskey(c, :α1) ? c.α1 : (haskey(c, :alpha1) ? c.alpha1 : nothing)
    α2 = haskey(c, :α2) ? c.α2 : (haskey(c, :alpha2) ? c.alpha2 : nothing)
    α1 === nothing && throw(ArgumentError("scalarjump NamedTuple must include key :α1 or :alpha1"))
    α2 === nothing && throw(ArgumentError("scalarjump NamedTuple must include key :α2 or :alpha2"))
    haskey(c, :g) || throw(ArgumentError("scalarjump NamedTuple must include key :g"))
    return _normalize_scalarjump_constraint(
        CartesianOperators.ScalarJumpConstraint(α1, α2, c.g, Nd),
        Nd,
        T,
    )
end

function _normalize_scalarjump_constraint(c::Tuple, Nd::Int, ::Type{T}) where {T}
    length(c) == 3 || throw(DimensionMismatch("scalarjump tuple payload must have length 3 (α1, α2, g)"))
    return _normalize_scalarjump_constraint(
        CartesianOperators.ScalarJumpConstraint(c[1], c[2], c[3], Nd),
        Nd,
        T,
    )
end

function _normalize_scalarjump_constraint(c, Nd::Int, ::Type{T}) where {T}
    throw(ArgumentError("unsupported scalarjump payload $(typeof(c)); expected ScalarJumpConstraint, NamedTuple, or tuple"))
end

function _blockdiag_sparse(A::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int}) where {T}
    m1, n1 = size(A)
    m2, n2 = size(B)
    top = hcat(A, spzeros(T, m1, n2))
    bot = hcat(spzeros(T, m2, n1), B)
    return sparse(vcat(top, bot))
end

function _refresh_two_phase_dirichlet_cache!(
    dir_mask::BitVector,
    dir_vals::Vector{T},
    dir_aff::Vector{T},
    ops::CartesianOperators.AssembledOps{N,T},
    kappa_face::Vector{T},
    dof_omega::PenguinSolverCore.DofMap{Int},
) where {N,T}
    mask, vals = CartesianOperators.dirichlet_mask_values(ops.dims, ops.bc)
    length(mask) == length(dir_mask) || throw(DimensionMismatch("Dirichlet mask length changed unexpectedly"))
    copyto!(dir_mask, mask)
    @inbounds for i in eachindex(dir_vals)
        dir_vals[i] = convert(T, vals[i])
    end

    b_full = CartesianOperators.dirichlet_rhs(ops, kappa_face)
    idx_omega = dof_omega.indices
    length(dir_aff) == length(idx_omega) || throw(DimensionMismatch("dir_aff length does not match omega DOF count"))
    @inbounds for i in eachindex(idx_omega)
        dir_aff[i] = convert(T, b_full[idx_omega[i]])
    end
    return nothing
end

function _assemble_two_phase_constraints(
    ops1::CartesianOperators.AssembledOps{N,T},
    ops2::CartesianOperators.AssembledOps{N,T},
    fluxjump::CartesianOperators.FluxJumpConstraint{T},
    scalarjump::CartesianOperators.ScalarJumpConstraint{T},
    dof_omega1::PenguinSolverCore.DofMap{Int},
    dof_omega2::PenguinSolverCore.DofMap{Int},
    dof_gamma::PenguinSolverCore.DofMap{Int},
) where {N,T}
    idxω1 = dof_omega1.indices
    idxω2 = dof_omega2.indices
    idxγ = dof_gamma.indices
    nγ = length(idxγ)
    nω = length(idxω1) + length(idxω2)

    Cω1_f, Cγ1_f, Cω2_f, Cγ2_f, r_f_full =
        CartesianOperators.fluxjump_constraint_matrices(ops1, ops2, fluxjump)
    Cγ1_s, Cγ2_s, r_s_full =
        CartesianOperators.scalarjump_constraint_matrices(ops1, ops2, scalarjump)

    Cω_f = hcat(Cω1_f[idxγ, idxω1], Cω2_f[idxγ, idxω2])
    Cγ_f = hcat(Cγ1_f[idxγ, idxγ], Cγ2_f[idxγ, idxγ])
    r_f = collect(T.(r_f_full[idxγ]))

    Cω_s = spzeros(T, nγ, nω)
    Cγ_s = hcat(Cγ1_s[idxγ, idxγ], Cγ2_s[idxγ, idxγ])
    r_s = collect(T.(r_s_full[idxγ]))

    Cω = sparse(vcat(Cω_f, Cω_s))
    Cγ = sparse(vcat(Cγ_f, Cγ_s))
    r = vcat(r_f, r_s)
    return Cω, Cγ, r
end

function _refresh_two_phase_diffusion_blocks!(sys::TwoPhaseDiffusionSystem{N,T}) where {N,T}
    KWinv1 = spdiagm(0 => sys.kappa_face1) * sys.ops1.Winv
    KWinv2 = spdiagm(0 => sys.kappa_face2) * sys.ops2.Winv

    Loo1_full = sparse(-sys.ops1.G' * KWinv1 * sys.ops1.G)
    Log1_full = sparse(-sys.ops1.G' * KWinv1 * sys.ops1.H)
    Loo2_full = sparse(-sys.ops2.G' * KWinv2 * sys.ops2.G)
    Log2_full = sparse(-sys.ops2.G' * KWinv2 * sys.ops2.H)

    idxω1 = sys.dof_omega1.indices
    idxω2 = sys.dof_omega2.indices
    idxγ = sys.dof_gamma.indices
    sys.Loo1 = Loo1_full[idxω1, idxω1]
    sys.Log1 = Log1_full[idxω1, idxγ]
    sys.Loo2 = Loo2_full[idxω2, idxω2]
    sys.Log2 = Log2_full[idxω2, idxγ]

    _refresh_two_phase_dirichlet_cache!(sys.dir_mask1, sys.dir_vals1, sys.dir_aff1, sys.ops1, sys.kappa_face1, sys.dof_omega1)
    _refresh_two_phase_dirichlet_cache!(sys.dir_mask2, sys.dir_vals2, sys.dir_aff2, sys.ops2, sys.kappa_face2, sys.dof_omega2)
    return nothing
end

function _refresh_two_phase_constraints!(sys::TwoPhaseDiffusionSystem{N,T}) where {N,T}
    Cω, Cγ, r = _assemble_two_phase_constraints(
        sys.ops1,
        sys.ops2,
        sys.fluxjump,
        sys.scalarjump,
        sys.dof_omega1,
        sys.dof_omega2,
        sys.dof_gamma,
    )
    sys.Cω = Cω
    sys.Cγ = Cγ
    sys.r = r
    return nothing
end

function build_system(
    moments1::CartesianGeometry.GeometricMoments{N,T},
    moments2::CartesianGeometry.GeometricMoments{N,T},
    prob::TwoPhaseDiffusionProblem;
    vtol::Union{Nothing,Real}=nothing,
    igamma_tol::Union{Nothing,Real}=nothing,
    kappa_averaging::Symbol=:harmonic,
) where {N,T}
    ops1 = CartesianOperators.assembled_ops(moments1; bc=prob.bc1)
    ops2 = CartesianOperators.assembled_ops(moments2; bc=prob.bc2)

    ops1.Nd == ops2.Nd || throw(DimensionMismatch("ops1/ops2 Nd mismatch"))
    ops1.dims == ops2.dims || throw(DimensionMismatch("ops1/ops2 dims mismatch"))

    Nd = ops1.Nd
    pad_mask = padded_mask(ops1.dims)

    V1 = T.(moments1.V)
    V2 = T.(moments2.V)
    maxV = max(maximum(abs, V1; init=zero(T)), maximum(abs, V2; init=zero(T)))
    vtol_local = vtol === nothing ? sqrt(eps(T)) * maxV : convert(T, vtol)

    ω_material1 = falses(Nd)
    ω_material2 = falses(Nd)
    @inbounds for i in 1:Nd
        ω_material1[i] = (moments1.cell_type[i] != 0) && (V1[i] > vtol_local)
        ω_material2[i] = (moments2.cell_type[i] != 0) && (V2[i] > vtol_local)
    end

    dir_mask1, dir_vals1_raw = CartesianOperators.dirichlet_mask_values(ops1.dims, ops1.bc)
    dir_mask2, dir_vals2_raw = CartesianOperators.dirichlet_mask_values(ops2.dims, ops2.bc)
    dir_vals1 = collect(T.(dir_vals1_raw))
    dir_vals2 = collect(T.(dir_vals2_raw))

    ω_mask1 = copy(ω_material1)
    ω_mask2 = copy(ω_material2)
    ω_mask1 .&= .!dir_mask1
    ω_mask2 .&= .!dir_mask2
    ω_mask1 .&= .!pad_mask
    ω_mask2 .&= .!pad_mask

    ω_active1 = findall(ω_mask1)
    ω_active2 = findall(ω_mask2)
    isempty(ω_active1) && throw(ArgumentError("no active omega1 DOFs after masking"))
    isempty(ω_active2) && throw(ArgumentError("no active omega2 DOFs after masking"))

    maxIγ = max(maximum(abs, ops1.Iγ; init=zero(T)), maximum(abs, ops2.Iγ; init=zero(T)))
    igamma_tol_local = igamma_tol === nothing ? sqrt(eps(T)) * maxIγ : convert(T, igamma_tol)
    γ_mask = falses(Nd)
    @inbounds for i in 1:Nd
        Iγi = max(abs(ops1.Iγ[i]), abs(ops2.Iγ[i]))
        γ_mask[i] = (Iγi > igamma_tol_local) && ω_mask1[i] && ω_mask2[i]
    end
    γ_active = findall(γ_mask)

    dof_omega1 = PenguinSolverCore.DofMap(ω_active1)
    dof_omega2 = PenguinSolverCore.DofMap(ω_active2)
    dof_gamma = PenguinSolverCore.DofMap(γ_active)

    fluxjump = _normalize_fluxjump_constraint(prob.fluxjump, Nd, T)
    scalarjump = _normalize_scalarjump_constraint(prob.scalarjump, Nd, T)

    kappa_cell1 = _normalize_kappa_cell(prob.kappa1, Nd, T)
    kappa_cell2 = _normalize_kappa_cell(prob.kappa2, Nd, T)
    kappa_face1 = CartesianOperators.cell_to_face_values(ops1, kappa_cell1; averaging=kappa_averaging)
    kappa_face2 = CartesianOperators.cell_to_face_values(ops2, kappa_cell2; averaging=kappa_averaging)

    KWinv1 = spdiagm(0 => kappa_face1) * ops1.Winv
    KWinv2 = spdiagm(0 => kappa_face2) * ops2.Winv

    idxω1 = dof_omega1.indices
    idxω2 = dof_omega2.indices
    idxγ = dof_gamma.indices

    Loo1_full = sparse(-ops1.G' * KWinv1 * ops1.G)
    Log1_full = sparse(-ops1.G' * KWinv1 * ops1.H)
    Loo2_full = sparse(-ops2.G' * KWinv2 * ops2.G)
    Log2_full = sparse(-ops2.G' * KWinv2 * ops2.H)

    Loo1 = Loo1_full[idxω1, idxω1]
    Log1 = Log1_full[idxω1, idxγ]
    Loo2 = Loo2_full[idxω2, idxω2]
    Log2 = Log2_full[idxω2, idxγ]

    b1_full = CartesianOperators.dirichlet_rhs(ops1, kappa_face1)
    b2_full = CartesianOperators.dirichlet_rhs(ops2, kappa_face2)
    dir_aff1 = collect(T.(b1_full[idxω1]))
    dir_aff2 = collect(T.(b2_full[idxω2]))

    Cω, Cγ, r = _assemble_two_phase_constraints(ops1, ops2, fluxjump, scalarjump, dof_omega1, dof_omega2, dof_gamma)
    Cγ_fact = isempty(idxγ) ? nothing : lu(Cγ)

    M1 = spdiagm(0 => V1[idxω1])
    M2 = spdiagm(0 => V2[idxω2])
    M = _blockdiag_sparse(M1, M2)

    sourcefun1 = _normalize_sourcefun(prob.source1)
    sourcefun2 = _normalize_sourcefun(prob.source2)

    nγ = length(idxγ)
    nω1 = length(idxω1)
    nω2 = length(idxω2)

    return TwoPhaseDiffusionSystem{N,T}(
        PenguinSolverCore.InvalidationCache(),
        PenguinSolverCore.UpdateManager(),
        moments1,
        moments2,
        ops1,
        ops2,
        dof_omega1,
        dof_omega2,
        dof_gamma,
        M,
        Loo1,
        Log1,
        Loo2,
        Log2,
        Cω,
        Cγ,
        r,
        Cγ_fact,
        fluxjump,
        scalarjump,
        kappa_cell1,
        kappa_face1,
        kappa_cell2,
        kappa_face2,
        kappa_averaging,
        dir_mask1,
        dir_vals1,
        dir_aff1,
        dir_mask2,
        dir_vals2,
        dir_aff2,
        sourcefun1,
        sourcefun2,
        zeros(T, 2 * nγ),
        zeros(T, nω1),
        zeros(T, nω2),
        false,
        false,
        0,
    )
end
