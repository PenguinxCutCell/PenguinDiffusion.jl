struct TwoPhaseDiffusionProblem{K1,K2,BC1,BC2,FJ,SJ,SRC1,SRC2}
    kappa1::K1
    kappa2::K2
    bc1::BC1
    bc2::BC2
    fluxjump::FJ
    scalarjump::SJ
    source1::SRC1
    source2::SRC2
end

TwoPhaseDiffusionProblem(kappa1, kappa2, bc1, bc2, fluxjump, scalarjump) =
    TwoPhaseDiffusionProblem(kappa1, kappa2, bc1, bc2, fluxjump, scalarjump, nothing, nothing)

mutable struct TwoPhaseDiffusionSystem{N,T} <: PenguinSolverCore.AbstractSystem
    cache::PenguinSolverCore.InvalidationCache
    updates::PenguinSolverCore.UpdateManager

    moments1::CartesianGeometry.GeometricMoments{N,T}
    moments2::CartesianGeometry.GeometricMoments{N,T}

    ops1::CartesianOperators.AssembledOps{N,T}
    ops2::CartesianOperators.AssembledOps{N,T}
    kops1::CartesianOperators.KernelOps{N,T}
    kwork1::CartesianOperators.KernelWork{T}
    kops2::CartesianOperators.KernelOps{N,T}
    kwork2::CartesianOperators.KernelWork{T}

    dof_omega1::PenguinSolverCore.DofMap{Int}
    dof_omega2::PenguinSolverCore.DofMap{Int}
    dof_gamma::PenguinSolverCore.DofMap{Int}

    M::SparseMatrixCSC{T,Int}

    Loo1::SparseMatrixCSC{T,Int}
    Log1::SparseMatrixCSC{T,Int}
    Loo2::SparseMatrixCSC{T,Int}
    Log2::SparseMatrixCSC{T,Int}

    Cω::SparseMatrixCSC{T,Int}
    Cγ::SparseMatrixCSC{T,Int}
    r::Vector{T}
    Cγ_fact

    fluxjump::CartesianOperators.FluxJumpConstraint{T}
    scalarjump::CartesianOperators.ScalarJumpConstraint{T}

    kappa_cell1::Vector{T}
    kappa_face1::Vector{T}
    kappa_cell2::Vector{T}
    kappa_face2::Vector{T}
    kappa_averaging::Symbol

    dir_mask1::BitVector
    dir_vals1::Vector{T}
    dir_aff1::Vector{T}

    dir_mask2::BitVector
    dir_vals2::Vector{T}
    dir_aff2::Vector{T}

    sourcefun1::Union{Nothing,Function}
    sourcefun2::Union{Nothing,Function}

    tmp_gamma::Vector{T}
    tmp1::Vector{T}
    tmp2::Vector{T}
    tmp_full_omega1::Vector{T}
    tmp_full_gamma1::Vector{T}
    tmp_full_out1::Vector{T}
    tmp_full_omega2::Vector{T}
    tmp_full_gamma2::Vector{T}
    tmp_full_out2::Vector{T}

    matrixfree_unsteady::Bool

    diffusion_dirty::Bool
    constraints_dirty::Bool
    rebuild_calls::Int
end

function _set_two_phase_constraint_vector!(
    dest::Vector{T},
    sys::TwoPhaseDiffusionSystem{N,T},
    value;
    name::AbstractString,
) where {N,T}
    Nd = sys.ops1.Nd
    idx_gamma = sys.dof_gamma.indices

    if value isa Number
        fill!(dest, convert(T, value))
        return dest
    elseif value isa AbstractVector
        if length(value) == Nd
            @inbounds for i in 1:Nd
                dest[i] = convert(T, value[i])
            end
            return dest
        elseif length(value) == length(idx_gamma)
            @inbounds for i in eachindex(idx_gamma)
                dest[idx_gamma[i]] = convert(T, value[i])
            end
            return dest
        end
        throw(DimensionMismatch("$name vector has length $(length(value)); expected $Nd (full) or $(length(idx_gamma)) (gamma-reduced)"))
    end

    throw(ArgumentError("unsupported $name type $(typeof(value)); expected scalar or vector"))
end

function _set_two_phase_kappa_cell!(
    dest::Vector{T},
    idx_omega::Vector{Int},
    Nd::Int,
    value;
    name::AbstractString,
) where {T}
    if value isa Number
        fill!(dest, convert(T, value))
        return dest
    elseif value isa AbstractVector
        if length(value) == Nd
            @inbounds for i in 1:Nd
                dest[i] = convert(T, value[i])
            end
            return dest
        elseif length(value) == length(idx_omega)
            fill!(dest, zero(T))
            @inbounds for i in eachindex(idx_omega)
                dest[idx_omega[i]] = convert(T, value[i])
            end
            return dest
        end
        throw(DimensionMismatch("$name vector has length $(length(value)); expected $Nd (full) or $(length(idx_omega)) (omega-reduced)"))
    end

    throw(ArgumentError("unsupported $name type $(typeof(value)); expected scalar or vector"))
end

function _refresh_two_phase_rhs!(sys::TwoPhaseDiffusionSystem{N,T}) where {N,T}
    idx_gamma = sys.dof_gamma.indices
    nγ = length(idx_gamma)
    nγ == 0 && return sys.r

    @inbounds for i in 1:nγ
        idx = idx_gamma[i]
        Iγi = sys.ops1.Iγ[idx]
        sys.r[i] = Iγi * sys.fluxjump.g[idx]
        sys.r[nγ + i] = Iγi * sys.scalarjump.g[idx]
    end
    return sys.r
end

function enable_matrixfree_unsteady!(sys::TwoPhaseDiffusionSystem, enabled::Bool=true)
    sys.matrixfree_unsteady = enabled
    return sys
end
