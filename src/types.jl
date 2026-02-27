struct DiffusionProblem{K,BC,IC,SRC}
    kappa::K
    bc::BC
    interface::IC
    source::SRC
end

DiffusionProblem(kappa, bc, interface) =
    DiffusionProblem(kappa, bc, interface, nothing)

mutable struct DiffusionSystem{N,T} <: PenguinSolverCore.AbstractSystem
    cache::PenguinSolverCore.InvalidationCache
    updates::PenguinSolverCore.UpdateManager

    moments::CartesianGeometry.GeometricMoments{N,T}
    ops::CartesianOperators.AssembledOps{N,T}
    kops::CartesianOperators.KernelOps{N,T}
    kwork::CartesianOperators.KernelWork{T}
    interface::CartesianOperators.RobinConstraint{T}

    dof_omega::PenguinSolverCore.DofMap{Int}
    dof_gamma::PenguinSolverCore.DofMap{Int}

    M::SparseMatrixCSC{T,Int}

    L_oo::SparseMatrixCSC{T,Int}
    L_og::SparseMatrixCSC{T,Int}

    C_omega::SparseMatrixCSC{T,Int}
    C_gamma::SparseMatrixCSC{T,Int}
    r_gamma::Vector{T}
    C_gamma_fact

    kappa_cell::Vector{T}
    kappa_face::Vector{T}
    kappa_averaging::Symbol
    gfun::Union{Nothing,Function}
    sourcefun::Union{Nothing,Function}

    dirichlet_mask::BitVector
    dirichlet_values::Vector{T}
    dirichlet_affine::Vector{T}

    tmp_gamma::Vector{T}
    tmp_omega::Vector{T}
    tmp_full_omega::Vector{T}
    tmp_full_gamma::Vector{T}
    tmp_full_out::Vector{T}

    matrixfree_unsteady::Bool

    diffusion_dirty::Bool
    constraints_dirty::Bool
    rebuild_calls::Int
end

@inline function _evaluate_callable(f, sys, u, p, t)
    if applicable(f, sys, u, p, t)
        return f(sys, u, p, t)
    elseif applicable(f, u, p, t)
        return f(u, p, t)
    elseif applicable(f, t)
        return f(t)
    elseif applicable(f)
        return f()
    end
    throw(ArgumentError("callable $(typeof(f)) must accept one of (sys, u, p, t), (u, p, t), (t), or ()"))
end

function _set_r_gamma!(sys::DiffusionSystem{N,T}, value) where {N,T}
    _set_interface_vector!(sys.interface.g, sys, value; name="interface g")
    _refresh_r_gamma_from_interface!(sys)
    return sys.r_gamma
end

function _set_kappa_cell!(sys::DiffusionSystem{N,T}, value; name::AbstractString="kappa") where {N,T}
    Nd = sys.ops.Nd
    idx_omega = sys.dof_omega.indices

    if value isa Number
        fill!(sys.kappa_cell, convert(T, value))
        return sys.kappa_cell
    elseif value isa AbstractVector
        if length(value) == Nd
            @inbounds for i in 1:Nd
                sys.kappa_cell[i] = convert(T, value[i])
            end
            return sys.kappa_cell
        elseif length(value) == length(idx_omega)
            fill!(sys.kappa_cell, zero(T))
            @inbounds for i in eachindex(idx_omega)
                sys.kappa_cell[idx_omega[i]] = convert(T, value[i])
            end
            return sys.kappa_cell
        end
        throw(DimensionMismatch("$name vector has length $(length(value)); expected $Nd (full) or $(length(idx_omega)) (omega-reduced)"))
    end

    throw(ArgumentError("unsupported $name type $(typeof(value)); expected scalar or vector"))
end

function _set_kappa!(sys::DiffusionSystem{N,T}, value; name::AbstractString="kappa") where {N,T}
    _set_kappa_cell!(sys, value; name=name)
    CartesianOperators.cell_to_face_values!(sys.kappa_face, sys.ops, sys.kappa_cell; averaging=sys.kappa_averaging)
    return sys.kappa_face
end

function _set_interface_vector!(dest::Vector{T}, sys::DiffusionSystem{N,T}, value; name::AbstractString) where {N,T}
    Nd = sys.ops.Nd
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

function _refresh_r_gamma_from_interface!(sys::DiffusionSystem{N,T}) where {N,T}
    idx_gamma = sys.dof_gamma.indices
    @inbounds for i in eachindex(idx_gamma)
        idx = idx_gamma[i]
        sys.r_gamma[i] = sys.ops.Iγ[idx] * sys.interface.g[idx]
    end
    return sys.r_gamma
end

function _refresh_dirichlet_cache!(sys::DiffusionSystem{N,T}) where {N,T}
    mask, vals = CartesianOperators.dirichlet_mask_values(sys.ops.dims, sys.ops.bc)
    length(mask) == length(sys.dirichlet_mask) ||
        throw(DimensionMismatch("Dirichlet mask length changed unexpectedly"))
    copyto!(sys.dirichlet_mask, mask)
    @inbounds for i in eachindex(sys.dirichlet_values)
        sys.dirichlet_values[i] = convert(T, vals[i])
    end

    b_full = CartesianOperators.dirichlet_rhs(sys.ops, sys.kappa_face)
    idx_omega = sys.dof_omega.indices
    length(sys.dirichlet_affine) == length(idx_omega) ||
        throw(DimensionMismatch("dirichlet_affine length does not match omega DOF count"))
    @inbounds for i in eachindex(idx_omega)
        sys.dirichlet_affine[i] = convert(T, b_full[idx_omega[i]])
    end
    return nothing
end

function enable_matrixfree_unsteady!(sys::DiffusionSystem, enabled::Bool=true)
    sys.matrixfree_unsteady = enabled
    return sys
end
