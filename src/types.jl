struct DiffusionProblem{T,BC,IC,SRC}
    kappa::T
    bc::BC
    interface::IC
    source::SRC
end

DiffusionProblem(kappa::T, bc, interface, source=nothing) where {T<:Real} =
    DiffusionProblem{T,typeof(bc),typeof(interface),typeof(source)}(kappa, bc, interface, source)

mutable struct DiffusionSystem{N,T} <: PenguinSolverCore.AbstractSystem
    cache::PenguinSolverCore.InvalidationCache
    updates::PenguinSolverCore.UpdateManager

    moments::CartesianGeometry.GeometricMoments{N,T}
    ops::CartesianOperators.AssembledOps{N,T}

    dof_omega::PenguinSolverCore.DofMap{Int}
    dof_gamma::PenguinSolverCore.DofMap{Int}

    M::SparseMatrixCSC{T,Int}

    L_oo::SparseMatrixCSC{T,Int}
    L_og::SparseMatrixCSC{T,Int}

    C_omega::SparseMatrixCSC{T,Int}
    C_gamma::SparseMatrixCSC{T,Int}
    r_gamma::Vector{T}
    C_gamma_fact

    kappa::T
    gfun::Union{Nothing,Function}
    sourcefun::Union{Nothing,Function}

    dirichlet_mask::BitVector
    dirichlet_values::Vector{T}

    tmp_gamma::Vector{T}
    tmp_omega::Vector{T}

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
    n_gamma = length(sys.r_gamma)
    n_gamma == 0 && return sys.r_gamma

    if value isa Number
        fill!(sys.r_gamma, convert(T, value))
        return sys.r_gamma
    elseif value isa AbstractVector
        if length(value) == n_gamma
            @inbounds for i in 1:n_gamma
                sys.r_gamma[i] = convert(T, value[i])
            end
            return sys.r_gamma
        elseif length(value) == sys.ops.Nd
            idx_gamma = sys.dof_gamma.indices
            @inbounds for i in eachindex(idx_gamma)
                sys.r_gamma[i] = convert(T, value[idx_gamma[i]])
            end
            return sys.r_gamma
        end
        throw(DimensionMismatch("interface RHS vector has length $(length(value)); expected $n_gamma (reduced) or $(sys.ops.Nd) (full)"))
    end

    throw(ArgumentError("unsupported interface RHS type $(typeof(value)); expected scalar or vector"))
end
