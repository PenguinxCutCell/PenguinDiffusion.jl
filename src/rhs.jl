PenguinSolverCore.mass_matrix(sys::DiffusionSystem) = sys.M

function _source_to_reduced!(out::AbstractVector{T}, sys::DiffusionSystem{N,T}, src) where {N,T}
    idx_omega = sys.dof_omega.indices
    V = sys.moments.V
    # Convention: source callbacks provide physical source density, then we mass-weight by V.

    if src isa Number
        s = convert(T, src)
        @inbounds for i in eachindex(idx_omega)
            idx = idx_omega[i]
            out[i] = V[idx] * s
        end
        return out
    elseif src isa AbstractVector
        if length(src) == length(idx_omega)
            @inbounds for i in eachindex(idx_omega)
                idx = idx_omega[i]
                out[i] = V[idx] * convert(T, src[i])
            end
            return out
        elseif length(src) == sys.ops.Nd
            @inbounds for i in eachindex(idx_omega)
                idx = idx_omega[i]
                out[i] = V[idx] * convert(T, src[idx])
            end
            return out
        end
        throw(DimensionMismatch("source vector has length $(length(src)); expected $(length(idx_omega)) (reduced) or $(sys.ops.Nd) (full)"))
    end

    throw(ArgumentError("source callback must return scalar or vector, got $(typeof(src))"))
end

function PenguinSolverCore.rhs!(du, sys::DiffusionSystem, u, p, t)
    sys.gfun === nothing || _set_r_gamma!(sys, _evaluate_callable(sys.gfun, sys, u, p, t))

    apply_L!(du, sys, u)
    @inbounds for i in eachindex(du)
        du[i] *= sys.kappa
    end

    if sys.sourcefun !== nothing
        src = _evaluate_callable(sys.sourcefun, sys, u, p, t)
        _source_to_reduced!(sys.tmp_omega, sys, src)
        @inbounds for i in eachindex(du)
            du[i] += sys.tmp_omega[i]
        end
    end

    return du
end
