PenguinSolverCore.mass_matrix(sys::TwoPhaseDiffusionSystem) = sys.M

function _source_to_reduced_phase!(
    out::AbstractVector{T},
    moments::CartesianGeometry.GeometricMoments{N,T},
    idx_omega::Vector{Int},
    Nd::Int,
    src,
) where {N,T}
    V = moments.V
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
        elseif length(src) == Nd
            @inbounds for i in eachindex(idx_omega)
                idx = idx_omega[i]
                out[i] = V[idx] * convert(T, src[idx])
            end
            return out
        end
        throw(DimensionMismatch("source vector has length $(length(src)); expected $(length(idx_omega)) (reduced) or $Nd (full)"))
    end

    throw(ArgumentError("source callback must return scalar or vector, got $(typeof(src))"))
end

function PenguinSolverCore.rhs!(du, sys::TwoPhaseDiffusionSystem, u, p, t)
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nω = nω1 + nω2
    length(u) == nω || throw(DimensionMismatch("state length $(length(u)) != $nω"))
    length(du) == nω || throw(DimensionMismatch("rhs length $(length(du)) != $nω"))

    apply_L!(du, sys, u)

    @inbounds for i in 1:nω1
        du[i] += sys.dir_aff1[i]
    end
    @inbounds for i in 1:nω2
        du[nω1 + i] += sys.dir_aff2[i]
    end

    if sys.sourcefun1 !== nothing
        src1 = _evaluate_callable(sys.sourcefun1, sys, u, p, t)
        _source_to_reduced_phase!(sys.tmp1, sys.moments1, sys.dof_omega1.indices, sys.ops1.Nd, src1)
        @inbounds for i in 1:nω1
            du[i] += sys.tmp1[i]
        end
    end

    if sys.sourcefun2 !== nothing
        src2 = _evaluate_callable(sys.sourcefun2, sys, u, p, t)
        _source_to_reduced_phase!(sys.tmp2, sys.moments2, sys.dof_omega2.indices, sys.ops2.Nd, src2)
        @inbounds for i in 1:nω2
            du[nω1 + i] += sys.tmp2[i]
        end
    end

    return du
end
