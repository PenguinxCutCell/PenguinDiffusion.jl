function solve_uγ!(
    uγ::AbstractVector,
    Cω::AbstractMatrix,
    Cγ_fact,
    r::AbstractVector,
    uω::AbstractVector,
)
    n = length(uγ)
    n == 0 && return uγ
    Cγ_fact === nothing && throw(ArgumentError("Cγ factorization is required when gamma DOFs are present"))

    mul!(uγ, Cω, uω)
    @inbounds for i in eachindex(uγ)
        uγ[i] = r[i] - uγ[i]
    end
    ldiv!(Cγ_fact, uγ)
    return uγ
end

solve_uγ!(uγ::AbstractVector, sys::TwoPhaseDiffusionSystem, uω::AbstractVector) =
    solve_uγ!(uγ, sys.Cω, sys.Cγ_fact, sys.r, uω)

function apply_L!(out::AbstractVector, sys::TwoPhaseDiffusionSystem{N,T}, u::AbstractVector) where {N,T}
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)
    length(u) == nω1 + nω2 || throw(DimensionMismatch("state length $(length(u)) != nω1+nω2 ($(nω1+nω2))"))
    length(out) == nω1 + nω2 || throw(DimensionMismatch("output length $(length(out)) != nω1+nω2 ($(nω1+nω2))"))

    @views begin
        uω1 = u[1:nω1]
        uω2 = u[(nω1 + 1):(nω1 + nω2)]
        out1 = out[1:nω1]
        out2 = out[(nω1 + 1):(nω1 + nω2)]
    end

    @views mul!(out[1:nω1], sys.Loo1, u[1:nω1])
    @views mul!(out[(nω1 + 1):(nω1 + nω2)], sys.Loo2, u[(nω1 + 1):(nω1 + nω2)])

    nγ == 0 && return out

    solve_uγ!(sys.tmp_gamma, sys, u)
    @views begin
        uγ1 = sys.tmp_gamma[1:nγ]
        uγ2 = sys.tmp_gamma[(nγ + 1):(2 * nγ)]

        mul!(sys.tmp1, sys.Log1, uγ1)
        mul!(sys.tmp2, sys.Log2, uγ2)

        for i in 1:nω1
            out[i] += sys.tmp1[i]
        end
        for i in 1:nω2
            out[nω1 + i] += sys.tmp2[i]
        end
    end
    return out
end
