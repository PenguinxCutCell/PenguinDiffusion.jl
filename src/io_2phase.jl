function full_state(sys::TwoPhaseDiffusionSystem{N,T}, u::AbstractVector) where {N,T}
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nω = nω1 + nω2
    length(u) == nω || throw(DimensionMismatch("state length $(length(u)) != nω1+nω2 ($nω)"))

    Nd = sys.ops1.Nd
    Tω1_full = zeros(T, Nd)
    Tγ1_full = zeros(T, Nd)
    Tω2_full = zeros(T, Nd)
    Tγ2_full = zeros(T, Nd)

    @views begin
        PenguinSolverCore.prolong!(Tω1_full, u[1:nω1], sys.dof_omega1)
        PenguinSolverCore.prolong!(Tω2_full, u[(nω1 + 1):nω], sys.dof_omega2)
    end

    @inbounds for i in eachindex(sys.dir_mask1)
        if sys.dir_mask1[i]
            Tω1_full[i] = sys.dir_vals1[i]
        end
        if sys.dir_mask2[i]
            Tω2_full[i] = sys.dir_vals2[i]
        end
    end

    nγ = length(sys.dof_gamma.indices)
    if nγ > 0
        solve_uγ!(sys.tmp_gamma, sys, u)
        @views begin
            PenguinSolverCore.prolong!(Tγ1_full, sys.tmp_gamma[1:nγ], sys.dof_gamma)
            PenguinSolverCore.prolong!(Tγ2_full, sys.tmp_gamma[(nγ + 1):(2 * nγ)], sys.dof_gamma)
        end
    end

    return Tω1_full, Tγ1_full, Tω2_full, Tγ2_full
end
