function full_state(sys::DiffusionSystem{N,T}, u_omega::AbstractVector) where {N,T}
    length(u_omega) == length(sys.dof_omega.indices) ||
        throw(DimensionMismatch("reduced omega length $(length(u_omega)) does not match DOF map length $(length(sys.dof_omega.indices))"))

    x_omega_full = zeros(T, sys.ops.Nd)
    x_gamma_full = zeros(T, sys.ops.Nd)

    PenguinSolverCore.prolong!(x_omega_full, u_omega, sys.dof_omega)

    @inbounds for i in eachindex(sys.dirichlet_mask)
        if sys.dirichlet_mask[i]
            x_omega_full[i] = sys.dirichlet_values[i]
        end
    end

    if !isempty(sys.dof_gamma.indices)
        solve_x_gamma!(sys.tmp_gamma, sys, u_omega)
        PenguinSolverCore.prolong!(x_gamma_full, sys.tmp_gamma, sys.dof_gamma)
    end

    return x_omega_full, x_gamma_full
end
