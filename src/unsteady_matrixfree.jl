function _weighted_laplacian_kernel_full!(
    out_full::AbstractVector{T},
    sys::DiffusionSystem{N,T},
    x_omega_full::AbstractVector{T},
    x_gamma_full::AbstractVector{T},
) where {N,T}
    kops = sys.kops
    kwork = sys.kwork
    Nd = kops.Nd

    CartesianOperators.gradient!(kwork.g, kops, x_omega_full, x_gamma_full, kwork)
    @inbounds for i in eachindex(kwork.g)
        kwork.g[i] *= sys.kappa_face[i]
    end

    fill!(out_full, zero(T))
    @inbounds for d in 1:N
        Bd = kops.B[d]
        off = (d - 1) * Nd

        copyto!(kwork.t1, 1, kwork.g, off + 1, Nd)
        CartesianOperators.dmT!(kwork.t2, kwork.t1, kops.dims, d, kops.bc.lo[d], kops.bc.hi[d])
        for i in 1:Nd
            out_full[i] -= Bd[i] * kwork.t2[i]
        end
    end
    return out_full
end

function apply_L_matrixfree!(out_omega::AbstractVector, sys::DiffusionSystem{N,T}, u_omega::AbstractVector) where {N,T}
    Nd = sys.ops.Nd
    idx_omega = sys.dof_omega.indices

    fill!(sys.tmp_full_omega, zero(T))
    PenguinSolverCore.prolong!(sys.tmp_full_omega, u_omega, sys.dof_omega)
    @inbounds for i in 1:Nd
        if sys.dirichlet_mask[i]
            sys.tmp_full_omega[i] = sys.dirichlet_values[i]
        end
    end

    fill!(sys.tmp_full_gamma, zero(T))
    if !isempty(sys.dof_gamma.indices)
        solve_x_gamma!(sys.tmp_gamma, sys, u_omega)
        PenguinSolverCore.prolong!(sys.tmp_full_gamma, sys.tmp_gamma, sys.dof_gamma)
    end

    _weighted_laplacian_kernel_full!(sys.tmp_full_out, sys, sys.tmp_full_omega, sys.tmp_full_gamma)
    @inbounds for i in eachindex(idx_omega)
        out_omega[i] = sys.tmp_full_out[idx_omega[i]]
    end
    return out_omega
end
