function _weighted_laplacian_kernel_full!(
    out_full::AbstractVector{T},
    kops::CartesianOperators.KernelOps{N,T},
    kwork::CartesianOperators.KernelWork{T},
    kappa_face::AbstractVector{T},
    x_omega_full::AbstractVector{T},
    x_gamma_full::AbstractVector{T},
) where {N,T}
    Nd = kops.Nd

    CartesianOperators.gradient!(kwork.g, kops, x_omega_full, x_gamma_full, kwork)
    @inbounds for i in eachindex(kwork.g)
        kwork.g[i] *= kappa_face[i]
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

    _weighted_laplacian_kernel_full!(
        sys.tmp_full_out,
        sys.kops,
        sys.kwork,
        sys.kappa_face,
        sys.tmp_full_omega,
        sys.tmp_full_gamma,
    )
    @inbounds for i in eachindex(idx_omega)
        out_omega[i] = sys.tmp_full_out[idx_omega[i]]
    end
    return out_omega
end

function apply_L_matrixfree!(out::AbstractVector, sys::TwoPhaseDiffusionSystem{N,T}, u::AbstractVector) where {N,T}
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nω = nω1 + nω2
    length(u) == nω || throw(DimensionMismatch("state length $(length(u)) != nω1+nω2 ($nω)"))
    length(out) == nω || throw(DimensionMismatch("output length $(length(out)) != nω1+nω2 ($nω)"))

    Nd = sys.ops1.Nd
    idxω1 = sys.dof_omega1.indices
    idxω2 = sys.dof_omega2.indices
    nγ = length(sys.dof_gamma.indices)

    @views begin
        uω1 = u[1:nω1]
        uω2 = u[(nω1 + 1):nω]

        fill!(sys.tmp_full_omega1, zero(T))
        fill!(sys.tmp_full_omega2, zero(T))
        PenguinSolverCore.prolong!(sys.tmp_full_omega1, uω1, sys.dof_omega1)
        PenguinSolverCore.prolong!(sys.tmp_full_omega2, uω2, sys.dof_omega2)
    end

    @inbounds for i in 1:Nd
        if sys.dir_mask1[i]
            sys.tmp_full_omega1[i] = sys.dir_vals1[i]
        end
        if sys.dir_mask2[i]
            sys.tmp_full_omega2[i] = sys.dir_vals2[i]
        end
    end

    fill!(sys.tmp_full_gamma1, zero(T))
    fill!(sys.tmp_full_gamma2, zero(T))
    if nγ > 0
        solve_uγ!(sys.tmp_gamma, sys, u)
        @views begin
            PenguinSolverCore.prolong!(sys.tmp_full_gamma1, sys.tmp_gamma[1:nγ], sys.dof_gamma)
            PenguinSolverCore.prolong!(sys.tmp_full_gamma2, sys.tmp_gamma[(nγ + 1):(2 * nγ)], sys.dof_gamma)
        end
    end

    _weighted_laplacian_kernel_full!(
        sys.tmp_full_out1,
        sys.kops1,
        sys.kwork1,
        sys.kappa_face1,
        sys.tmp_full_omega1,
        sys.tmp_full_gamma1,
    )
    _weighted_laplacian_kernel_full!(
        sys.tmp_full_out2,
        sys.kops2,
        sys.kwork2,
        sys.kappa_face2,
        sys.tmp_full_omega2,
        sys.tmp_full_gamma2,
    )

    @inbounds for i in eachindex(idxω1)
        out[i] = sys.tmp_full_out1[idxω1[i]]
    end
    @inbounds for i in eachindex(idxω2)
        out[nω1 + i] = sys.tmp_full_out2[idxω2[i]]
    end
    return out
end
