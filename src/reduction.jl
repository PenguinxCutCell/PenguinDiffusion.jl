function solve_x_gamma!(
    x_gamma::AbstractVector,
    C_omega::AbstractMatrix,
    C_gamma_fact,
    r_gamma::AbstractVector,
    u_omega::AbstractVector,
)
    n_gamma = length(x_gamma)
    n_gamma == 0 && return x_gamma
    C_gamma_fact === nothing && throw(ArgumentError("C_gamma factorization is required when gamma DOFs are present"))

    mul!(x_gamma, C_omega, u_omega)
    @inbounds for i in eachindex(x_gamma)
        x_gamma[i] = r_gamma[i] - x_gamma[i]
    end
    ldiv!(C_gamma_fact, x_gamma)
    return x_gamma
end

solve_x_gamma!(x_gamma::AbstractVector, sys::DiffusionSystem, u_omega::AbstractVector) =
    solve_x_gamma!(x_gamma, sys.C_omega, sys.C_gamma_fact, sys.r_gamma, u_omega)

function apply_L!(
    out_omega::AbstractVector,
    L_oo::AbstractMatrix,
    L_og::AbstractMatrix,
    C_omega::AbstractMatrix,
    C_gamma_fact,
    r_gamma::AbstractVector,
    tmp_gamma::AbstractVector,
    tmp_omega::AbstractVector,
    u_omega::AbstractVector,
)
    mul!(out_omega, L_oo, u_omega)
    isempty(tmp_gamma) && return out_omega

    solve_x_gamma!(tmp_gamma, C_omega, C_gamma_fact, r_gamma, u_omega)
    mul!(tmp_omega, L_og, tmp_gamma)
    @inbounds for i in eachindex(out_omega)
        out_omega[i] += tmp_omega[i]
    end
    return out_omega
end

apply_L!(out_omega::AbstractVector, sys::DiffusionSystem, u_omega::AbstractVector) =
    apply_L!(
        out_omega,
        sys.L_oo,
        sys.L_og,
        sys.C_omega,
        sys.C_gamma_fact,
        sys.r_gamma,
        sys.tmp_gamma,
        sys.tmp_omega,
        u_omega,
    )
