function _steady_probe_state(sys::DiffusionSystem{N,T}, u_eval) where {N,T}
    n_omega = length(sys.dof_omega.indices)
    if u_eval === nothing
        return zeros(T, n_omega)
    end
    length(u_eval) == n_omega ||
        throw(DimensionMismatch("u_eval has length $(length(u_eval)); expected $n_omega"))
    return convert(Vector{T}, u_eval)
end

function _steady_rhs(
    sys::DiffusionSystem{N,T};
    p=nothing,
    t::Real=zero(T),
    u_eval=nothing,
) where {N,T}
    n_omega = length(sys.dof_omega.indices)
    rhs = zeros(T, n_omega)

    if !isempty(sys.dof_gamma.indices)
        q = copy(sys.r_gamma)
        ldiv!(sys.C_gamma_fact, q)
        mul!(rhs, sys.L_og, q)
        @inbounds for i in eachindex(rhs)
            rhs[i] *= -sys.kappa
        end
    end

    @inbounds for i in eachindex(rhs)
        rhs[i] -= sys.kappa * sys.dirichlet_affine[i]
    end

    if sys.sourcefun !== nothing
        u_probe = _steady_probe_state(sys, u_eval)
        src = _evaluate_callable(sys.sourcefun, sys, u_probe, p, t)
        src_reduced = zeros(T, n_omega)
        _source_to_reduced!(src_reduced, sys, src)
        @inbounds for i in eachindex(rhs)
            rhs[i] -= src_reduced[i]
        end
    end

    return rhs
end

function _steady_mul!(
    out::AbstractVector{T},
    x::AbstractVector{T},
    sys::DiffusionSystem{N,T},
    tmp_gamma::Vector{T},
    tmp_omega::Vector{T},
) where {N,T}
    mul!(out, sys.L_oo, x)
    if !isempty(tmp_gamma)
        mul!(tmp_gamma, sys.C_omega, x)
        @inbounds for i in eachindex(tmp_gamma)
            tmp_gamma[i] = -tmp_gamma[i]
        end
        ldiv!(sys.C_gamma_fact, tmp_gamma)
        mul!(tmp_omega, sys.L_og, tmp_gamma)
        @inbounds for i in eachindex(out)
            out[i] += tmp_omega[i]
        end
    end
    @inbounds for i in eachindex(out)
        out[i] *= sys.kappa
    end
    return out
end

function steady_linear_problem(
    sys::DiffusionSystem{N,T};
    p=nothing,
    t::Real=zero(T),
    u0=nothing,
    u_eval=nothing,
) where {N,T}
    iszero(sys.kappa) && throw(ArgumentError("steady solver requires nonzero kappa"))

    n_omega = length(sys.dof_omega.indices)
    n_gamma = length(sys.dof_gamma.indices)
    if n_gamma > 0 && sys.C_gamma_fact === nothing
        throw(ArgumentError("C_gamma factorization is missing while gamma DOFs are present"))
    end

    u0_vec = u0 === nothing ? zeros(T, n_omega) : convert(Vector{T}, u0)
    length(u0_vec) == n_omega ||
        throw(DimensionMismatch("u0 has length $(length(u0_vec)); expected $n_omega"))

    rhs = _steady_rhs(sys; p=p, t=t, u_eval=u_eval)

    tmp_gamma = zeros(T, n_gamma)
    tmp_omega = zeros(T, n_omega)
    op! = (out, x, _u, _p, _t) -> _steady_mul!(out, x, sys, tmp_gamma, tmp_omega)
    Aop = LinearSolve.FunctionOperator(
        op!,
        u0_vec,
        similar(u0_vec);
        isinplace=true,
        T=T,
        isconstant=true,
    )

    return LinearSolve.LinearProblem(Aop, rhs; u0=u0_vec, p=p)
end

function steady_solve(
    sys::DiffusionSystem;
    alg=LinearSolve.SimpleGMRES(),
    p=nothing,
    t=0.0,
    u0=nothing,
    u_eval=nothing,
    kwargs...,
)
    prob = steady_linear_problem(sys; p=p, t=t, u0=u0, u_eval=u_eval)
    return LinearSolve.solve(prob, alg; kwargs...)
end
