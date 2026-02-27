function _steady_probe_state(sys::DiffusionSystem{N,T}, u_eval) where {N,T}
    n_omega = length(sys.dof_omega.indices)
    if u_eval === nothing
        return zeros(T, n_omega)
    end
    length(u_eval) == n_omega ||
        throw(DimensionMismatch("u_eval has length $(length(u_eval)); expected $n_omega"))
    return convert(Vector{T}, u_eval)
end

function _steady_probe_state(sys::TwoPhaseDiffusionSystem{N,T}, u_eval) where {N,T}
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nω = nω1 + nω2
    if u_eval === nothing
        return zeros(T, nω)
    end
    length(u_eval) == nω ||
        throw(DimensionMismatch("u_eval has length $(length(u_eval)); expected $nω"))
    return convert(Vector{T}, u_eval)
end

function _steady_affine_matrixfree(sys::DiffusionSystem{N,T}) where {N,T}
    nω = length(sys.dof_omega.indices)
    zero_state = zeros(T, nω)
    affine = zeros(T, nω)
    apply_L_matrixfree!(affine, sys, zero_state)
    return affine
end

function _steady_affine_matrixfree(sys::TwoPhaseDiffusionSystem{N,T}) where {N,T}
    nω = length(sys.dof_omega1.indices) + length(sys.dof_omega2.indices)
    zero_state = zeros(T, nω)
    affine = zeros(T, nω)
    apply_L_matrixfree!(affine, sys, zero_state)
    return affine
end

function steady_linear_problem(
    sys::DiffusionSystem{N,T};
    p=nothing,
    t::Real=zero(T),
    u0=nothing,
    u_eval=nothing,
) where {N,T}
    maximum(abs, sys.kappa_face; init=zero(T)) == zero(T) &&
        throw(ArgumentError("steady solver requires nonzero kappa (all face coefficients are zero)"))

    n_omega = length(sys.dof_omega.indices)
    n_gamma = length(sys.dof_gamma.indices)
    if n_gamma > 0 && sys.C_gamma_fact === nothing
        throw(ArgumentError("C_gamma factorization is missing while gamma DOFs are present"))
    end

    u0_vec = u0 === nothing ? zeros(T, n_omega) : convert(Vector{T}, u0)
    length(u0_vec) == n_omega ||
        throw(DimensionMismatch("u0 has length $(length(u0_vec)); expected $n_omega"))

    affine = _steady_affine_matrixfree(sys)
    rhs = zeros(T, n_omega)
    @inbounds for i in eachindex(rhs)
        rhs[i] = -affine[i]
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

    op! = (out, x, _u, _p, _t) -> begin
        apply_L_matrixfree!(out, sys, x)
        @inbounds for i in eachindex(out)
            out[i] -= affine[i]
        end
        out
    end
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

function steady_linear_problem(
    sys::TwoPhaseDiffusionSystem{N,T};
    p=nothing,
    t::Real=zero(T),
    u0=nothing,
    u_eval=nothing,
) where {N,T}
    k1_zero = maximum(abs, sys.kappa_face1; init=zero(T)) == zero(T)
    k2_zero = maximum(abs, sys.kappa_face2; init=zero(T)) == zero(T)
    (k1_zero && k2_zero) &&
        throw(ArgumentError("steady solver requires nonzero kappa in at least one phase"))

    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nω = nω1 + nω2
    nγ = length(sys.dof_gamma.indices)
    if nγ > 0 && sys.Cγ_fact === nothing
        throw(ArgumentError("Cγ factorization is missing while gamma DOFs are present"))
    end

    u0_vec = u0 === nothing ? zeros(T, nω) : convert(Vector{T}, u0)
    length(u0_vec) == nω ||
        throw(DimensionMismatch("u0 has length $(length(u0_vec)); expected $nω"))

    affine = _steady_affine_matrixfree(sys)
    rhs = zeros(T, nω)
    @inbounds for i in eachindex(rhs)
        rhs[i] = -affine[i]
    end

    u_probe = _steady_probe_state(sys, u_eval)
    if sys.sourcefun1 !== nothing
        src1 = _evaluate_callable(sys.sourcefun1, sys, u_probe, p, t)
        src1_reduced = zeros(T, nω1)
        _source_to_reduced_phase!(src1_reduced, sys.moments1, sys.dof_omega1.indices, sys.ops1.Nd, src1)
        @inbounds for i in 1:nω1
            rhs[i] -= src1_reduced[i]
        end
    end
    if sys.sourcefun2 !== nothing
        src2 = _evaluate_callable(sys.sourcefun2, sys, u_probe, p, t)
        src2_reduced = zeros(T, nω2)
        _source_to_reduced_phase!(src2_reduced, sys.moments2, sys.dof_omega2.indices, sys.ops2.Nd, src2)
        @inbounds for i in 1:nω2
            rhs[nω1 + i] -= src2_reduced[i]
        end
    end

    op! = (out, x, _u, _p, _t) -> begin
        apply_L_matrixfree!(out, sys, x)
        @inbounds for i in eachindex(out)
            out[i] -= affine[i]
        end
        out
    end
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

function steady_solve(
    sys::TwoPhaseDiffusionSystem;
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
