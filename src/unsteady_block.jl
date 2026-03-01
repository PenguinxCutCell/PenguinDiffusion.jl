@inline function _normalize_unsteady_scheme(scheme)
    if scheme isa Symbol
        s = Symbol(uppercase(String(scheme)))
    elseif scheme isa AbstractString
        s = Symbol(uppercase(strip(scheme)))
    else
        throw(ArgumentError("unsupported scheme type $(typeof(scheme)); expected Symbol or String"))
    end
    if s === :BE || s === :CN
        return s
    end
    throw(ArgumentError("unsupported scheme `$scheme`; expected :BE or :CN"))
end

@inline function _refresh_robin_g!(sys::DiffusionSystem, u, p, t)
    sys.gfun === nothing && return nothing
    _set_r_gamma!(sys, _evaluate_callable(sys.gfun, sys, u, p, t))
    return nothing
end

function _affine_source_reduced!(out::AbstractVector{T}, sys::DiffusionSystem{N,T}, u, p, t) where {N,T}
    nω = length(sys.dof_omega.indices)
    length(out) == nω || throw(DimensionMismatch("output length $(length(out)) != n_omega $nω"))

    @inbounds for i in 1:nω
        out[i] = sys.dirichlet_affine[i]
    end

    if sys.sourcefun !== nothing
        src = _evaluate_callable(sys.sourcefun, sys, u, p, t)
        _source_to_reduced!(sys.tmp_omega, sys, src)
        @inbounds for i in 1:nω
            out[i] += sys.tmp_omega[i]
        end
    end
    return out
end

"""
    unsteady_block_matrix(sys, dt; scheme=:BE)

Assemble the monophase unsteady block matrix on reduced DOFs using unknown ordering
`x = [u_omega; u_gamma]`.

For `scheme=:BE`:
- `A11 = M - dt*L_oo`
- `A12 = -dt*L_og`
- `A21 = C_omega`
- `A22 = C_gamma`

For `scheme=:CN`:
- `A11 = M - dt/2*L_oo`
- `A12 = -dt/2*L_og`
- `A21 = C_omega`
- `A22 = C_gamma`
"""
function unsteady_block_matrix(sys::DiffusionSystem{N,T}, dt::Real; scheme=:BE) where {N,T}
    Δt = convert(T, dt)
    Δt > zero(T) || throw(ArgumentError("dt must be positive; got $dt"))

    scheme_sym = _normalize_unsteady_scheme(scheme)
    nγ = length(sys.dof_gamma.indices)

    if scheme_sym === :BE
        A11 = sys.M - Δt * sys.L_oo
        if nγ == 0
            return A11
        end
        A12 = -Δt * sys.L_og
        return vcat(hcat(A11, A12), hcat(sys.C_omega, sys.C_gamma))
    end

    halfdt = Δt / 2
    A11 = sys.M - halfdt * sys.L_oo
    if nγ == 0
        return A11
    end
    A12 = -halfdt * sys.L_og
    return vcat(hcat(A11, A12), hcat(sys.C_omega, sys.C_gamma))
end

"""
    unsteady_block_solve(sys, u0_omega, tspan; dt, scheme=:CN, ...)

Fixed-step implicit time loop using the assembled block matrix with unknown ordering
`x = [u_omega; u_gamma]`, reproducing the legacy assembled strategy.

`scheme=:BE` is first-order in time; `scheme=:CN` is second-order and typically
gives much lower temporal error at the same `dt`.

Returns a named tuple with fields:
- `t`: saved time points
- `omega`: saved `u_omega` states
- `gamma`: saved `u_gamma` states
- `states`: saved block states `[u_omega; u_gamma]`
- `scheme`, `dt`
"""
function unsteady_block_solve(
    sys::DiffusionSystem{N,T},
    u0_omega::AbstractVector,
    tspan::Tuple{<:Real,<:Real};
    dt::Real,
    scheme=:CN,
    alg=LinearSolve.KLUFactorization(),
    p=nothing,
    save_everystep::Bool=true,
    verbose::Bool=false,
    kwargs...,
) where {N,T}
    t0, tf = tspan
    tf >= t0 || throw(ArgumentError("invalid tspan $(tspan); expected tspan[2] >= tspan[1]"))

    Δt_base = convert(T, dt)
    Δt_base > zero(T) || throw(ArgumentError("dt must be positive; got $dt"))
    scheme_sym = _normalize_unsteady_scheme(scheme)

    nω = length(sys.dof_omega.indices)
    nγ = length(sys.dof_gamma.indices)
    length(u0_omega) == nω ||
        throw(DimensionMismatch("u0_omega has length $(length(u0_omega)); expected $nω"))

    state = zeros(T, nω + nγ)
    @inbounds for i in 1:nω
        state[i] = convert(T, u0_omega[i])
    end

    uω = view(state, 1:nω)
    uγ = view(state, (nω + 1):(nω + nγ))

    t = convert(T, t0)
    t_end = convert(T, tf)
    tol = eps(T) * max(one(T), abs(t_end))

    # Initialize scheduled updates and time-dependent Robin g at t0.
    PenguinSolverCore.apply_scheduled_updates!(sys, uω, p, t; step=0)
    _refresh_robin_g!(sys, uω, p, t)
    if nγ > 0
        solve_x_gamma!(uγ, sys, uω)
    end

    times = T[t]
    omega_hist = Vector{Vector{T}}([copy(uω)])
    gamma_hist = Vector{Vector{T}}([copy(uγ)])
    state_hist = Vector{Vector{T}}([copy(state)])

    current_dt = zero(T)
    current_rebuild = sys.rebuild_calls
    A = spzeros(T, nω + nγ, nω + nγ)
    linear_cache = nothing

    aff_prev = zeros(T, nω)
    aff_next = zeros(T, nω)
    Lu_prev = zeros(T, nω)
    tmpω = zeros(T, nω)
    rhs_omega = zeros(T, nω)
    rhs_gamma = zeros(T, nγ)
    rhs = zeros(T, nω + nγ)

    step = 0
    while t + tol < t_end
        step += 1
        step_dt = min(Δt_base, t_end - t)
        t_next = t + step_dt

        # Save previous-time operators/constraint blocks and forcing.
        Loo_prev = sys.L_oo
        Log_prev = sys.L_og
        _refresh_robin_g!(sys, uω, p, t)
        _affine_source_reduced!(aff_prev, sys, uω, p, t)

        mul!(Lu_prev, Loo_prev, uω)
        if nγ > 0
            mul!(tmpω, Log_prev, uγ)
            @inbounds for i in 1:nω
                Lu_prev[i] += tmpω[i]
            end
        end

        # Apply updates at t_{n+1}; this can mutate matrices/RHS data and rebuild.
        rebuild_before = sys.rebuild_calls
        PenguinSolverCore.apply_scheduled_updates!(sys, uω, p, t_next; step=step)
        _refresh_robin_g!(sys, uω, p, t_next)
        matrix_changed = sys.rebuild_calls != rebuild_before

        if (step_dt != current_dt) || matrix_changed || (current_rebuild != sys.rebuild_calls)
            A = unsteady_block_matrix(sys, step_dt; scheme=scheme_sym)
            current_dt = step_dt
            current_rebuild = sys.rebuild_calls
            lprob = LinearSolve.LinearProblem(A, rhs; u0=state, p=p)
            linear_cache = LinearSolve.init(lprob, alg; kwargs...)
        end

        _affine_source_reduced!(aff_next, sys, uω, p, t_next)

        mul!(rhs_omega, sys.M, uω)
        if scheme_sym === :BE
            @inbounds for i in 1:nω
                rhs_omega[i] += step_dt * aff_next[i]
            end
            if nγ > 0
                copyto!(rhs_gamma, sys.r_gamma)
            end
        else
            halfdt = step_dt / 2
            @inbounds for i in 1:nω
                rhs_omega[i] += halfdt * Lu_prev[i] + halfdt * (aff_prev[i] + aff_next[i])
            end
            if nγ > 0
                copyto!(rhs_gamma, sys.r_gamma)
            end
        end

        @inbounds for i in 1:nω
            rhs[i] = rhs_omega[i]
        end
        if nγ > 0
            @inbounds for i in 1:nγ
                rhs[nω + i] = rhs_gamma[i]
            end
        end

        linear_cache === nothing && error("internal error: linear solver cache was not initialized")
        copyto!(linear_cache.b, rhs)
        copyto!(linear_cache.u, state)
        lsol = LinearSolve.solve!(linear_cache)
        x_next = lsol.u
        @inbounds for i in eachindex(state)
            state[i] = x_next[i]
        end

        t = t_next
        if save_everystep || t + tol >= t_end
            push!(times, t)
            push!(omega_hist, copy(uω))
            push!(gamma_hist, copy(uγ))
            push!(state_hist, copy(state))
        end

        if verbose
            println("Time: ", t)
            println("Solver Extremum: ", maximum(abs, state; init=zero(T)))
        end
    end

    return (
        t=times,
        omega=omega_hist,
        gamma=gamma_hist,
        states=state_hist,
        scheme=scheme_sym,
        dt=Δt_base,
    )
end
