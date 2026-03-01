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

@inline function _pack_two_phase_omega!(
    uω::AbstractVector{T},
    state::AbstractVector{T},
    nω1::Int,
    nγ::Int,
    nω2::Int,
) where {T}
    length(uω) == nω1 + nω2 ||
        throw(DimensionMismatch("omega buffer length $(length(uω)) != nω1+nω2 ($(nω1 + nω2))"))
    length(state) == nω1 + nγ + nω2 + nγ ||
        throw(DimensionMismatch("state length $(length(state)) != nω1+nγ+nω2+nγ ($(nω1 + nγ + nω2 + nγ))"))

    @inbounds for i in 1:nω1
        uω[i] = state[i]
    end
    offω2 = nω1 + nγ
    @inbounds for i in 1:nω2
        uω[nω1 + i] = state[offω2 + i]
    end
    return uω
end

@inline function _copy_two_phase_gamma_from_reduced!(
    state::AbstractVector{T},
    uγ::AbstractVector{T},
    nω1::Int,
    nγ::Int,
    nω2::Int,
) where {T}
    length(uγ) == 2 * nγ || throw(DimensionMismatch("gamma vector length $(length(uγ)) != 2nγ ($(2 * nγ))"))
    nγ == 0 && return state

    offγ1 = nω1
    offγ2 = nω1 + nγ + nω2
    @inbounds for i in 1:nγ
        state[offγ1 + i] = uγ[i]
        state[offγ2 + i] = uγ[nγ + i]
    end
    return state
end

function _two_phase_affine_source_reduced!(
    out1::AbstractVector{T},
    out2::AbstractVector{T},
    sys::TwoPhaseDiffusionSystem{N,T},
    uω::AbstractVector,
    p,
    t,
) where {N,T}
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    length(out1) == nω1 || throw(DimensionMismatch("out1 length $(length(out1)) != nω1 $nω1"))
    length(out2) == nω2 || throw(DimensionMismatch("out2 length $(length(out2)) != nω2 $nω2"))

    copyto!(out1, sys.dir_aff1)
    copyto!(out2, sys.dir_aff2)

    if sys.sourcefun1 !== nothing
        src1 = _evaluate_callable(sys.sourcefun1, sys, uω, p, t)
        _source_to_reduced_phase!(sys.tmp1, sys.moments1, sys.dof_omega1.indices, sys.ops1.Nd, src1)
        @inbounds for i in 1:nω1
            out1[i] += sys.tmp1[i]
        end
    end
    if sys.sourcefun2 !== nothing
        src2 = _evaluate_callable(sys.sourcefun2, sys, uω, p, t)
        _source_to_reduced_phase!(sys.tmp2, sys.moments2, sys.dof_omega2.indices, sys.ops2.Nd, src2)
        @inbounds for i in 1:nω2
            out2[i] += sys.tmp2[i]
        end
    end
    return out1, out2
end

function _two_phase_constraint_subblocks(sys::TwoPhaseDiffusionSystem{N,T}) where {N,T}
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)

    if nγ == 0
        Zγω1 = spzeros(T, 0, nω1)
        Zγω2 = spzeros(T, 0, nω2)
        Zγγ = spzeros(T, 0, 0)
        return Zγω1, Zγγ, Zγω2, Zγγ, Zγγ, Zγγ
    end

    Cω_flux = sys.Cω[1:nγ, :]
    Cω_flux1 = Cω_flux[:, 1:nω1]
    Cω_flux2 = Cω_flux[:, (nω1 + 1):(nω1 + nω2)]
    Cγ_flux1 = sys.Cγ[1:nγ, 1:nγ]
    Cγ_flux2 = sys.Cγ[1:nγ, (nγ + 1):(2 * nγ)]
    Cγ_scal1 = sys.Cγ[(nγ + 1):(2 * nγ), 1:nγ]
    Cγ_scal2 = sys.Cγ[(nγ + 1):(2 * nγ), (nγ + 1):(2 * nγ)]
    return Cω_flux1, Cγ_flux1, Cω_flux2, Cγ_flux2, Cγ_scal1, Cγ_scal2
end

"""
    diphasic_unsteady_block_matrix(sys, dt; scheme=:BE)

Assemble two-phase unsteady block matrix with unknown ordering:
`x = [uω1; uγ1; uω2; uγ2]`.
"""
function diphasic_unsteady_block_matrix(sys::TwoPhaseDiffusionSystem{N,T}, dt::Real; scheme=:BE) where {N,T}
    Δt = convert(T, dt)
    Δt > zero(T) || throw(ArgumentError("dt must be positive; got $dt"))
    scheme_sym = _normalize_unsteady_scheme(scheme)

    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)
    nω = nω1 + nω2

    M1 = sys.M[1:nω1, 1:nω1]
    M2 = sys.M[(nω1 + 1):nω, (nω1 + 1):nω]
    fac = scheme_sym === :BE ? Δt : (Δt / 2)

    A11 = M1 - fac * sys.Loo1
    A12 = -fac * sys.Log1
    A33 = M2 - fac * sys.Loo2
    A34 = -fac * sys.Log2

    if nγ == 0
        Z12 = spzeros(T, nω1, nω2)
        Z21 = spzeros(T, nω2, nω1)
        return vcat(hcat(A11, Z12), hcat(Z21, A33))
    end

    Cω_flux1, Cγ_flux1, Cω_flux2, Cγ_flux2, Cγ_scal1, Cγ_scal2 =
        _two_phase_constraint_subblocks(sys)

    Zω1ω2 = spzeros(T, nω1, nω2)
    Zω1γ2 = spzeros(T, nω1, nγ)
    Zγω1 = spzeros(T, nγ, nω1)
    Zγω2 = spzeros(T, nγ, nω2)
    Zω2ω1 = spzeros(T, nω2, nω1)
    Zω2γ1 = spzeros(T, nω2, nγ)

    row1 = hcat(A11, A12, Zω1ω2, Zω1γ2)
    row2 = hcat(Zγω1, Cγ_scal1, Zγω2, Cγ_scal2)
    row3 = hcat(Zω2ω1, Zω2γ1, A33, A34)
    row4 = hcat(Cω_flux1, Cγ_flux1, Cω_flux2, Cγ_flux2)
    return vcat(row1, row2, row3, row4)
end

unsteady_block_matrix(sys::TwoPhaseDiffusionSystem, dt::Real; scheme=:BE) =
    diphasic_unsteady_block_matrix(sys, dt; scheme=scheme)

function _diphasic_unsteady_block_rhs!(
    rhs::AbstractVector{T},
    rhsω1::AbstractVector{T},
    rhsω2::AbstractVector{T},
    Lu1_prev::AbstractVector{T},
    Lu2_prev::AbstractVector{T},
    tmp1::AbstractVector{T},
    tmp2::AbstractVector{T},
    aff1_prev::AbstractVector{T},
    aff1_next::AbstractVector{T},
    aff2_prev::AbstractVector{T},
    aff2_next::AbstractVector{T},
    uω_prev::AbstractVector{T},
    sys::TwoPhaseDiffusionSystem{N,T},
    state_prev::AbstractVector{T},
    dt::T,
    t_prev::T,
    t_next::T,
    scheme_sym::Symbol,
    p,
    Loo1_prev,
    Log1_prev,
    Loo2_prev,
    Log2_prev,
    M1,
    M2,
) where {N,T}
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)
    nfull = nω1 + nγ + nω2 + nγ
    length(state_prev) == nfull ||
        throw(DimensionMismatch("state_prev length $(length(state_prev)) != nω1+nγ+nω2+nγ ($nfull)"))
    length(rhs) == nfull || throw(DimensionMismatch("rhs length $(length(rhs)) != $nfull"))

    @views begin
        uω1_prev = state_prev[1:nω1]
        uγ1_prev = state_prev[(nω1 + 1):(nω1 + nγ)]
        uω2_prev = state_prev[(nω1 + nγ + 1):(nω1 + nγ + nω2)]
        uγ2_prev = state_prev[(nω1 + nγ + nω2 + 1):nfull]

        _two_phase_affine_source_reduced!(aff1_prev, aff2_prev, sys, uω_prev, p, t_prev)
        _two_phase_affine_source_reduced!(aff1_next, aff2_next, sys, uω_prev, p, t_next)

        mul!(rhsω1, M1, uω1_prev)
        mul!(rhsω2, M2, uω2_prev)

        if scheme_sym === :BE
            @inbounds for i in 1:nω1
                rhsω1[i] += dt * aff1_next[i]
            end
            @inbounds for i in 1:nω2
                rhsω2[i] += dt * aff2_next[i]
            end
        else
            halfdt = dt / 2

            mul!(Lu1_prev, Loo1_prev, uω1_prev)
            mul!(tmp1, Log1_prev, uγ1_prev)
            @inbounds for i in 1:nω1
                Lu1_prev[i] += tmp1[i]
                rhsω1[i] += halfdt * (Lu1_prev[i] + aff1_prev[i] + aff1_next[i])
            end

            mul!(Lu2_prev, Loo2_prev, uω2_prev)
            mul!(tmp2, Log2_prev, uγ2_prev)
            @inbounds for i in 1:nω2
                Lu2_prev[i] += tmp2[i]
                rhsω2[i] += halfdt * (Lu2_prev[i] + aff2_prev[i] + aff2_next[i])
            end
        end
    end

    @inbounds for i in 1:nω1
        rhs[i] = rhsω1[i]
    end
    if nγ > 0
        # Row order is [scalar jump; flux jump], while sys.r stores [flux; scalar].
        @inbounds for i in 1:nγ
            rhs[nω1 + i] = sys.r[nγ + i]
        end
    end
    offω2 = nω1 + nγ
    @inbounds for i in 1:nω2
        rhs[offω2 + i] = rhsω2[i]
    end
    if nγ > 0
        offγ2 = nω1 + nγ + nω2
        @inbounds for i in 1:nγ
            rhs[offγ2 + i] = sys.r[i]
        end
    end
    return rhs
end

function _diphasic_unsteady_block_rhs(
    sys::TwoPhaseDiffusionSystem{N,T},
    state_prev::AbstractVector,
    dt::Real,
    t_prev::Real,
    t_next::Real;
    scheme=:BE,
    p=nothing,
    Loo1_prev=sys.Loo1,
    Log1_prev=sys.Log1,
    Loo2_prev=sys.Loo2,
    Log2_prev=sys.Log2,
) where {N,T}
    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)
    nω = nω1 + nω2
    nfull = nω1 + nγ + nω2 + nγ

    xprev = convert(Vector{T}, state_prev)
    length(xprev) == nfull ||
        throw(DimensionMismatch("state_prev has length $(length(xprev)); expected $nfull"))

    Δt = convert(T, dt)
    Δt > zero(T) || throw(ArgumentError("dt must be positive; got $dt"))
    tprevT = convert(T, t_prev)
    tnextT = convert(T, t_next)
    scheme_sym = _normalize_unsteady_scheme(scheme)

    uω_prev = zeros(T, nω)
    _pack_two_phase_omega!(uω_prev, xprev, nω1, nγ, nω2)

    rhs = zeros(T, nfull)
    rhsω1 = zeros(T, nω1)
    rhsω2 = zeros(T, nω2)
    Lu1_prev = zeros(T, nω1)
    Lu2_prev = zeros(T, nω2)
    tmp1 = zeros(T, nω1)
    tmp2 = zeros(T, nω2)
    aff1_prev = zeros(T, nω1)
    aff1_next = zeros(T, nω1)
    aff2_prev = zeros(T, nω2)
    aff2_next = zeros(T, nω2)
    M1 = sys.M[1:nω1, 1:nω1]
    M2 = sys.M[(nω1 + 1):nω, (nω1 + 1):nω]

    return _diphasic_unsteady_block_rhs!(
        rhs,
        rhsω1,
        rhsω2,
        Lu1_prev,
        Lu2_prev,
        tmp1,
        tmp2,
        aff1_prev,
        aff1_next,
        aff2_prev,
        aff2_next,
        uω_prev,
        sys,
        xprev,
        Δt,
        tprevT,
        tnextT,
        scheme_sym,
        p,
        Loo1_prev,
        Log1_prev,
        Loo2_prev,
        Log2_prev,
        M1,
        M2,
    )
end

"""
    diphasic_unsteady_block_solve(sys, u0, tspan; dt, scheme=:CN, ...)

Fixed-step two-phase assembled block loop with ordering
`x = [uω1; uγ1; uω2; uγ2]`.
"""
function diphasic_unsteady_block_solve(
    sys::TwoPhaseDiffusionSystem{N,T},
    u0::AbstractVector,
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

    nω1 = length(sys.dof_omega1.indices)
    nω2 = length(sys.dof_omega2.indices)
    nγ = length(sys.dof_gamma.indices)
    nω = nω1 + nω2
    nfull = nω1 + nγ + nω2 + nγ

    state = zeros(T, nfull)
    if length(u0) == nω
        @inbounds for i in 1:nω1
            state[i] = convert(T, u0[i])
        end
        offω2 = nω1 + nγ
        @inbounds for i in 1:nω2
            state[offω2 + i] = convert(T, u0[nω1 + i])
        end
    elseif length(u0) == nfull
        @inbounds for i in 1:nfull
            state[i] = convert(T, u0[i])
        end
    else
        throw(DimensionMismatch("u0 has length $(length(u0)); expected nω1+nω2 ($nω) or nω1+nγ+nω2+nγ ($nfull)"))
    end

    uω = zeros(T, nω)
    _pack_two_phase_omega!(uω, state, nω1, nγ, nω2)

    t = convert(T, t0)
    t_end = convert(T, tf)
    tol = eps(T) * max(one(T), abs(t_end))

    PenguinSolverCore.apply_scheduled_updates!(sys, uω, p, t; step=0)
    if nγ > 0
        solve_uγ!(sys.tmp_gamma, sys, uω)
        _copy_two_phase_gamma_from_reduced!(state, sys.tmp_gamma, nω1, nγ, nω2)
    end

    times = T[t]
    omega1_hist = Vector{Vector{T}}([copy(view(state, 1:nω1))])
    gamma1_hist = Vector{Vector{T}}([copy(view(state, (nω1 + 1):(nω1 + nγ)))])
    omega2_hist = Vector{Vector{T}}([copy(view(state, (nω1 + nγ + 1):(nω1 + nγ + nω2)))])
    gamma2_hist = Vector{Vector{T}}([copy(view(state, (nω1 + nγ + nω2 + 1):nfull))])
    state_hist = Vector{Vector{T}}([copy(state)])

    current_dt = zero(T)
    current_rebuild = sys.rebuild_calls
    A = spzeros(T, nfull, nfull)
    linear_cache = nothing

    rhs = zeros(T, nfull)
    rhsω1 = zeros(T, nω1)
    rhsω2 = zeros(T, nω2)
    Lu1_prev = zeros(T, nω1)
    Lu2_prev = zeros(T, nω2)
    tmp1 = zeros(T, nω1)
    tmp2 = zeros(T, nω2)
    aff1_prev = zeros(T, nω1)
    aff1_next = zeros(T, nω1)
    aff2_prev = zeros(T, nω2)
    aff2_next = zeros(T, nω2)
    M1 = sys.M[1:nω1, 1:nω1]
    M2 = sys.M[(nω1 + 1):nω, (nω1 + 1):nω]

    step = 0
    while t + tol < t_end
        step += 1
        step_dt = min(Δt_base, t_end - t)
        t_next = t + step_dt

        Loo1_prev = sys.Loo1
        Log1_prev = sys.Log1
        Loo2_prev = sys.Loo2
        Log2_prev = sys.Log2

        _pack_two_phase_omega!(uω, state, nω1, nγ, nω2)

        rebuild_before = sys.rebuild_calls
        PenguinSolverCore.apply_scheduled_updates!(sys, uω, p, t_next; step=step)
        matrix_changed = sys.rebuild_calls != rebuild_before

        if (step_dt != current_dt) || matrix_changed || (current_rebuild != sys.rebuild_calls)
            A = diphasic_unsteady_block_matrix(sys, step_dt; scheme=scheme_sym)
            current_dt = step_dt
            current_rebuild = sys.rebuild_calls
            lprob = LinearSolve.LinearProblem(A, rhs; u0=state, p=p)
            linear_cache = LinearSolve.init(lprob, alg; kwargs...)
        end

        _diphasic_unsteady_block_rhs!(
            rhs,
            rhsω1,
            rhsω2,
            Lu1_prev,
            Lu2_prev,
            tmp1,
            tmp2,
            aff1_prev,
            aff1_next,
            aff2_prev,
            aff2_next,
            uω,
            sys,
            state,
            step_dt,
            t,
            t_next,
            scheme_sym,
            p,
            Loo1_prev,
            Log1_prev,
            Loo2_prev,
            Log2_prev,
            M1,
            M2,
        )

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
            push!(omega1_hist, copy(view(state, 1:nω1)))
            push!(gamma1_hist, copy(view(state, (nω1 + 1):(nω1 + nγ))))
            push!(omega2_hist, copy(view(state, (nω1 + nγ + 1):(nω1 + nγ + nω2))))
            push!(gamma2_hist, copy(view(state, (nω1 + nγ + nω2 + 1):nfull)))
            push!(state_hist, copy(state))
        end

        if verbose
            println("Time: ", t)
            println("Solver Extremum: ", maximum(abs, state; init=zero(T)))
        end
    end

    return (
        t=times,
        omega1=omega1_hist,
        gamma1=gamma1_hist,
        omega2=omega2_hist,
        gamma2=gamma2_hist,
        states=state_hist,
        scheme=scheme_sym,
        dt=Δt_base,
    )
end

unsteady_block_solve(
    sys::TwoPhaseDiffusionSystem,
    u0::AbstractVector,
    tspan::Tuple{<:Real,<:Real};
    kwargs...,
) = diphasic_unsteady_block_solve(sys, u0, tspan; kwargs...)
