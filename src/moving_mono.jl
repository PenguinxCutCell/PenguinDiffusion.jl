@inline function _moving_constraint_vector(name::AbstractString, v, n::Int, ::Type{T}) where {T}
    if v isa Number
        return fill(convert(T, v), n)
    elseif v isa AbstractVector
        length(v) == n || throw(DimensionMismatch("$name has length $(length(v)); expected $n"))
        out = Vector{T}(undef, n)
        @inbounds for i in 1:n
            out[i] = convert(T, v[i])
        end
        return out
    end
    throw(ArgumentError("$name must be scalar or vector, got $(typeof(v))"))
end

@inline _convert_spatial_payload(::Type{T}, p::CartesianOperators.ScalarPayload{S}) where {T,S} =
    CartesianOperators.ScalarPayload{T}(convert(T, p.value))
@inline _convert_spatial_payload(::Type{T}, p::CartesianOperators.RefPayload{T}) where {T} = p
@inline _convert_spatial_payload(::Type{T}, p::CartesianOperators.VecPayload{T}) where {T} = p
@inline _convert_spatial_payload(::Type{T}, p::CartesianOperators.RefPayload{S}) where {T,S} =
    CartesianOperators.RefPayload{T}(Ref{T}(convert(T, p.value[])))
@inline _convert_spatial_payload(::Type{T}, p::CartesianOperators.VecPayload{S}) where {T,S} =
    CartesianOperators.VecPayload{T}(convert(Vector{T}, p.values))

@inline function _convert_spatial_bc(::Type{T}, bc::CartesianOperators.Neumann{S}) where {T,S}
    return CartesianOperators.Neumann{T}(convert(T, bc.g))
end

@inline function _convert_spatial_bc(::Type{T}, bc::CartesianOperators.Dirichlet{S,P}) where {T,S,P}
    return CartesianOperators.Dirichlet(_convert_spatial_payload(T, bc.u))
end

@inline _convert_spatial_bc(::Type{T}, ::CartesianOperators.Periodic) where {T} = CartesianOperators.Periodic{T}()

@inline function _normalize_spatial_bc(bc::CartesianOperators.BoxBC{N,S}, ::Type{T}) where {N,S,T}
    return CartesianOperators.BoxBC(
        ntuple(d -> _convert_spatial_bc(T, bc.lo[d]), N),
        ntuple(d -> _convert_spatial_bc(T, bc.hi[d]), N),
    )
end

@inline function _spacetime_bc(bc::CartesianOperators.BoxBC{N,T}) where {N,T}
    z = zero(T)
    lo = ntuple(d -> d <= N ? bc.lo[d] : CartesianOperators.Neumann{T}(z), N + 1)
    hi = ntuple(d -> d <= N ? bc.hi[d] : CartesianOperators.Neumann{T}(z), N + 1)
    return CartesianOperators.BoxBC(lo, hi)
end

@inline function _moving_source_mass!(out::AbstractVector{T}, sys, u, p, t, Vweights::AbstractVector{T}) where {T}
    sys.sourcefun === nothing && return fill!(out, zero(T))

    src = _evaluate_callable(sys.sourcefun, sys, u, p, t)
    if src isa Number
        s = convert(T, src)
        @inbounds for i in eachindex(out)
            out[i] = Vweights[i] * s
        end
        return out
    elseif src isa AbstractVector
        if length(src) == length(out)
            @inbounds for i in eachindex(out)
                out[i] = Vweights[i] * convert(T, src[i])
            end
            return out
        elseif length(src) == sys.spatial_nd
            idx = sys.omega_idx
            @inbounds for i in eachindex(out)
                out[i] = Vweights[i] * convert(T, src[idx[i]])
            end
            return out
        end
        throw(DimensionMismatch("moving source has length $(length(src)); expected $(length(out)) (reduced) or $(sys.spatial_nd) (full)"))
    end
    throw(ArgumentError("moving source callback must return scalar or full vector, got $(typeof(src))"))
end

function _zero_rows!(A::SparseMatrixCSC{T,Int}, rows::AbstractVector{Int}) where {T}
    isempty(rows) && return A
    mark = falses(size(A, 1))
    @inbounds for r in rows
        mark[r] = true
    end
    @inbounds for j in 1:size(A, 2)
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            if mark[i]
                A.nzval[p] = zero(T)
            end
        end
    end
    dropzeros!(A)
    return A
end

function _moving_dirichlet_mask_values(sys)
    return CartesianOperators.dirichlet_mask_values(sys.spatial_dims, sys.bc_box)
end

function _set_moving_robin_g!(sys, value)
    nd = sys.spatial_nd
    T = eltype(sys.robin.g)
    if value isa Number
        fill!(sys.robin.g, convert(T, value))
        return sys.robin.g
    elseif value isa AbstractVector
        if length(value) == nd
            @inbounds for i in 1:nd
                sys.robin.g[i] = convert(T, value[i])
            end
            return sys.robin.g
        elseif length(value) == length(sys.omega_idx)
            fill!(sys.robin.g, zero(T))
            @inbounds for i in eachindex(sys.omega_idx)
                sys.robin.g[sys.omega_idx[i]] = convert(T, value[i])
            end
            return sys.robin.g
        end
        throw(DimensionMismatch("moving Robin g has length $(length(value)); expected $nd (full) or $(length(sys.omega_idx)) (omega-reduced)"))
    end
    throw(ArgumentError("moving Robin g must be scalar or vector, got $(typeof(value))"))
end

mutable struct MovingDiffusionMonoSystem{N,T,PHI,SRC} <: PenguinSolverCore.AbstractSystem
    phi::PHI
    xyz::NTuple{N,Vector{T}}
    bc_box::CartesianOperators.BoxBC{N,T}
    robin::CartesianOperators.RobinConstraint{T}
    kappa::T
    sourcefun::SRC
    geom_method::Symbol

    t0::T
    t1::T

    cache::PenguinSolverCore.InvalidationCache
    updates::PenguinSolverCore.UpdateManager

    stmom
    opsST
    blk_plus

    spatial_dims::NTuple{N,Int}
    spatial_nd::Int
    omega_idx::Vector{Int}
    gamma_idx::Vector{Int}

    Vn::Vector{T}
    Vn1::Vector{T}

    L_oo::SparseMatrixCSC{T,Int}
    L_og::SparseMatrixCSC{T,Int}
    C_omega::SparseMatrixCSC{T,Int}
    C_gamma::SparseMatrixCSC{T,Int}
    r_gamma::Vector{T}
    C_gamma_fact

    rebuild_calls::Int
end

function build_moving_system(
    phi,
    xyz::NTuple{N,<:AbstractVector},
    prob::DiffusionProblem;
    t0::Real=0.0,
    t1::Real=1.0,
    method::Symbol=:implicitintegration,
) where {N}
    T = float(eltype(xyz[1]))
    xyzT = ntuple(d -> collect(T.(xyz[d])), N)
    spatial_dims = ntuple(d -> length(xyzT[d]), N)
    spatial_nd = prod(spatial_dims)

    prob.kappa isa Number || throw(ArgumentError("moving mono currently supports scalar kappa only"))
    κ = convert(T, prob.kappa)
    bc = _normalize_spatial_bc(prob.bc, T)
    robin = _normalize_robin_interface(prob.interface, T)
    sourcefun = _normalize_sourcefun(prob.source)

    empty_sparse = spzeros(T, 0, 0)
    empty_sparse_nγ = spzeros(T, 0, 0)
    empty_row = spzeros(T, 0, 0)
    empty_square = spzeros(T, 0, 0)

    sys = MovingDiffusionMonoSystem{N,T,typeof(phi),typeof(sourcefun)}(
        phi,
        xyzT,
        bc,
        robin,
        κ,
        sourcefun,
        method,
        convert(T, t0),
        convert(T, t1),
        PenguinSolverCore.InvalidationCache(),
        PenguinSolverCore.UpdateManager(),
        nothing,
        nothing,
        nothing,
        spatial_dims,
        spatial_nd,
        Int[],
        Int[],
        zeros(T, 0),
        zeros(T, 0),
        empty_sparse,
        empty_sparse_nγ,
        empty_row,
        empty_square,
        T[],
        nothing,
        0,
    )

    PenguinSolverCore.rebuild!(sys, zeros(T, spatial_nd), nothing, convert(T, t0))
    return sys
end

function _solve_moving_gamma!(xγ::AbstractVector, sys::MovingDiffusionMonoSystem, uω::AbstractVector)
    nγ = length(sys.gamma_idx)
    nγ == 0 && return xγ
    sys.C_gamma_fact === nothing && throw(ArgumentError("C_gamma factorization is missing for moving system"))
    mul!(xγ, sys.C_omega, uω)
    @inbounds for i in 1:nγ
        xγ[i] = sys.r_gamma[i] - xγ[i]
    end
    ldiv!(sys.C_gamma_fact, xγ)
    return xγ
end

function PenguinSolverCore.rebuild!(sys::MovingDiffusionMonoSystem{N,T}, u, p, t) where {N,T}
    st = CartesianGeometry.integrate_spacetime(
        sys.phi,
        sys.xyz,
        sys.t0,
        sys.t1,
        T,
        zero;
        method=sys.geom_method,
    )
    ops_st = CartesianOperators.assembled_ops(st.mST; bc=_spacetime_bc(sys.bc_box))
    blk = CartesianOperators.extract_spacetime_blocks(ops_st; layer=:plus)

    nfull = length(blk.Iγ)
    nfull == sys.spatial_nd || throw(DimensionMismatch("moving spatial Nd changed from $(sys.spatial_nd) to $nfull"))

    Vn_full = Vector{T}(st.Vcap0)
    Vn1_full = Vector{T}(st.Vcap1)
    vmax = maximum(abs, Vn1_full; init=zero(T))
    vtol = sqrt(eps(T)) * max(vmax, one(T))
    pad = padded_mask(sys.spatial_dims)

    omega_mask = falses(nfull)
    @inbounds for i in 1:nfull
        omega_mask[i] = (Vn1_full[i] > vtol) && !pad[i]
    end
    omega_idx = findall(omega_mask)
    isempty(omega_idx) && throw(ArgumentError("moving rebuild produced zero active omega DOFs"))

    κ = sys.kappa
    KWinv = κ * blk.Winv
    L_oo_full = sparse(-blk.G' * KWinv * blk.G)
    L_og_full = sparse(-blk.G' * KWinv * blk.H)

    Iγ = Vector{T}(blk.Iγ)
    ig_tol = sqrt(eps(T)) * max(maximum(abs, Iγ; init=zero(T)), one(T))
    gamma_idx = findall(i -> (Iγ[i] > ig_tol) && omega_mask[i], eachindex(Iγ))

    nω = length(omega_idx)
    L_oo = sparse(L_oo_full[omega_idx, omega_idx])
    Vn = Vector{T}(Vn_full[omega_idx])
    Vn1 = Vector{T}(Vn1_full[omega_idx])

    nγ = length(gamma_idx)
    if nγ == 0
        C_omega = spzeros(T, 0, nω)
        C_gamma = spzeros(T, 0, 0)
        r_gamma = zeros(T, 0)
        L_og = spzeros(T, nω, 0)
        C_gamma_fact = nothing
    else
        av = _moving_constraint_vector("Robin a", sys.robin.a, nfull, T)
        bv = _moving_constraint_vector("Robin b", sys.robin.b, nfull, T)
        gv = _moving_constraint_vector("Robin g", sys.robin.g, nfull, T)

        Ia = spdiagm(0 => av)
        Ib = spdiagm(0 => bv)
        Iγd = spdiagm(0 => Iγ)

        Cω_full = Ib * (blk.H' * (blk.Winv * blk.G))
        Cγ_full = Ia * Iγd + Ib * (blk.H' * (blk.Winv * blk.H))
        r_full = Iγ .* gv

        C_omega = sparse(Cω_full[gamma_idx, omega_idx])
        C_gamma = sparse(Cγ_full[gamma_idx, gamma_idx])
        r_gamma = Vector{T}(r_full[gamma_idx])
        L_og = sparse(L_og_full[omega_idx, gamma_idx])
        C_gamma_fact = lu(C_gamma)
    end

    sys.stmom = st
    sys.opsST = ops_st
    sys.blk_plus = blk
    sys.omega_idx = omega_idx
    sys.gamma_idx = gamma_idx
    sys.Vn = Vn
    sys.Vn1 = Vn1
    sys.L_oo = L_oo
    sys.L_og = L_og
    sys.C_omega = C_omega
    sys.C_gamma = C_gamma
    sys.r_gamma = r_gamma
    sys.C_gamma_fact = C_gamma_fact
    sys.rebuild_calls += 1
    return nothing
end

function moving_unsteady_block_matrix(
    sys::MovingDiffusionMonoSystem{N,T},
    dt::Real;
    scheme=:CN,
) where {N,T}
    Δt = convert(T, dt)
    Δt > zero(T) || throw(ArgumentError("dt must be positive; got $dt"))
    s = _normalize_unsteady_scheme(scheme)

    M1 = spdiagm(0 => sys.Vn1)
    nγ = length(sys.gamma_idx)
    if s === :BE
        A11 = M1 - Δt * sys.L_oo
        if nγ == 0
            return A11
        end
        A12 = -Δt * sys.L_og
        return vcat(hcat(A11, A12), hcat(sys.C_omega, sys.C_gamma))
    end

    halfdt = Δt / 2
    A11 = M1 - halfdt * sys.L_oo
    if nγ == 0
        return A11
    end
    A12 = -halfdt * sys.L_og
    return vcat(hcat(A11, A12), hcat(sys.C_omega, sys.C_gamma))
end

function moving_unsteady_block_solve(
    sys::MovingDiffusionMonoSystem{N,T},
    u0_omega::AbstractVector,
    tspan::Tuple{<:Real,<:Real};
    dt::Real,
    scheme=:CN,
    robin_gfun=nothing,
    alg=LinearSolve.KLUFactorization(),
    p=nothing,
    save_everystep::Bool=true,
    verbose::Bool=false,
    kwargs...,
) where {N,T}
    t_init, t_final = tspan
    t_final >= t_init || throw(ArgumentError("invalid tspan $(tspan); expected tspan[2] >= tspan[1]"))

    Δt_base = convert(T, dt)
    Δt_base > zero(T) || throw(ArgumentError("dt must be positive; got $dt"))
    scheme_sym = _normalize_unsteady_scheme(scheme)

    sys.t0 = convert(T, t_init)
    sys.t1 = sys.t0 + Δt_base
    if robin_gfun !== nothing
        g0 = _evaluate_callable(robin_gfun, sys, u0_omega, p, sys.t0)
        _set_moving_robin_g!(sys, g0)
    end
    PenguinSolverCore.rebuild!(sys, u0_omega, p, sys.t0)

    PenguinSolverCore.clear_updates!(sys)
    PenguinSolverCore.add_update!(sys, PenguinSolverCore.EveryStep(), PenguinSolverCore.SpaceTimeSlabUpdater(Δt_base))

    nω = length(sys.omega_idx)
    nγ = length(sys.gamma_idx)
    if length(u0_omega) != nω && length(u0_omega) != sys.spatial_nd
        throw(DimensionMismatch("u0_omega has length $(length(u0_omega)); expected $nω (reduced) or $(sys.spatial_nd) (full)"))
    end

    u0_reduced = Vector{T}(undef, nω)
    if length(u0_omega) == nω
        @inbounds for i in 1:nω
            u0_reduced[i] = convert(T, u0_omega[i])
        end
    else
        @inbounds for i in 1:nω
            u0_reduced[i] = convert(T, u0_omega[sys.omega_idx[i]])
        end
    end

    nγ = length(sys.gamma_idx)
    state = zeros(T, nω + nγ)
    @inbounds for i in 1:nω
        state[i] = u0_reduced[i]
    end
    if nγ > 0
        _solve_moving_gamma!(view(state, (nω + 1):(nω + nγ)), sys, view(state, 1:nω))
    end

    times = T[convert(T, t_init)]
    omega_hist = Vector{Vector{T}}([copy(view(state, 1:nω))])
    gamma_hist = Vector{Vector{T}}([copy(view(state, (nω + 1):(nω + nγ)))])
    state_hist = Vector{Vector{T}}([copy(state)])

    current_dt = zero(T)
    current_rebuild = -1
    A = spzeros(T, nω + nγ, nω + nγ)
    linear_cache = nothing
    rhs = zeros(T, nω + nγ)

    t = convert(T, t_init)
    t_end = convert(T, t_final)
    tol = eps(T) * max(one(T), abs(t_end))

    step = 0
    while t + tol < t_end
        step += 1
        step_dt = min(Δt_base, t_end - t)
        t_next = t + step_dt

        omega_idx_prev = copy(sys.omega_idx)
        nγ_prev = length(sys.gamma_idx)
        gamma_idx_prev = copy(sys.gamma_idx)
        nω_prev = length(omega_idx_prev)
        uω = view(state, 1:nω)
        uγ = view(state, (nω + 1):(nω + nγ_prev))

        src_prev = zeros(T, nω_prev)
        Lu_prev = zeros(T, nω_prev)
        tmpω = zeros(T, nω_prev)

        _moving_source_mass!(src_prev, sys, uω, p, t, sys.Vn)
        mul!(Lu_prev, sys.L_oo, uω)
        if nγ_prev > 0
            mul!(tmpω, sys.L_og, uγ)
            @inbounds for i in eachindex(Lu_prev)
                Lu_prev[i] += tmpω[i]
            end
        end

        rebuild_before = sys.rebuild_calls
        if robin_gfun !== nothing
            gnext = _evaluate_callable(robin_gfun, sys, uω, p, t_next)
            _set_moving_robin_g!(sys, gnext)
        end
        if step_dt == Δt_base
            PenguinSolverCore.apply_scheduled_updates!(sys, uω, p, t_next; step=step)
        else
            sys.t0 = t
            sys.t1 = t_next
            PenguinSolverCore.rebuild!(sys, uω, p, t_next)
        end
        matrix_changed = sys.rebuild_calls != rebuild_before

        nω_new = length(sys.omega_idx)
        nγ_new = length(sys.gamma_idx)
        if nω_new != nω_prev || nγ_new != nγ_prev || sys.omega_idx != omega_idx_prev || sys.gamma_idx != gamma_idx_prev
            omega_full = zeros(T, sys.spatial_nd)
            @inbounds for i in 1:nω_prev
                omega_full[omega_idx_prev[i]] = state[i]
            end

            gamma_full = zeros(T, sys.spatial_nd)
            if nγ_prev > 0
                @inbounds for i in 1:nγ_prev
                    gamma_full[gamma_idx_prev[i]] = state[nω_prev + i]
                end
            end

            state_new = zeros(T, nω_new + nγ_new)
            @inbounds for i in 1:nω_new
                state_new[i] = omega_full[sys.omega_idx[i]]
            end
            @inbounds for i in 1:nγ_new
                state_new[nω_new + i] = gamma_full[sys.gamma_idx[i]]
            end
            state = state_new

            nω = nω_new
            A = spzeros(T, nω + nγ_new, nω + nγ_new)
            rhs = zeros(T, nω + nγ_new)
            linear_cache = nothing
            current_rebuild = -1
            current_dt = zero(T)

            if nγ_new > 0
                _solve_moving_gamma!(view(state, (nω + 1):(nω + nγ_new)), sys, view(state, 1:nω))
            end
            matrix_changed = true
        end

        nω = nω_new
        uω = view(state, 1:nω)
        uγ = view(state, (nω + 1):(nω + nγ_new))

        if (step_dt != current_dt) || matrix_changed || (current_rebuild != sys.rebuild_calls)
            A = moving_unsteady_block_matrix(sys, step_dt; scheme=scheme_sym)
            current_dt = step_dt
            current_rebuild = sys.rebuild_calls
            lprob = LinearSolve.LinearProblem(A, rhs; u0=state, p=p)
            linear_cache = LinearSolve.init(lprob, alg; kwargs...)
        end

        src_next = zeros(T, nω)
        rhsω = zeros(T, nω)
        if length(Lu_prev) != nω
            Lu_prev = zeros(T, nω)
            src_prev = zeros(T, nω)
        end
        _moving_source_mass!(src_next, sys, uω, p, t_next, sys.Vn1)

        @inbounds for i in 1:nω
            rhsω[i] = sys.Vn[i] * uω[i]
        end
        if scheme_sym === :BE
            @inbounds for i in 1:nω
                rhsω[i] += step_dt * src_next[i]
            end
        else
            halfdt = step_dt / 2
            @inbounds for i in 1:nω
                rhsω[i] += halfdt * (Lu_prev[i] + src_prev[i] + src_next[i])
            end
        end

        @inbounds for i in 1:nω
            rhs[i] = rhsω[i]
        end
        if nγ_new > 0
            @inbounds for i in 1:nγ_new
                rhs[nω + i] = sys.r_gamma[i]
            end
        end

        linear_cache === nothing && error("internal error: moving linear solver cache not initialized")
        copyto!(linear_cache.b, rhs)
        copyto!(linear_cache.u, state)
        lsol = LinearSolve.solve!(linear_cache)
        copyto!(state, lsol.u)

        t = t_next
        if save_everystep || t + tol >= t_end
            push!(times, t)
            nγ_now = length(sys.gamma_idx)
            push!(omega_hist, copy(view(state, 1:nω)))
            push!(gamma_hist, copy(view(state, (nω + 1):(nω + nγ_now))))
            push!(state_hist, copy(state))
        end

        if verbose
            println("Moving step time: ", t, " nγ=", length(sys.gamma_idx))
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
