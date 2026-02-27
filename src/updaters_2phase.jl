mutable struct FluxJumpGUpdater{F} <: PenguinSolverCore.AbstractUpdater
    gfun::F
end

mutable struct FluxJumpBUpdater{F} <: PenguinSolverCore.AbstractUpdater
    bfun::F
end

mutable struct ScalarJumpGUpdater{F} <: PenguinSolverCore.AbstractUpdater
    gfun::F
end

mutable struct ScalarJumpAlphaUpdater{F} <: PenguinSolverCore.AbstractUpdater
    αfun::F
end

mutable struct Kappa1Updater{F} <: PenguinSolverCore.AbstractUpdater
    kappa_fun::F
end

mutable struct Kappa2Updater{F} <: PenguinSolverCore.AbstractUpdater
    kappa_fun::F
end

mutable struct BoxDirichletUpdater1{F} <: PenguinSolverCore.AbstractUpdater
    ufun::F
end

mutable struct BoxDirichletUpdater2{F} <: PenguinSolverCore.AbstractUpdater
    ufun::F
end

mutable struct Source1Updater{F} <: PenguinSolverCore.AbstractUpdater
    source_fun::F
end

mutable struct Source2Updater{F} <: PenguinSolverCore.AbstractUpdater
    source_fun::F
end

function _unpack_fluxjump_b_update(payload)
    if payload isa NamedTuple
        haskey(payload, :b1) || throw(ArgumentError("FluxJumpBUpdater NamedTuple payload must include key :b1"))
        haskey(payload, :b2) || throw(ArgumentError("FluxJumpBUpdater NamedTuple payload must include key :b2"))
        g = haskey(payload, :g) ? payload.g : nothing
        return payload.b1, payload.b2, g
    elseif payload isa Tuple
        if length(payload) == 2
            return payload[1], payload[2], nothing
        elseif length(payload) == 3
            return payload[1], payload[2], payload[3]
        end
        throw(DimensionMismatch("FluxJumpBUpdater tuple payload must have length 2 (b1,b2) or 3 (b1,b2,g)"))
    end
    throw(ArgumentError("FluxJumpBUpdater payload must be NamedTuple or tuple; got $(typeof(payload))"))
end

function _unpack_scalarjump_alpha_update(payload)
    if payload isa NamedTuple
        α1 = haskey(payload, :α1) ? payload.α1 : (haskey(payload, :alpha1) ? payload.alpha1 : nothing)
        α2 = haskey(payload, :α2) ? payload.α2 : (haskey(payload, :alpha2) ? payload.alpha2 : nothing)
        α1 === nothing && throw(ArgumentError("ScalarJumpAlphaUpdater NamedTuple payload must include :α1 or :alpha1"))
        α2 === nothing && throw(ArgumentError("ScalarJumpAlphaUpdater NamedTuple payload must include :α2 or :alpha2"))
        g = haskey(payload, :g) ? payload.g : nothing
        return α1, α2, g
    elseif payload isa Tuple
        if length(payload) == 2
            return payload[1], payload[2], nothing
        elseif length(payload) == 3
            return payload[1], payload[2], payload[3]
        end
        throw(DimensionMismatch("ScalarJumpAlphaUpdater tuple payload must have length 2 (α1,α2) or 3 (α1,α2,g)"))
    end
    throw(ArgumentError("ScalarJumpAlphaUpdater payload must be NamedTuple or tuple; got $(typeof(payload))"))
end

function PenguinSolverCore.update!(upd::FluxJumpGUpdater, sys::TwoPhaseDiffusionSystem, u, p, t)
    values = _evaluate_callable(upd.gfun, sys, u, p, t)
    _set_two_phase_constraint_vector!(sys.fluxjump.g, sys, values; name="fluxjump g")
    _refresh_two_phase_rhs!(sys)
    return :rhs_only
end

function PenguinSolverCore.update!(upd::FluxJumpBUpdater, sys::TwoPhaseDiffusionSystem, u, p, t)
    payload = _evaluate_callable(upd.bfun, sys, u, p, t)
    b1, b2, g = _unpack_fluxjump_b_update(payload)
    _set_two_phase_constraint_vector!(sys.fluxjump.b1, sys, b1; name="fluxjump b1")
    _set_two_phase_constraint_vector!(sys.fluxjump.b2, sys, b2; name="fluxjump b2")
    g === nothing || _set_two_phase_constraint_vector!(sys.fluxjump.g, sys, g; name="fluxjump g")
    sys.constraints_dirty = true
    return :matrix
end

function PenguinSolverCore.update!(upd::ScalarJumpGUpdater, sys::TwoPhaseDiffusionSystem, u, p, t)
    values = _evaluate_callable(upd.gfun, sys, u, p, t)
    _set_two_phase_constraint_vector!(sys.scalarjump.g, sys, values; name="scalarjump g")
    _refresh_two_phase_rhs!(sys)
    return :rhs_only
end

function PenguinSolverCore.update!(upd::ScalarJumpAlphaUpdater, sys::TwoPhaseDiffusionSystem, u, p, t)
    payload = _evaluate_callable(upd.αfun, sys, u, p, t)
    α1, α2, g = _unpack_scalarjump_alpha_update(payload)
    _set_two_phase_constraint_vector!(sys.scalarjump.α1, sys, α1; name="scalarjump α1")
    _set_two_phase_constraint_vector!(sys.scalarjump.α2, sys, α2; name="scalarjump α2")
    g === nothing || _set_two_phase_constraint_vector!(sys.scalarjump.g, sys, g; name="scalarjump g")
    sys.constraints_dirty = true
    return :matrix
end

function PenguinSolverCore.update!(upd::Kappa1Updater, sys::TwoPhaseDiffusionSystem{N,T}, u, p, t) where {N,T}
    payload = _evaluate_callable(upd.kappa_fun, sys, u, p, t)
    _set_two_phase_kappa_cell!(sys.kappa_cell1, sys.dof_omega1.indices, sys.ops1.Nd, payload; name="kappa1")
    CartesianOperators.cell_to_face_values!(sys.kappa_face1, sys.ops1, sys.kappa_cell1; averaging=sys.kappa_averaging)
    sys.diffusion_dirty = true
    return :matrix
end

function PenguinSolverCore.update!(upd::Kappa2Updater, sys::TwoPhaseDiffusionSystem{N,T}, u, p, t) where {N,T}
    payload = _evaluate_callable(upd.kappa_fun, sys, u, p, t)
    _set_two_phase_kappa_cell!(sys.kappa_cell2, sys.dof_omega2.indices, sys.ops2.Nd, payload; name="kappa2")
    CartesianOperators.cell_to_face_values!(sys.kappa_face2, sys.ops2, sys.kappa_cell2; averaging=sys.kappa_averaging)
    sys.diffusion_dirty = true
    return :matrix
end

function PenguinSolverCore.update!(upd::BoxDirichletUpdater1, sys::TwoPhaseDiffusionSystem, u, p, t)
    values = _evaluate_callable(upd.ufun, sys, u, p, t)
    changed = _update_box_dirichlet!(sys.ops1.bc, values)
    changed || return :nothing
    _refresh_two_phase_dirichlet_cache!(sys.dir_mask1, sys.dir_vals1, sys.dir_aff1, sys.ops1, sys.kappa_face1, sys.dof_omega1)
    return :rhs_only
end

function PenguinSolverCore.update!(upd::BoxDirichletUpdater2, sys::TwoPhaseDiffusionSystem, u, p, t)
    values = _evaluate_callable(upd.ufun, sys, u, p, t)
    changed = _update_box_dirichlet!(sys.ops2.bc, values)
    changed || return :nothing
    _refresh_two_phase_dirichlet_cache!(sys.dir_mask2, sys.dir_vals2, sys.dir_aff2, sys.ops2, sys.kappa_face2, sys.dof_omega2)
    return :rhs_only
end

function PenguinSolverCore.update!(upd::Source1Updater, sys::TwoPhaseDiffusionSystem, u, p, t)
    payload = _evaluate_callable(upd.source_fun, sys, u, p, t)
    if payload === nothing || payload isa Function
        sys.sourcefun1 = payload
    else
        fixed = payload
        sys.sourcefun1 = (_sys, _u, _p, _t) -> fixed
    end
    return :rhs_only
end

function PenguinSolverCore.update!(upd::Source2Updater, sys::TwoPhaseDiffusionSystem, u, p, t)
    payload = _evaluate_callable(upd.source_fun, sys, u, p, t)
    if payload === nothing || payload isa Function
        sys.sourcefun2 = payload
    else
        fixed = payload
        sys.sourcefun2 = (_sys, _u, _p, _t) -> fixed
    end
    return :rhs_only
end
