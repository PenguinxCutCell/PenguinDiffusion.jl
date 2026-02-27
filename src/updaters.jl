mutable struct RobinGUpdater{F} <: PenguinSolverCore.AbstractUpdater
    gfun::F
end

mutable struct RobinABUpdater{F} <: PenguinSolverCore.AbstractUpdater
    abfun::F
end

mutable struct BoxDirichletUpdater{F} <: PenguinSolverCore.AbstractUpdater
    ufun::F
end

mutable struct KappaUpdater{F} <: PenguinSolverCore.AbstractUpdater
    kappa_fun::F
end

mutable struct SourceUpdater{F} <: PenguinSolverCore.AbstractUpdater
    source_fun::F
end

function _set_dirichlet_payload!(payload::CartesianOperators.ScalarPayload, _value)
    throw(ArgumentError("cannot update Dirichlet ScalarPayload in-place; build BC with Dirichlet(Ref(value)) or Dirichlet(full_vector)"))
end

function _set_dirichlet_payload!(payload::CartesianOperators.RefPayload, value)
    value isa Number || throw(ArgumentError("RefPayload update expects a scalar, got $(typeof(value))"))
    CartesianOperators.set!(payload, value)
    return nothing
end

function _set_dirichlet_payload!(payload::CartesianOperators.VecPayload, value)
    if value isa Number || value isa AbstractVector
        CartesianOperators.set!(payload, value)
        return nothing
    end
    throw(ArgumentError("VecPayload update expects a scalar or full vector, got $(typeof(value))"))
end

function _update_side_dirichlet!(side_bc::CartesianOperators.AbstractBC, value)
    value === nothing && return false
    side_bc isa CartesianOperators.Dirichlet || return false
    _set_dirichlet_payload!((side_bc::CartesianOperators.Dirichlet).u, value)
    return true
end

function _update_side_group!(sides::Tuple, side_values)
    side_values === nothing && return false

    N = length(sides)
    changed = false
    if side_values isa Tuple
        length(side_values) == N || throw(DimensionMismatch("per-side tuple has length $(length(side_values)); expected $N"))
        @inbounds for d in 1:N
            changed |= _update_side_dirichlet!(sides[d], side_values[d])
        end
        return changed
    end

    @inbounds for d in 1:N
        changed |= _update_side_dirichlet!(sides[d], side_values)
    end
    return changed
end

function _update_box_dirichlet!(bc::CartesianOperators.BoxBC{N}, values) where {N}
    changed = false

    if values isa Number || values isa AbstractVector
        changed |= _update_side_group!(bc.lo, values)
        changed |= _update_side_group!(bc.hi, values)
        return changed
    end

    if values isa NamedTuple
        has_lo = haskey(values, :lo)
        has_hi = haskey(values, :hi)
        (has_lo || has_hi) || throw(ArgumentError("NamedTuple update must include :lo and/or :hi keys"))
        has_lo && (changed |= _update_side_group!(bc.lo, values.lo))
        has_hi && (changed |= _update_side_group!(bc.hi, values.hi))
        return changed
    end

    if values isa Tuple
        length(values) == 2 || throw(DimensionMismatch("tuple Dirichlet update must be (lo, hi)"))
        changed |= _update_side_group!(bc.lo, values[1])
        changed |= _update_side_group!(bc.hi, values[2])
        return changed
    end

    throw(ArgumentError("unsupported BoxDirichletUpdater payload $(typeof(values)); expected scalar, vector, NamedTuple, or (lo, hi) tuple"))
end

function PenguinSolverCore.update!(upd::RobinGUpdater, sys::DiffusionSystem, u, p, t)
    isempty(sys.r_gamma) && return :nothing
    # Convention: updater returns full/reduced/scalar g values.
    values = _evaluate_callable(upd.gfun, sys, u, p, t)
    _set_r_gamma!(sys, values)
    return :rhs_only
end

function _unpack_robin_ab_update(payload)
    if payload isa NamedTuple
        haskey(payload, :a) || throw(ArgumentError("RobinABUpdater NamedTuple payload must include key :a"))
        haskey(payload, :b) || throw(ArgumentError("RobinABUpdater NamedTuple payload must include key :b"))
        g = haskey(payload, :g) ? payload.g : nothing
        return payload.a, payload.b, g
    elseif payload isa Tuple
        if length(payload) == 2
            return payload[1], payload[2], nothing
        elseif length(payload) == 3
            return payload[1], payload[2], payload[3]
        end
        throw(DimensionMismatch("RobinABUpdater tuple payload must have length 2 (a,b) or 3 (a,b,g)"))
    end
    throw(ArgumentError("RobinABUpdater payload must be NamedTuple or tuple; got $(typeof(payload))"))
end

function PenguinSolverCore.update!(upd::RobinABUpdater, sys::DiffusionSystem, u, p, t)
    isempty(sys.r_gamma) && return :nothing

    payload = _evaluate_callable(upd.abfun, sys, u, p, t)
    aval, bval, gval = _unpack_robin_ab_update(payload)

    _set_interface_vector!(sys.interface.a, sys, aval; name="Robin a")
    _set_interface_vector!(sys.interface.b, sys, bval; name="Robin b")
    gval === nothing || _set_interface_vector!(sys.interface.g, sys, gval; name="Robin g")

    sys.constraints_dirty = true
    return :matrix
end

function PenguinSolverCore.update!(upd::BoxDirichletUpdater, sys::DiffusionSystem, u, p, t)
    values = _evaluate_callable(upd.ufun, sys, u, p, t)
    changed = _update_box_dirichlet!(sys.ops.bc, values)
    changed || return :nothing
    _refresh_dirichlet_cache!(sys)
    return :rhs_only
end

function PenguinSolverCore.update!(upd::KappaUpdater, sys::DiffusionSystem, u, p, t)
    sys.kappa = convert(typeof(sys.kappa), _evaluate_callable(upd.kappa_fun, sys, u, p, t))
    return :matrix
end

function PenguinSolverCore.update!(upd::SourceUpdater, sys::DiffusionSystem, u, p, t)
    payload = _evaluate_callable(upd.source_fun, sys, u, p, t)
    if payload === nothing || payload isa Function
        sys.sourcefun = payload
    else
        fixed = payload
        sys.sourcefun = (_sys, _u, _p, _t) -> fixed
    end
    return :rhs_only
end
