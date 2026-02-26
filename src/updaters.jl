mutable struct RobinGUpdater{F} <: PenguinSolverCore.AbstractUpdater
    gfun::F
end

mutable struct KappaUpdater{F} <: PenguinSolverCore.AbstractUpdater
    kappa_fun::F
end

mutable struct SourceUpdater{F} <: PenguinSolverCore.AbstractUpdater
    source_fun::F
end

function PenguinSolverCore.update!(upd::RobinGUpdater, sys::DiffusionSystem, u, p, t)
    isempty(sys.r_gamma) && return :nothing
    # Convention: updater returns reduced r_gamma values (or a full vector/scalar that can be restricted).
    values = _evaluate_callable(upd.gfun, sys, u, p, t)
    _set_r_gamma!(sys, values)
    return :rhs_only
end

function PenguinSolverCore.update!(upd::KappaUpdater, sys::DiffusionSystem, u, p, t)
    sys.kappa = convert(eltype(sys.r_gamma), _evaluate_callable(upd.kappa_fun, sys, u, p, t))
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
