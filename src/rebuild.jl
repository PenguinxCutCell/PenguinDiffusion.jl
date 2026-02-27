function PenguinSolverCore.rebuild!(sys::DiffusionSystem, u, p, t)
    if sys.constraints_dirty
        _refresh_constraint_blocks!(sys)
        sys.constraints_dirty = false
    end

    if !isempty(sys.r_gamma)
        sys.C_gamma_fact = lu(sys.C_gamma)
    else
        sys.C_gamma_fact = nothing
    end
    sys.rebuild_calls += 1
    return nothing
end
