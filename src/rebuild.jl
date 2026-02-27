function PenguinSolverCore.rebuild!(sys::DiffusionSystem, u, p, t)
    if sys.diffusion_dirty
        _refresh_diffusion_blocks!(sys)
        sys.diffusion_dirty = false
    end

    if sys.constraints_dirty
        _refresh_constraint_blocks!(sys)
        sys.constraints_dirty = false

        if !isempty(sys.r_gamma)
            sys.C_gamma_fact = lu(sys.C_gamma)
        else
            sys.C_gamma_fact = nothing
        end
    end

    sys.rebuild_calls += 1
    return nothing
end
