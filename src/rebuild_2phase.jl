function PenguinSolverCore.rebuild!(sys::TwoPhaseDiffusionSystem, u, p, t)
    if sys.diffusion_dirty
        _refresh_two_phase_diffusion_blocks!(sys)
        sys.diffusion_dirty = false
    end

    if sys.constraints_dirty
        _refresh_two_phase_constraints!(sys)
        sys.constraints_dirty = false

        if !isempty(sys.dof_gamma.indices)
            sys.Cγ_fact = lu(sys.Cγ)
        else
            sys.Cγ_fact = nothing
        end
    end

    sys.rebuild_calls += 1
    return nothing
end
