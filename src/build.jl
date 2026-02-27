function padded_mask(dims::NTuple{N,Int}) where {N}
    nd = prod(dims)
    mask = falses(nd)
    li = LinearIndices(dims)
    @inbounds for I in CartesianIndices(dims)
        if any(d -> I[d] == dims[d], 1:N)
            mask[li[I]] = true
        end
    end
    return mask
end

function _constraint_matrices(
    ops::CartesianOperators.AssembledOps,
    interface::CartesianOperators.RobinConstraint,
)
    return CartesianOperators.robin_constraint_matrices(ops, interface)
end

function _constraint_matrices(::CartesianOperators.AssembledOps, interface)
    throw(ArgumentError("v0 supports only RobinConstraint; got $(typeof(interface))"))
end

function _normalize_robin_interface(interface::CartesianOperators.RobinConstraint, ::Type{T}) where {T}
    if interface isa CartesianOperators.RobinConstraint{T}
        return interface
    end
    return CartesianOperators.RobinConstraint(
        convert(Vector{T}, interface.a),
        convert(Vector{T}, interface.b),
        convert(Vector{T}, interface.g),
    )
end

function _normalize_robin_interface(interface, ::Type{T}) where {T}
    throw(ArgumentError("v0 supports only RobinConstraint; got $(typeof(interface))"))
end

function _refresh_constraint_blocks!(sys::DiffusionSystem{N,T}) where {N,T}
    C_omega_full, C_gamma_full, r_full = _constraint_matrices(sys.ops, sys.interface)
    idx_omega = sys.dof_omega.indices
    idx_gamma = sys.dof_gamma.indices

    sys.C_omega = C_omega_full[idx_gamma, idx_omega]
    sys.C_gamma = C_gamma_full[idx_gamma, idx_gamma]
    sys.r_gamma = collect(T.(r_full[idx_gamma]))
    return nothing
end

function _normalize_sourcefun(source)
    if source === nothing
        return nothing
    elseif source isa Function
        return source
    end

    fixed = source
    return (_sys, _u, _p, _t) -> fixed
end

function build_system(
    moments::CartesianGeometry.GeometricMoments{N,T},
    prob::DiffusionProblem;
    vtol::Union{Nothing,Real} = nothing,
    igamma_tol::Union{Nothing,Real} = nothing,
) where {N,T}
    ops = CartesianOperators.assembled_ops(moments; bc=prob.bc)
    interface = _normalize_robin_interface(prob.interface, T)

    V = T.(moments.V)
    maxV = maximum(abs, V; init=zero(T))
    vtol_local = vtol === nothing ? sqrt(eps(T)) * maxV : convert(T, vtol)

    omega_material_mask = falses(ops.Nd)
    @inbounds for i in eachindex(V)
        omega_material_mask[i] = (moments.cell_type[i] != 0) && (V[i] > vtol_local)
    end

    dir_mask, dir_vals_raw = CartesianOperators.dirichlet_mask_values(ops.dims, ops.bc)
    dir_vals = collect(T.(dir_vals_raw))
    pad_mask = padded_mask(ops.dims)

    omega_mask = copy(omega_material_mask)
    omega_mask .&= .!dir_mask
    omega_mask .&= .!pad_mask
    omega_active = findall(omega_mask)
    isempty(omega_active) && throw(ArgumentError("no active omega DOFs after masking (cell_type/V/padding/Dirichlet)"))

    maxIgamma = maximum(abs, ops.Iγ; init=zero(T))
    igamma_tol_local = igamma_tol === nothing ? sqrt(eps(T)) * maxIgamma : convert(T, igamma_tol)

    gamma_mask = falses(ops.Nd)
    @inbounds for i in eachindex(ops.Iγ)
        gamma_mask[i] = (ops.Iγ[i] > igamma_tol_local) && omega_mask[i]
    end
    gamma_active = findall(gamma_mask)

    dof_omega = PenguinSolverCore.DofMap(omega_active)
    dof_gamma = PenguinSolverCore.DofMap(gamma_active)

    L_oo_full = sparse(-ops.G' * ops.Winv * ops.G)
    L_og_full = sparse(-ops.G' * ops.Winv * ops.H)
    L_oo = L_oo_full[dof_omega.indices, dof_omega.indices]
    L_og = L_og_full[dof_omega.indices, dof_gamma.indices]

    C_omega_full, C_gamma_full, r_full = _constraint_matrices(ops, interface)
    C_omega = C_omega_full[dof_gamma.indices, dof_omega.indices]
    C_gamma = C_gamma_full[dof_gamma.indices, dof_gamma.indices]
    r_gamma = collect(T.(r_full[dof_gamma.indices]))

    C_gamma_fact = isempty(dof_gamma.indices) ? nothing : lu(C_gamma)

    M = spdiagm(0 => V[dof_omega.indices])
    b_full = CartesianOperators.dirichlet_rhs(ops)
    dirichlet_affine = collect(T.(b_full[dof_omega.indices]))

    sourcefun = _normalize_sourcefun(prob.source)

    return DiffusionSystem{N,T}(
        PenguinSolverCore.InvalidationCache(),
        PenguinSolverCore.UpdateManager(),
        moments,
        ops,
        interface,
        dof_omega,
        dof_gamma,
        M,
        L_oo,
        L_og,
        C_omega,
        C_gamma,
        r_gamma,
        C_gamma_fact,
        convert(T, prob.kappa),
        nothing,
        sourcefun,
        dir_mask,
        dir_vals,
        dirichlet_affine,
        zeros(T, length(dof_gamma.indices)),
        zeros(T, length(dof_omega.indices)),
        false,
        0,
    )
end
