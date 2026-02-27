using Test
using LinearAlgebra
using SparseArrays
using PenguinDiffusion
using CartesianGeometry
using CartesianOperators
using PenguinSolverCore

function build_cut_moments()
    x = collect(range(0.0, 1.0; length=8))
    y = collect(range(0.0, 1.0; length=7))
    levelset(x, y, _=0) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.28
    return CartesianGeometry.geometric_moments(levelset, (x, y), Float64, zero; method=:implicitintegration)
end

function build_test_system(; kappa=1.0, source=nothing, matrixfree_unsteady=false)
    moments = build_cut_moments()
    bc = CartesianOperators.BoxBC(Val(2), Float64)
    ops = CartesianOperators.assembled_ops(moments; bc=bc)

    a = ones(Float64, ops.Nd)
    b = zeros(Float64, ops.Nd)
    g = zeros(Float64, ops.Nd)
    interface = CartesianOperators.RobinConstraint(a, b, g)

    prob = PenguinDiffusion.DiffusionProblem(kappa, bc, interface, source)
    sys = PenguinDiffusion.build_system(moments, prob; matrixfree_unsteady=matrixfree_unsteady)
    length(sys.dof_gamma.indices) > 0 || error("test setup expected at least one interface DOF")
    return sys
end

function build_dirichlet_test_system(; kappa=1.0, source=nothing, ulo=1.5, uhi=2.0, matrixfree_unsteady=false)
    x = collect(range(0.0, 1.0; length=8))
    y = collect(range(0.0, 1.0; length=7))
    full_domain(x, y, _=0) = -1.0
    moments = CartesianGeometry.geometric_moments(full_domain, (x, y), Float64, zero; method=:implicitintegration)
    ulo_ref = Ref(Float64(ulo))
    uhi_ref = Ref(Float64(uhi))
    bc = CartesianOperators.BoxBC(
        (CartesianOperators.Dirichlet(ulo_ref), CartesianOperators.Neumann(0.0)),
        (CartesianOperators.Dirichlet(uhi_ref), CartesianOperators.Neumann(0.0)),
    )
    ops = CartesianOperators.assembled_ops(moments; bc=bc)
    interface = CartesianOperators.RobinConstraint(ones(Float64, ops.Nd), zeros(Float64, ops.Nd), zeros(Float64, ops.Nd))
    prob = PenguinDiffusion.DiffusionProblem(kappa, bc, interface, source)
    sys = PenguinDiffusion.build_system(moments, prob; matrixfree_unsteady=matrixfree_unsteady)
    return sys, ulo_ref, uhi_ref
end

include("test_reduction_contract.jl")
include("test_updates_and_rebuild.jl")
include("test_unsteady_matrixfree.jl")
include("test_sciml_integration.jl")
include("test_steady_solver.jl")
include("test_manufactured_boxbc.jl")
