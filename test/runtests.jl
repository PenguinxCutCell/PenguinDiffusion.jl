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

function build_test_system(; kappa=1.0, source=nothing)
    moments = build_cut_moments()
    bc = CartesianOperators.BoxBC(Val(2), Float64)
    ops = CartesianOperators.assembled_ops(moments; bc=bc)

    a = ones(Float64, ops.Nd)
    b = zeros(Float64, ops.Nd)
    g = zeros(Float64, ops.Nd)
    interface = CartesianOperators.RobinConstraint(a, b, g)

    prob = PenguinDiffusion.DiffusionProblem(kappa, bc, interface, source)
    sys = PenguinDiffusion.build_system(moments, prob)
    length(sys.dof_gamma.indices) > 0 || error("test setup expected at least one interface DOF")
    return sys
end

include("test_reduction_contract.jl")
include("test_updates_and_rebuild.jl")
include("test_sciml_integration.jl")
