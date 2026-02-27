module PenguinDiffusion

using LinearAlgebra
using LinearSolve
using SparseArrays

using CartesianGeometry
using CartesianOperators
using PenguinSolverCore

include("types.jl")
include("build.jl")
include("reduction.jl")
include("unsteady_matrixfree.jl")
include("rhs.jl")
include("rebuild.jl")
include("io.jl")
include("updaters.jl")
include("steady.jl")

export DiffusionProblem
export DiffusionSystem
export build_system
export build_matrixfree_system
export enable_matrixfree_unsteady!
export full_state
export RobinGUpdater, RobinABUpdater, BoxDirichletUpdater, KappaUpdater, SourceUpdater
export steady_linear_problem, steady_solve

end
